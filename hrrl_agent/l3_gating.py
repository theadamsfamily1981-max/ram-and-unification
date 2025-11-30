"""
Layer 3: MLP Gating Controller

Learned gating over affect and appraisal signals.

Input: [V, A, D, appraisal, epistemic, aleatoric]
Output: τ (temperature), lr_scale, mem_write_p, empathy_w

Unlike the pure-functional gating in grok_tgsfn, this uses an MLP
to learn the mapping, allowing more flexible adaptation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import NamedTuple, List

from .config import L3Config


class GatingOutputs(NamedTuple):
    """Outputs from the gating controller."""
    tau: torch.Tensor  # Temperature/exploration parameter [0, 1]
    lr_scale: torch.Tensor  # Learning rate scaling [lr_min, lr_max]
    mem_write_p: torch.Tensor  # Memory write probability [0, 1]
    empathy_w: torch.Tensor  # Empathy/prosocial weight [0, 1]
    att_gain: torch.Tensor  # Attention gain [0, 2]
    raw_logits: torch.Tensor  # Pre-activation outputs for analysis


class GatingControllerL3(nn.Module):
    """
    MLP-based Gating Controller.

    Takes affect (PAD) and cognitive appraisal as input,
    outputs gating signals that modulate learning, memory,
    and behavior.

    Input dimensions:
    - V, A, D: 3 (from L1 homeostat)
    - appraisal: appraisal_dim (from L2)
    - epistemic, aleatoric: 2 (from L2)
    Total: 3 + appraisal_dim + 2
    """

    def __init__(self, config: L3Config, appraisal_dim: int = 8):
        super().__init__()
        self.config = config
        self.appraisal_dim = appraisal_dim

        # Input: [V, A, D, appraisal, epistemic, aleatoric]
        input_dim = 3 + appraisal_dim + 2

        # Build MLP
        layers = []
        prev_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)

        # Output heads (5 outputs)
        self.num_outputs = 5
        self.output_head = nn.Linear(prev_dim, self.num_outputs)

        # Store config bounds
        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        self.lr_scale_min = config.lr_scale_min
        self.lr_scale_max = config.lr_scale_max

    def forward(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        dominance: torch.Tensor,
        appraisal: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor
    ) -> GatingOutputs:
        """
        Compute gating signals.

        Args:
            valence: Pleasure signal [...] or [..., 1]
            arousal: Activation signal
            dominance: Control signal
            appraisal: Cognitive appraisal [..., appraisal_dim]
            epistemic: Epistemic uncertainty
            aleatoric: Aleatoric uncertainty

        Returns:
            GatingOutputs with all gating signals
        """
        # Ensure proper shapes
        if valence.dim() == 0:
            valence = valence.unsqueeze(0)
        if arousal.dim() == 0:
            arousal = arousal.unsqueeze(0)
        if dominance.dim() == 0:
            dominance = dominance.unsqueeze(0)
        if epistemic.dim() == 0:
            epistemic = epistemic.unsqueeze(0)
        if aleatoric.dim() == 0:
            aleatoric = aleatoric.unsqueeze(0)

        # Stack PAD
        if valence.dim() == 1:
            pad = torch.stack([valence, arousal, dominance], dim=-1)
        else:
            pad = torch.cat([valence, arousal, dominance], dim=-1)

        # Ensure uncertainty dims match
        if epistemic.dim() == 1:
            unc = torch.stack([epistemic, aleatoric], dim=-1)
        else:
            unc = torch.cat([epistemic, aleatoric], dim=-1)

        # Concatenate all inputs
        x = torch.cat([pad, appraisal, unc], dim=-1)

        # Forward through MLP
        features = self.backbone(x)
        raw_logits = self.output_head(features)

        # Split and apply activations
        # tau: temperature [tau_min, tau_max]
        tau = torch.sigmoid(raw_logits[..., 0])
        tau = self.tau_min + tau * (self.tau_max - self.tau_min)

        # lr_scale: learning rate scaling [lr_min, lr_max]
        lr_scale = torch.sigmoid(raw_logits[..., 1])
        lr_scale = self.lr_scale_min + lr_scale * (self.lr_scale_max - self.lr_scale_min)

        # mem_write_p: memory write probability [0, 1]
        mem_write_p = torch.sigmoid(raw_logits[..., 2])

        # empathy_w: empathy/prosocial weight [0, 1]
        empathy_w = torch.sigmoid(raw_logits[..., 3])

        # att_gain: attention gain [0, 2]
        att_gain = 2.0 * torch.sigmoid(raw_logits[..., 4])

        return GatingOutputs(
            tau=tau,
            lr_scale=lr_scale,
            mem_write_p=mem_write_p,
            empathy_w=empathy_w,
            att_gain=att_gain,
            raw_logits=raw_logits
        )


class ResidualGatingController(nn.Module):
    """
    Gating controller with residual connections and explicit
    affect-to-output pathways.

    Combines learned MLP with explicit functional mappings
    (like those in grok_tgsfn) via residual addition.
    """

    def __init__(self, config: L3Config, appraisal_dim: int = 8):
        super().__init__()
        self.config = config
        self.appraisal_dim = appraisal_dim

        input_dim = 3 + appraisal_dim + 2

        # Learned component
        self.learned = GatingControllerL3(config, appraisal_dim)

        # Explicit functional weights (learnable but initialized to theory)
        # tau = σ(3 - 2A + U_epi)
        self.tau_explicit = nn.Linear(3, 1, bias=True)
        self.tau_explicit.weight.data = torch.tensor([[-2.0, 0.0, 1.0]])  # [A, D, U_epi]
        self.tau_explicit.bias.data = torch.tensor([3.0])

        # lr_scale = σ(D + control - A + U_epi)
        self.lr_explicit = nn.Linear(4, 1, bias=True)
        self.lr_explicit.weight.data = torch.tensor([[1.0, -1.0, 1.0, 0.0]])  # [D, A, U_epi, ...]
        self.lr_explicit.bias.data = torch.tensor([0.0])

        # mem_write_p = σ(3V + 2r + U_epi - A)
        self.mem_explicit = nn.Linear(3, 1, bias=True)
        self.mem_explicit.weight.data = torch.tensor([[3.0, -1.0, 1.0]])  # [V, A, U_epi]
        self.mem_explicit.bias.data = torch.tensor([0.0])

        # Mixing weight between learned and explicit
        self.mix_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        dominance: torch.Tensor,
        appraisal: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor
    ) -> GatingOutputs:
        """Forward with residual explicit pathway."""
        # Get learned outputs
        learned_out = self.learned(
            valence, arousal, dominance,
            appraisal, epistemic, aleatoric
        )

        # Compute explicit outputs
        # Ensure proper shapes
        if valence.dim() == 0:
            valence = valence.unsqueeze(0)
        if arousal.dim() == 0:
            arousal = arousal.unsqueeze(0)
        if dominance.dim() == 0:
            dominance = dominance.unsqueeze(0)
        if epistemic.dim() == 0:
            epistemic = epistemic.unsqueeze(0)

        tau_in = torch.stack([arousal, dominance, epistemic], dim=-1)
        tau_explicit = torch.sigmoid(self.tau_explicit(tau_in)).squeeze(-1)

        lr_in = torch.stack([dominance, arousal, epistemic, aleatoric], dim=-1)
        lr_explicit = torch.sigmoid(self.lr_explicit(lr_in)).squeeze(-1)
        lr_explicit = self.config.lr_scale_min + lr_explicit * (
            self.config.lr_scale_max - self.config.lr_scale_min
        )

        mem_in = torch.stack([valence, arousal, epistemic], dim=-1)
        mem_explicit = torch.sigmoid(self.mem_explicit(mem_in)).squeeze(-1)

        # Mix learned and explicit
        alpha = torch.sigmoid(self.mix_alpha)

        tau = alpha * learned_out.tau + (1 - alpha) * tau_explicit
        lr_scale = alpha * learned_out.lr_scale + (1 - alpha) * lr_explicit
        mem_write_p = alpha * learned_out.mem_write_p + (1 - alpha) * mem_explicit

        # These are purely learned (no explicit formula)
        empathy_w = learned_out.empathy_w
        att_gain = learned_out.att_gain

        return GatingOutputs(
            tau=tau,
            lr_scale=lr_scale,
            mem_write_p=mem_write_p,
            empathy_w=empathy_w,
            att_gain=att_gain,
            raw_logits=learned_out.raw_logits
        )


class GatingControllerWithHistory(nn.Module):
    """
    Gating controller that also considers temporal history
    via an LSTM.
    """

    def __init__(
        self,
        config: L3Config,
        appraisal_dim: int = 8,
        hidden_size: int = 64
    ):
        super().__init__()
        self.config = config

        input_dim = 3 + appraisal_dim + 2

        # LSTM for temporal context
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Linear(32, 5)  # 5 gating outputs
        )

        self.tau_min = config.tau_min
        self.tau_max = config.tau_max
        self.lr_scale_min = config.lr_scale_min
        self.lr_scale_max = config.lr_scale_max

        # Hidden state
        self.hidden = None

    def reset_hidden(self, batch_size: int = 1, device: torch.device = None):
        """Reset LSTM hidden state."""
        if device is None:
            device = next(self.parameters()).device

        self.hidden = (
            torch.zeros(1, batch_size, self.lstm.hidden_size, device=device),
            torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        )

    def forward(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        dominance: torch.Tensor,
        appraisal: torch.Tensor,
        epistemic: torch.Tensor,
        aleatoric: torch.Tensor
    ) -> GatingOutputs:
        """Forward with temporal context."""
        # Build input
        if valence.dim() == 0:
            valence = valence.unsqueeze(0)
        if arousal.dim() == 0:
            arousal = arousal.unsqueeze(0)
        if dominance.dim() == 0:
            dominance = dominance.unsqueeze(0)
        if epistemic.dim() == 0:
            epistemic = epistemic.unsqueeze(0)
        if aleatoric.dim() == 0:
            aleatoric = aleatoric.unsqueeze(0)

        pad = torch.stack([valence, arousal, dominance], dim=-1)
        unc = torch.stack([epistemic, aleatoric], dim=-1)
        x = torch.cat([pad, appraisal, unc], dim=-1)

        # Add sequence dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B, 1, D]

        # Initialize hidden if needed
        batch_size = x.size(0)
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            self.reset_hidden(batch_size, x.device)

        # LSTM forward
        lstm_out, self.hidden = self.lstm(x, self.hidden)

        # Take last output
        features = lstm_out[:, -1, :]  # [B, hidden]

        # Output head
        raw_logits = self.output_head(features)

        # Apply activations
        tau = torch.sigmoid(raw_logits[..., 0])
        tau = self.tau_min + tau * (self.tau_max - self.tau_min)

        lr_scale = torch.sigmoid(raw_logits[..., 1])
        lr_scale = self.lr_scale_min + lr_scale * (self.lr_scale_max - self.lr_scale_min)

        mem_write_p = torch.sigmoid(raw_logits[..., 2])
        empathy_w = torch.sigmoid(raw_logits[..., 3])
        att_gain = 2.0 * torch.sigmoid(raw_logits[..., 4])

        return GatingOutputs(
            tau=tau.squeeze(0) if tau.size(0) == 1 else tau,
            lr_scale=lr_scale.squeeze(0) if lr_scale.size(0) == 1 else lr_scale,
            mem_write_p=mem_write_p.squeeze(0) if mem_write_p.size(0) == 1 else mem_write_p,
            empathy_w=empathy_w.squeeze(0) if empathy_w.size(0) == 1 else empathy_w,
            att_gain=att_gain.squeeze(0) if att_gain.size(0) == 1 else att_gain,
            raw_logits=raw_logits
        )


if __name__ == "__main__":
    # Test gating controllers
    print("Testing GatingControllerL3...")

    config = L3Config()
    gating = GatingControllerL3(config, appraisal_dim=8)

    # Single sample
    V = torch.tensor(0.5)
    A = torch.tensor(0.7)
    D = torch.tensor(0.3)
    appraisal = torch.randn(8)
    epistemic = torch.tensor(0.2)
    aleatoric = torch.tensor(0.1)

    out = gating(V, A, D, appraisal, epistemic, aleatoric)
    print(f"  tau: {out.tau.item():.4f}")
    print(f"  lr_scale: {out.lr_scale.item():.4f}")
    print(f"  mem_write_p: {out.mem_write_p.item():.4f}")
    print(f"  empathy_w: {out.empathy_w.item():.4f}")
    print(f"  att_gain: {out.att_gain.item():.4f}")

    # Batch
    print("\nBatched forward...")
    V = torch.rand(4)
    A = torch.rand(4)
    D = torch.rand(4)
    appraisal = torch.randn(4, 8)
    epistemic = torch.rand(4)
    aleatoric = torch.rand(4)

    out = gating(V, A, D, appraisal, epistemic, aleatoric)
    print(f"  tau shape: {out.tau.shape}")

    # Test residual controller
    print("\nTesting ResidualGatingController...")
    residual = ResidualGatingController(config, appraisal_dim=8)
    out_res = residual(V, A, D, appraisal, epistemic, aleatoric)
    print(f"  tau: {out_res.tau}")

    # Test history-aware controller
    print("\nTesting GatingControllerWithHistory...")
    history = GatingControllerWithHistory(config, appraisal_dim=8)
    for t in range(3):
        out_hist = history(
            V[0], A[0], D[0],
            appraisal[0], epistemic[0], aleatoric[0]
        )
        print(f"  Step {t}: tau={out_hist.tau.item():.4f}")
