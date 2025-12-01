# tfan_agent/gating.py
# Layer 3: Gating / Adaptive Policy Controller
#
# Maps internal state to adaptive gating parameters that modulate:
#   - Policy temperature (exploration vs exploitation)
#   - Learning rate scaling (adapt faster or slower)
#   - Memory write probability (what's worth remembering)
#   - Auxiliary gain (general modulation factor)
#
# Inputs:
#   - V, A, D (from L1 Homeostat)
#   - Epistemic & Aleatoric uncertainty estimates
#   - Appraisal vector (from L2)
#
# This is the "neuromodulatory" layer that implements arousal-gated
# attention, uncertainty-driven exploration, and affect-based learning.

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GatingConfig:
    """
    Configuration for Layer 3: Gating / Policy Controller.

    Inputs:
        - V, A, D           -> 3 scalars (from L1 PAD)
        - epistemic, aleatoric uncertainty -> 2 scalars
        - appraisal vector  -> appraisal_dim dims (from L2)

    Outputs:
        - temperature       -> [0, +∞)   (softmax temperature for policy)
        - lr_scale          -> (0, 2)    (learning rate multiplier)
        - mem_write_p       -> (0, 1)    (probability of writing to memory)
        - aux_gain          -> (0, 1)    (general purpose modulation)
    """
    appraisal_dim: int = 8
    hidden_dim: int = 64
    device: str = "cpu"


class GatingControllerL3(nn.Module):
    """
    L3: Adaptive Policy & Gating Controller.

    This module implements neuromodulatory-style gating that adapts
    agent behavior based on internal state and uncertainty.

    Key gating outputs:

    1. Temperature: Controls exploration vs exploitation
       - High arousal + high epistemic uncertainty -> high temp (explore)
       - Low arousal + low uncertainty -> low temp (exploit)

    2. LR Scale: Adapts learning speed
       - High surprise/novelty -> faster learning
       - Familiar situations -> slower, stable learning

    3. Memory Write Probability: What's worth remembering?
       - High arousal events get prioritized
       - Routine events may be forgotten

    4. Aux Gain: General purpose modulation
       - Can be used for attention, output scaling, etc.

    Phase 1: Simple MLP mapping internal state -> gates
    Phase 2: Will add LC (locus coeruleus) dynamics, dopamine-like signals
    """

    def __init__(self, config: GatingConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Input: V,A,D + epistemic + aleatoric + appraisal
        in_dim = 3 + 2 + config.appraisal_dim
        hidden = config.hidden_dim

        # Shared trunk
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ).to(self.device)

        # Separate heads for each gate (allows different parameterizations)
        self.temp_head = nn.Linear(hidden, 1).to(self.device)
        self.lr_head = nn.Linear(hidden, 1).to(self.device)
        self.mem_head = nn.Linear(hidden, 1).to(self.device)
        self.aux_head = nn.Linear(hidden, 1).to(self.device)

        # Initialize heads for reasonable default values
        self._init_heads()

    def _init_heads(self):
        """Initialize heads to produce reasonable default gate values."""
        # Temperature: default ~1.0
        nn.init.zeros_(self.temp_head.weight)
        nn.init.constant_(self.temp_head.bias, 0.5)  # softplus(0.5) ≈ 0.97

        # LR scale: default ~1.0
        nn.init.zeros_(self.lr_head.weight)
        nn.init.zeros_(self.lr_head.bias)  # 2*sigmoid(0) = 1.0

        # Memory write: default ~0.5
        nn.init.zeros_(self.mem_head.weight)
        nn.init.zeros_(self.mem_head.bias)

        # Aux: default ~0.5
        nn.init.zeros_(self.aux_head.weight)
        nn.init.zeros_(self.aux_head.bias)

    def forward(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        dominance: torch.Tensor,
        epistemic_unc: torch.Tensor,
        aleatoric_unc: torch.Tensor,
        appraisal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gating parameters from internal state.

        Inputs may be scalars or batched. We broadcast scalars to batch size.

        Args:
            valence: (B,) or scalar - pleasure/displeasure from L1
            arousal: (B,) or scalar - activation level from L1
            dominance: (B,) or scalar - control/agency from L1
            epistemic_unc: (B,) or scalar - reducible uncertainty
            aleatoric_unc: (B,) or scalar - irreducible uncertainty
            appraisal: (B, appraisal_dim) - cognitive appraisal from L2

        Returns:
            dict with keys:
                "temperature": (B, 1) - policy softmax temperature
                "lr_scale": (B, 1) - learning rate multiplier
                "mem_write_p": (B, 1) - memory write probability
                "aux_gain": (B, 1) - auxiliary modulation gain
        """
        B = appraisal.size(0)
        device = self.device

        def _to_batch(x):
            """Convert scalar or 1D tensor to (B, 1) batch."""
            if not torch.is_tensor(x):
                x = torch.tensor([x], device=device)
            if x.dim() == 0:
                return x.view(1, 1).expand(B, -1).to(device)
            elif x.dim() == 1:
                if x.size(0) == 1:
                    return x.view(1, 1).expand(B, -1).to(device)
                return x.view(-1, 1).to(device)
            else:
                return x.to(device)

        v = _to_batch(valence)
        a = _to_batch(arousal)
        d = _to_batch(dominance)
        e = _to_batch(epistemic_unc)
        al = _to_batch(aleatoric_unc)

        # Concatenate all inputs
        x = torch.cat([v, a, d, e, al, appraisal.to(device)], dim=-1)

        # Shared feature extraction
        h = self.net(x)

        # Temperature: R -> (0, +inf) via softplus + epsilon
        # High arousal + high epistemic -> explore (high temp)
        temperature = F.softplus(self.temp_head(h)) + 1e-3

        # LR scale: R -> (0, 2) via sigmoid scaling
        # High novelty/surprise -> faster learning
        lr_scale = 2.0 * torch.sigmoid(self.lr_head(h))

        # Memory write probability: R -> (0, 1) via sigmoid
        # High arousal -> more likely to write
        mem_write_p = torch.sigmoid(self.mem_head(h))

        # Aux gain: (0, 1) general purpose
        aux_gain = torch.sigmoid(self.aux_head(h))

        return {
            "temperature": temperature,
            "lr_scale": lr_scale,
            "mem_write_p": mem_write_p,
            "aux_gain": aux_gain,
        }

    def get_gate_summary(self, gates: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Get scalar summary of gates for logging.
        """
        return {
            "temperature": float(gates["temperature"].mean().item()),
            "lr_scale": float(gates["lr_scale"].mean().item()),
            "mem_write_p": float(gates["mem_write_p"].mean().item()),
            "aux_gain": float(gates["aux_gain"].mean().item()),
        }


class LCModulatedGating(GatingControllerL3):
    """
    Extended gating with Locus Coeruleus (LC) dynamics.

    The LC is the brain's norepinephrine system, which modulates:
    - Arousal and alertness
    - Attention gain
    - Exploration vs exploitation balance

    This variant adds temporal dynamics to the gating, simulating
    the phasic/tonic modes of LC firing.

    (Optional enhancement for Phase 1.5)
    """

    def __init__(self, config: GatingConfig, tau_lc: float = 0.9):
        super().__init__(config)
        self.tau_lc = tau_lc

        # LC state (tonic firing rate)
        self.register_buffer(
            "lc_tonic",
            torch.zeros(1, device=torch.device(config.device))
        )

    def reset_lc(self):
        """Reset LC state between episodes."""
        self.lc_tonic.zero_()

    def forward(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        dominance: torch.Tensor,
        epistemic_unc: torch.Tensor,
        aleatoric_unc: torch.Tensor,
        appraisal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward with LC modulation."""
        # Get base gates
        gates = super().forward(
            valence, arousal, dominance,
            epistemic_unc, aleatoric_unc, appraisal
        )

        # Update LC tonic state based on arousal
        if torch.is_tensor(arousal):
            arousal_mean = arousal.mean()
        else:
            arousal_mean = torch.tensor(arousal, device=self.device)

        self.lc_tonic = self.tau_lc * self.lc_tonic + (1 - self.tau_lc) * arousal_mean

        # Modulate temperature by LC state
        # High tonic LC -> more exploration
        lc_factor = 1.0 + 0.5 * self.lc_tonic
        gates["temperature"] = gates["temperature"] * lc_factor

        # Add LC state to output for logging
        gates["lc_tonic"] = self.lc_tonic.expand(gates["temperature"].size(0), 1)

        return gates


if __name__ == "__main__":
    # Quick sanity check
    print("=== GatingControllerL3 Sanity Check ===")

    config = GatingConfig(appraisal_dim=8, hidden_dim=32, device="cpu")
    gating = GatingControllerL3(config)

    # Test forward pass
    B = 4
    valence = torch.tensor([-0.5])
    arousal = torch.tensor([0.8])
    dominance = torch.tensor([-0.2])
    epistemic = torch.tensor([0.6])
    aleatoric = torch.tensor([0.3])
    appraisal = torch.randn(B, 8)

    gates = gating(valence, arousal, dominance, epistemic, aleatoric, appraisal)

    print(f"Temperature shape: {gates['temperature'].shape}, mean={gates['temperature'].mean():.3f}")
    print(f"LR scale shape: {gates['lr_scale'].shape}, mean={gates['lr_scale'].mean():.3f}")
    print(f"Mem write p shape: {gates['mem_write_p'].shape}, mean={gates['mem_write_p'].mean():.3f}")
    print(f"Aux gain shape: {gates['aux_gain'].shape}, mean={gates['aux_gain'].mean():.3f}")

    # Check ranges
    assert (gates["temperature"] > 0).all(), "Temperature should be positive"
    assert (gates["lr_scale"] > 0).all() and (gates["lr_scale"] < 2).all(), "LR scale should be in (0, 2)"
    assert (gates["mem_write_p"] > 0).all() and (gates["mem_write_p"] < 1).all(), "Mem write should be in (0, 1)"

    summary = gating.get_gate_summary(gates)
    print(f"Gate summary: {summary}")

    print("\nGatingControllerL3 sanity check passed!")
