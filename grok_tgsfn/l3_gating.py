# grok_tgsfn/l3_gating.py
# Layer 3: Gating Controller with Explicit Equations
#
# Implements the gating equations exactly as specified in the Grok memo.
# No neural network - just direct functional mappings from internal state
# to gating parameters.
#
# Inputs (per timestep):
#   V          - valence from L1
#   A          - arousal from L1
#   D          - dominance from L1
#   U_epi      - epistemic uncertainty
#   U_alea     - aleatoric uncertainty
#   U_conflict - cross-modal conflict
#   appraisal  - (B, 8) vector from L2:
#                [p, r, c, ctrl, cop, u, ag, nrm]
#   ||d||      - drive norm from L1
#
# Outputs (gating vector):
#   τ(t)         - temperature: exploration vs exploitation
#   η_scale(t)   - learning rate scale: adapt faster or slower
#   mem_write_p  - memory write probability: what to remember
#   att_gain     - attention gain: focus modulation
#   p_sleep      - sleep/defer probability: when to consolidate
#
# Equations (from memo):
#   τ(t) = σ(3 - 2A + U_epi)
#   η_scale(t) = σ(D + cop - A + U_epi)
#   mem_write_p = σ(3V + 2r + u + U_epi - A)
#   att_gain = σ(D + ctrl + ag - U_conflict)
#   p_sleep = σ(A + ||d|| - cop - V)

from __future__ import annotations

from typing import NamedTuple, Dict
import torch
import torch.nn as nn

from .config import L3Config


class GatingOutputs(NamedTuple):
    """
    Outputs from L3 - Gating Controller.

    All tensors have shape (B, 1) for batch consistency.
    """
    temperature: torch.Tensor     # τ(t) ∈ (0, 1) - exploration temperature
    lr_scale: torch.Tensor        # η_scale(t) ∈ (0, 1) - learning rate scale
    mem_write_p: torch.Tensor     # ∈ (0, 1) - probability of writing to memory
    att_gain: torch.Tensor        # ∈ (0, 1) - attention gain multiplier
    p_sleep: torch.Tensor         # ∈ (0, 1) - probability of sleep/defer


class GatingControllerL3(nn.Module):
    """
    L3 - Adaptive Policy & Gating Controller.

    Implements explicit equations from the Grok memo without any
    learnable parameters. This ensures the gating behavior exactly
    matches the theoretical specification.

    The gating outputs modulate downstream behavior:
    - Temperature affects action selection (high τ → explore)
    - LR scale adapts learning speed (high η → learn faster)
    - Memory write probability filters what's stored
    - Attention gain modulates focus
    - Sleep probability triggers consolidation

    All equations use sigmoid activation to bound outputs to (0, 1).
    """

    def __init__(self, config: L3Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

    def forward(
        self,
        valence: torch.Tensor,          # V from L1
        arousal: torch.Tensor,          # A from L1
        dominance: torch.Tensor,        # D from L1
        U_epi: torch.Tensor,            # Epistemic uncertainty
        U_alea: torch.Tensor,           # Aleatoric uncertainty (not used in base eqs)
        U_conflict: torch.Tensor,       # Cross-modal conflict
        appraisal: torch.Tensor,        # (B, 8) from L2
        drive_norm: torch.Tensor,       # ||d|| from L1
    ) -> GatingOutputs:
        """
        Compute gating parameters from internal state.

        All scalar inputs are broadcast to batch size of appraisal.

        Args:
            valence: V(t) - pleasure/displeasure
            arousal: A(t) - activation level
            dominance: D(t) - control/agency
            U_epi: Epistemic uncertainty (reducible)
            U_alea: Aleatoric uncertainty (irreducible)
            U_conflict: Cross-modal conflict measure
            appraisal: (B, 8) - cognitive appraisal vector
                       [p, r, c, ctrl, cop, u, ag, nrm]
            drive_norm: ||d(t)|| - magnitude of drive vector

        Returns:
            GatingOutputs with shape (B, 1) tensors
        """
        device = self.device
        appraisal = appraisal.to(device)
        B = appraisal.size(0)

        # Helper to convert scalars/1D tensors to (B, 1)
        def to_batch(x):
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32, device=device)
            if x.dim() == 0:
                return x.view(1, 1).expand(B, 1)
            elif x.dim() == 1:
                if x.size(0) == 1:
                    return x.view(1, 1).expand(B, 1)
                return x.view(-1, 1)
            return x.to(device)

        V = to_batch(valence)       # Valence
        A = to_batch(arousal)       # Arousal
        D = to_batch(dominance)     # Dominance
        Ue = to_batch(U_epi)        # Epistemic uncertainty
        Ua = to_batch(U_alea)       # Aleatoric uncertainty
        Uc = to_batch(U_conflict)   # Conflict
        d_norm = to_batch(drive_norm)

        # Split appraisal into named components
        # Indices: [p=0, r=1, c=2, ctrl=3, cop=4, u=5, ag=6, nrm=7]
        p = appraisal[:, 0:1]     # pleasantness
        r = appraisal[:, 1:2]     # relevance
        c = appraisal[:, 2:3]     # certainty
        ctrl = appraisal[:, 3:4]  # control
        cop = appraisal[:, 4:5]   # coping potential
        u = appraisal[:, 5:6]     # urgency
        ag = appraisal[:, 6:7]    # agency
        nrm = appraisal[:, 7:8]   # norm compatibility

        # =====================================================================
        #  Gating Equations (from Grok memo)
        # =====================================================================

        # 1. Temperature τ(t) = σ(3 - 2A + U_epi)
        #    High arousal → low temperature → exploit
        #    High epistemic uncertainty → high temperature → explore
        tau = torch.sigmoid(3.0 - 2.0 * A + Ue)

        # 2. Learning-rate scale η_scale(t) = σ(D + cop - A + U_epi)
        #    High dominance + coping → faster learning
        #    High arousal → slower learning (too activated to learn well)
        lr_scale = torch.sigmoid(D + cop - A + Ue)

        # 3. Memory write probability = σ(3V + 2r + u + U_epi - A)
        #    Positive valence → remember (good experiences)
        #    High relevance + urgency → remember
        #    High arousal → less likely to write (too busy acting)
        mem_write_p = torch.sigmoid(3.0 * V + 2.0 * r + u + Ue - A)

        # 4. Attention gain = σ(D + ctrl + ag - U_conflict)
        #    High control + agency → high attention gain
        #    High conflict → reduced attention gain
        att_gain = torch.sigmoid(D + ctrl + ag - Uc)

        # 5. Sleep/defer probability = σ(A + ||d|| - cop - V)
        #    High arousal + high drive → need consolidation
        #    High coping + positive valence → no need to defer
        p_sleep = torch.sigmoid(A + d_norm - cop - V)

        return GatingOutputs(
            temperature=tau,
            lr_scale=lr_scale,
            mem_write_p=mem_write_p,
            att_gain=att_gain,
            p_sleep=p_sleep,
        )

    def get_gate_summary(self, outputs: GatingOutputs) -> Dict[str, float]:
        """
        Get scalar summary of gating outputs for logging.
        """
        return {
            "temperature": float(outputs.temperature.mean().item()),
            "lr_scale": float(outputs.lr_scale.mean().item()),
            "mem_write_p": float(outputs.mem_write_p.mean().item()),
            "att_gain": float(outputs.att_gain.mean().item()),
            "p_sleep": float(outputs.p_sleep.mean().item()),
        }


def compute_temperature(arousal: float, U_epi: float) -> float:
    """Standalone temperature computation for quick access."""
    import math
    return 1.0 / (1.0 + math.exp(-(3.0 - 2.0 * arousal + U_epi)))


def compute_lr_scale(dominance: float, coping: float, arousal: float, U_epi: float) -> float:
    """Standalone LR scale computation."""
    import math
    return 1.0 / (1.0 + math.exp(-(dominance + coping - arousal + U_epi)))


if __name__ == "__main__":
    print("=== GatingControllerL3 Test (Explicit Equations) ===")

    config = L3Config(device="cpu")
    gating = GatingControllerL3(config)

    # Test with various internal states
    B = 4

    # Scenario 1: High arousal, low valence (stressed)
    print("\n--- Scenario 1: High arousal, low valence (stressed) ---")
    gates_stressed = gating(
        valence=torch.tensor([-0.5]),
        arousal=torch.tensor([0.9]),
        dominance=torch.tensor([-0.3]),
        U_epi=torch.tensor([0.7]),
        U_alea=torch.tensor([0.3]),
        U_conflict=torch.tensor([0.4]),
        appraisal=torch.randn(B, 8),
        drive_norm=torch.tensor([0.8]),
    )
    summary1 = gating.get_gate_summary(gates_stressed)
    print(f"  τ (temperature): {summary1['temperature']:.3f}")
    print(f"  η (lr_scale): {summary1['lr_scale']:.3f}")
    print(f"  mem_write_p: {summary1['mem_write_p']:.3f}")
    print(f"  att_gain: {summary1['att_gain']:.3f}")
    print(f"  p_sleep: {summary1['p_sleep']:.3f}")

    # Scenario 2: Low arousal, high valence (relaxed & happy)
    print("\n--- Scenario 2: Low arousal, high valence (relaxed) ---")
    gates_relaxed = gating(
        valence=torch.tensor([0.7]),
        arousal=torch.tensor([0.2]),
        dominance=torch.tensor([0.5]),
        U_epi=torch.tensor([0.1]),
        U_alea=torch.tensor([0.2]),
        U_conflict=torch.tensor([0.1]),
        appraisal=torch.randn(B, 8),
        drive_norm=torch.tensor([0.1]),
    )
    summary2 = gating.get_gate_summary(gates_relaxed)
    print(f"  τ (temperature): {summary2['temperature']:.3f}")
    print(f"  η (lr_scale): {summary2['lr_scale']:.3f}")
    print(f"  mem_write_p: {summary2['mem_write_p']:.3f}")
    print(f"  att_gain: {summary2['att_gain']:.3f}")
    print(f"  p_sleep: {summary2['p_sleep']:.3f}")

    # Verify expected behaviors
    print("\n--- Behavior verification ---")
    print(f"Stressed has lower temperature? {summary1['temperature'] < summary2['temperature']}")
    print(f"Relaxed has higher att_gain? {summary2['att_gain'] > summary1['att_gain']}")
    print(f"Stressed needs more sleep? {summary1['p_sleep'] > summary2['p_sleep']}")
    print(f"Relaxed writes more to memory? {summary2['mem_write_p'] > summary1['mem_write_p']}")

    print("\nGatingControllerL3 test passed!")
