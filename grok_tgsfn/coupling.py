# grok_tgsfn/coupling.py
# Affective Coupling: Neuromodulation and Prosociality
#
# Implements the "Affective Neuromodulation Field" from the Grok memo:
#
# Neuromodulation mappings:
#   V(t) → threshold_bias    (dopamine-like: good valence lowers threshold)
#   A(t) → noise_scale       (norepinephrine-like: high arousal = exploration)
#   D(t) → precision_weight  (acetylcholine-like: control affects attention)
#
# Curvature modulation:
#   appraisal → curvature_gain  (uncertainty/urgency → negative curvature)
#
# Prosocial coupling (empathy):
#   n_eff = n_self + w_empathy · n_human(inferred)
#
# This module computes control signals that the TGSFN substrate can use
# to modulate its dynamics based on affective state.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import torch
import torch.nn as nn

from .config import CouplingConfig


class AffectiveCoupler(nn.Module):
    """
    Affective Coupling: Maps internal state to neuromodulatory signals.

    The coupler translates high-level affective state (V, A, D from L1,
    appraisal from L2) into control signals that modulate the spiking
    substrate (TGSFN layer).

    Key outputs:
    1. threshold_bias: Shifts spike thresholds
       - Positive valence → lower threshold → more spiking
       - Negative valence → higher threshold → less spiking

    2. noise_scale: Controls exploration vs exploitation
       - High arousal → more noise → exploration
       - Low arousal → less noise → exploitation

    3. precision_weight: Self vs other attention weighting
       - High dominance → focus on self prediction
       - Low dominance → attend to external (social) signals

    4. curvature_gain: Modulates manifold curvature
       - High uncertainty/urgency → increase negative curvature
       - This affects how quickly embeddings diverge

    5. n_eff: Effective needs (prosocial coupling)
       - Agent feels the needs of coupled agent (empathy)
       - Creates prosocial motivation from homeostatic core
    """

    def __init__(self, config: CouplingConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Learnable empathy weight
        # w_empathy ∈ [0, 1] determines how much we feel others' needs
        self.w_empathy = nn.Parameter(
            torch.tensor(config.init_empathy_w, device=self.device)
        )

        # Scaling parameters (can be learned or fixed)
        self.threshold_scale = config.threshold_scale
        self.noise_min = config.noise_scale_min

    def forward(
        self,
        valence: torch.Tensor,            # V(t) from L1
        arousal: torch.Tensor,            # A(t) from L1
        dominance: torch.Tensor,          # D(t) from L1
        drive: torch.Tensor,              # d(t) from L1 (K,)
        needs_self: torch.Tensor,         # n_self from L1 (K,)
        needs_human: torch.Tensor,        # n_human inferred (K,)
        appraisal: torch.Tensor,          # a(t) from L2 (8,)
    ) -> Dict[str, torch.Tensor]:
        """
        Compute neuromodulatory control signals.

        Args:
            valence: Scalar or (B,) - pleasure/displeasure
            arousal: Scalar or (B,) - activation level
            dominance: Scalar or (B,) - control/agency
            drive: (K,) or (B, K) - drive vector
            needs_self: (K,) or (B, K) - agent's own needs
            needs_human: (K,) or (B, K) - inferred human's needs
            appraisal: (8,) or (B, 8) - cognitive appraisal

        Returns:
            Dict with:
                n_eff: Effective needs (with prosocial coupling)
                threshold_bias: Threshold modulation
                noise_scale: Exploration noise
                self_precision: Attention weight on self
                other_precision: Attention weight on other
                curvature_gain: Manifold curvature modulation
        """
        device = self.device

        # Ensure everything is on device
        valence = valence.to(device)
        arousal = arousal.to(device)
        dominance = dominance.to(device)
        needs_self = needs_self.to(device)
        needs_human = needs_human.to(device)
        appraisal = appraisal.to(device)

        # =====================================================================
        #  Prosocial Coupling: n_eff = n_self + w_empathy · n_human
        # =====================================================================

        # Clamp empathy weight to [0, 1]
        w_emp = torch.sigmoid(self.w_empathy)  # Ensure valid range

        n_eff = needs_self + w_emp * needs_human

        # =====================================================================
        #  Threshold Bias (dopamine-like)
        # =====================================================================

        # Good valence → lower threshold (more likely to spike)
        # Bad valence → higher threshold (more inhibited)
        threshold_bias = -self.threshold_scale * valence

        # =====================================================================
        #  Noise Scale (norepinephrine-like)
        # =====================================================================

        # High arousal → more noise → exploration
        # Clamp to minimum to avoid zero noise
        noise_scale = arousal.clamp(min=self.noise_min)

        # =====================================================================
        #  Precision Weights (acetylcholine-like)
        # =====================================================================

        # High dominance → more weight on self prediction (confident)
        # Low dominance → more weight on external/social signals
        self_precision = torch.sigmoid(dominance)
        other_precision = 1.0 - self_precision

        # =====================================================================
        #  Curvature Gain (appraisal-driven)
        # =====================================================================

        # Extract relevant appraisal dimensions
        # Indices: [p=0, r=1, c=2, ctrl=3, cop=4, u=5, ag=6, nrm=7]
        if appraisal.dim() == 1:
            certainty = appraisal[2]
            urgency = appraisal[5]
        else:
            certainty = appraisal[:, 2]
            urgency = appraisal[:, 5]

        # High uncertainty (low certainty) + high urgency → increase curvature
        # This makes representations more spread out (uncertain)
        curvature_gain = torch.sigmoid(urgency - certainty)

        return {
            "n_eff": n_eff,
            "threshold_bias": threshold_bias,
            "noise_scale": noise_scale,
            "self_precision": self_precision,
            "other_precision": other_precision,
            "curvature_gain": curvature_gain,
            "empathy_weight": w_emp,
        }

    def apply_to_threshold(
        self,
        base_threshold: torch.Tensor,
        threshold_bias: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply threshold bias to base thresholds.

        Args:
            base_threshold: (N,) base spike thresholds
            threshold_bias: Scalar or (B,) bias from coupling

        Returns:
            Modified threshold (clamped to positive)
        """
        return (base_threshold + threshold_bias).clamp(min=0.1)

    def apply_to_noise(
        self,
        base_noise: torch.Tensor,
        noise_scale: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply noise scaling to base noise.

        Args:
            base_noise: Base noise tensor
            noise_scale: Scalar or (B,) scale factor

        Returns:
            Scaled noise
        """
        return base_noise * noise_scale

    def compute_coupled_reward(
        self,
        reward_self: float,
        reward_other: float,
    ) -> float:
        """
        Compute reward with prosocial coupling.

        r_coupled = r_self + w_empathy · r_other

        This makes the agent care about the other's reward.
        """
        w_emp = torch.sigmoid(self.w_empathy).item()
        return reward_self + w_emp * reward_other

    def get_coupling_state(self) -> Dict[str, float]:
        """Get current coupling parameters for logging."""
        return {
            "empathy_weight": torch.sigmoid(self.w_empathy).item(),
        }


class MultiAgentCoupler(AffectiveCoupler):
    """
    Extended coupler for multiple coupled agents.

    Supports coupling to multiple external agents with different
    empathy weights per agent.
    """

    def __init__(self, config: CouplingConfig, num_agents: int = 1):
        super().__init__(config)
        self.num_agents = num_agents

        # Per-agent empathy weights
        self.w_empathy = nn.Parameter(
            torch.full((num_agents,), config.init_empathy_w, device=self.device)
        )

    def forward_multi(
        self,
        valence: torch.Tensor,
        arousal: torch.Tensor,
        dominance: torch.Tensor,
        drive: torch.Tensor,
        needs_self: torch.Tensor,
        needs_others: torch.Tensor,  # (num_agents, K)
        appraisal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward with multiple coupled agents.

        Args:
            needs_others: (num_agents, K) - needs of all coupled agents

        Returns:
            Same as forward, but n_eff includes all agents
        """
        device = self.device
        needs_self = needs_self.to(device)
        needs_others = needs_others.to(device)

        # Weighted sum of others' needs
        w_emp = torch.sigmoid(self.w_empathy)  # (num_agents,)
        weighted_others = torch.einsum('a,ak->k', w_emp, needs_others)

        n_eff = needs_self + weighted_others

        # Get other signals from base method
        base_result = super().forward(
            valence, arousal, dominance, drive,
            needs_self, torch.zeros_like(needs_self), appraisal
        )

        base_result["n_eff"] = n_eff
        base_result["empathy_weights"] = w_emp

        return base_result


if __name__ == "__main__":
    print("=== AffectiveCoupler Test ===")

    config = CouplingConfig(
        init_empathy_w=0.5,
        threshold_scale=0.1,
        noise_scale_min=0.01,
        device="cpu",
    )
    coupler = AffectiveCoupler(config)

    # Test forward pass
    print("\n--- Single agent coupling ---")
    K = 4  # Number of needs

    result = coupler(
        valence=torch.tensor(0.3),
        arousal=torch.tensor(0.7),
        dominance=torch.tensor(0.5),
        drive=torch.randn(K),
        needs_self=torch.tensor([0.2, 0.5, 0.3, 0.1]),
        needs_human=torch.tensor([0.6, 0.2, 0.4, 0.3]),
        appraisal=torch.randn(8),
    )

    print(f"Empathy weight: {result['empathy_weight'].item():.4f}")
    print(f"Threshold bias: {result['threshold_bias'].item():.4f}")
    print(f"Noise scale: {result['noise_scale'].item():.4f}")
    print(f"Self precision: {result['self_precision'].item():.4f}")
    print(f"Curvature gain: {result['curvature_gain'].item():.4f}")
    print(f"n_eff: {result['n_eff'].tolist()}")

    # Verify prosocial coupling
    print("\n--- Prosocial coupling verification ---")
    n_self = torch.tensor([0.2, 0.5, 0.3, 0.1])
    n_human = torch.tensor([0.6, 0.2, 0.4, 0.3])
    w = torch.sigmoid(coupler.w_empathy).item()
    expected_n_eff = n_self + w * n_human
    actual_n_eff = result["n_eff"]
    print(f"Expected n_eff: {expected_n_eff.tolist()}")
    print(f"Actual n_eff: {actual_n_eff.tolist()}")
    print(f"Match: {torch.allclose(expected_n_eff, actual_n_eff)}")

    # Test coupled reward
    print("\n--- Coupled reward test ---")
    r_self = 0.5
    r_other = -0.2
    r_coupled = coupler.compute_coupled_reward(r_self, r_other)
    print(f"r_self={r_self}, r_other={r_other}, r_coupled={r_coupled:.4f}")

    print("\nAffectiveCoupler test passed!")
