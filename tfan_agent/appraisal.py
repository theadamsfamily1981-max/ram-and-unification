# tfan_agent/appraisal.py
# Layer 2: Cognitive Appraisal Engine
#
# Implements cognitive appraisal theory (Scherer, Lazarus) in neural form:
#   - Maps [observation, needs, goals] -> 8-D appraisal vector
#   - Appraisal dimensions capture the "meaning" of the situation
#   - Phase 1: Simple MLP (Euclidean)
#   - Phase 2: Will be replaced by hyperbolic/LLM-based appraisal
#
# The 8 appraisal dimensions (CoRE-inspired):
#   0: novelty/expectedness
#   1: intrinsic pleasantness
#   2: goal conduciveness
#   3: coping potential
#   4: norm compatibility
#   5: self-relevance
#   6: certainty
#   7: urgency

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


# Appraisal dimension names for reference
APPRAISAL_DIMS = [
    "novelty",           # 0: How unexpected/novel is this?
    "pleasantness",      # 1: Intrinsic hedonic tone
    "goal_conducive",    # 2: Does this help or hinder goals?
    "coping_potential",  # 3: Can I handle this?
    "norm_compatible",   # 4: Does this fit social/moral norms?
    "self_relevance",    # 5: How important is this to me?
    "certainty",         # 6: How certain am I about this?
    "urgency",           # 7: How urgent is action?
]


@dataclass
class AppraisalConfig:
    """
    Configuration for Layer 2: Cognitive Appraisal Engine.

    The appraisal head maps:
        [obs, needs, goals] -> 8-D appraisal vector

    Each dimension is in [-1, 1] after tanh activation.
    """
    obs_dim: int
    num_needs: int
    goal_dim: int = 0
    hidden_dim: int = 128
    appraisal_dim: int = 8
    device: str = "cpu"


class AppraisalHeadL2(nn.Module):
    """
    L2: Cognitive Appraisal Engine (Phase 1: Euclidean MLP).

    This module learns to assess situations along multiple dimensions
    that are relevant for emotional response and behavioral regulation.

    The appraisal vector feeds into:
    - L3 Gating Controller (for adaptive policy modulation)
    - Emotion classification (optional downstream)
    - Memory encoding prioritization

    In later phases this can be replaced by:
        - Hyperbolic embeddings for hierarchical appraisal
        - EMA-style process model with temporal dynamics
        - LLM-based explicit appraisal reasoning
    """

    def __init__(self, config: AppraisalConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        in_dim = config.obs_dim + config.num_needs + config.goal_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.appraisal_dim),
        ).to(self.device)

        # Initialize output layer with small weights for stable initial appraisals
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.1)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        obs: torch.Tensor,
        needs: torch.Tensor,
        goals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute appraisal vector from observation, needs, and goals.

        Args:
            obs:   (B, obs_dim) - external observation
            needs: (B, num_needs) or (num_needs,) - current need state from L1
            goals: (B, goal_dim) or None - optional goal representation

        Returns:
            appraisal: (B, appraisal_dim) - bounded in [-1, 1] via tanh
        """
        # Handle unbatched needs
        if needs.dim() == 1:
            needs = needs.unsqueeze(0).expand(obs.size(0), -1)

        # Handle missing or unbatched goals
        if goals is None and self.config.goal_dim > 0:
            goals = torch.zeros(
                obs.size(0), self.config.goal_dim, device=self.device
            )
        elif goals is not None and goals.dim() == 1:
            goals = goals.unsqueeze(0).expand(obs.size(0), -1)

        # Concatenate inputs
        if self.config.goal_dim > 0 and goals is not None:
            x = torch.cat([obs.to(self.device), needs.to(self.device), goals.to(self.device)], dim=-1)
        else:
            x = torch.cat([obs.to(self.device), needs.to(self.device)], dim=-1)

        # Forward through network
        appraisal = self.net(x)

        # Bound to [-1, 1] for stable downstream processing
        appraisal = torch.tanh(appraisal)

        return appraisal

    def get_appraisal_dict(self, appraisal: torch.Tensor) -> dict:
        """
        Convert appraisal tensor to named dictionary for logging.

        Args:
            appraisal: (appraisal_dim,) or (1, appraisal_dim)

        Returns:
            Dict mapping dimension names to float values
        """
        if appraisal.dim() == 2:
            appraisal = appraisal.squeeze(0)

        return {
            name: float(appraisal[i].item())
            for i, name in enumerate(APPRAISAL_DIMS[:len(appraisal)])
        }


class AppraisalWithAttention(nn.Module):
    """
    Extended appraisal head with attention over needs.

    This variant computes attention weights over needs before
    aggregating, allowing the model to focus on the most relevant
    needs for the current observation.

    (Optional enhancement for Phase 1.5)
    """

    def __init__(self, config: AppraisalConfig):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.ReLU(),
        ).to(self.device)

        # Need encoder (per-need)
        self.need_encoder = nn.Sequential(
            nn.Linear(1, config.hidden_dim // 4),
            nn.ReLU(),
        ).to(self.device)

        # Attention over needs
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=4,
            batch_first=True,
        ).to(self.device)

        # Project attended needs to hidden dim
        self.need_proj = nn.Linear(
            config.num_needs * (config.hidden_dim // 4),
            config.hidden_dim
        ).to(self.device)

        # Final appraisal head
        self.appraisal_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.appraisal_dim),
        ).to(self.device)

    def forward(
        self,
        obs: torch.Tensor,
        needs: torch.Tensor,
        goals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute appraisal with attention over needs.
        """
        B = obs.size(0)
        if needs.dim() == 1:
            needs = needs.unsqueeze(0).expand(B, -1)

        # Encode observation
        obs_enc = self.obs_encoder(obs.to(self.device))  # (B, hidden)

        # Encode each need separately
        needs_enc = []
        for i in range(self.config.num_needs):
            need_i = needs[:, i:i+1].to(self.device)  # (B, 1)
            enc_i = self.need_encoder(need_i)  # (B, hidden//4)
            needs_enc.append(enc_i)
        needs_enc = torch.cat(needs_enc, dim=-1)  # (B, num_needs * hidden//4)

        # Project needs
        needs_proj = self.need_proj(needs_enc)  # (B, hidden)

        # Combine and compute appraisal
        combined = torch.cat([obs_enc, needs_proj], dim=-1)  # (B, hidden*2)
        appraisal = self.appraisal_head(combined)
        appraisal = torch.tanh(appraisal)

        return appraisal


if __name__ == "__main__":
    # Quick sanity check
    print("=== AppraisalHeadL2 Sanity Check ===")

    config = AppraisalConfig(
        obs_dim=8,
        num_needs=4,
        goal_dim=0,
        hidden_dim=64,
        device="cpu",
    )
    appraisal_head = AppraisalHeadL2(config)

    # Test forward pass
    obs = torch.randn(2, 8)
    needs = torch.tensor([0.5, 0.2, 0.8, 0.1])

    appraisal = appraisal_head(obs, needs)
    print(f"Appraisal shape: {appraisal.shape}")
    print(f"Appraisal range: [{appraisal.min().item():.3f}, {appraisal.max().item():.3f}]")

    # Get named appraisal
    named = appraisal_head.get_appraisal_dict(appraisal[0])
    print(f"Named appraisal: {named}")

    print("\nAppraisalHeadL2 sanity check passed!")
