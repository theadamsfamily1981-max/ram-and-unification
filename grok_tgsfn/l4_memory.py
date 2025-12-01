# grok_tgsfn/l4_memory.py
# Layer 4: Memory Consolidation, Sleep, and Personalization
#
# Implements the memory-related equations from the Grok memo:
#
# 1. Salience:
#    sal(t) = ||d(t)|| · ||a(t)|| + β · U_epi(t)
#    High drive × high appraisal + epistemic uncertainty = salient
#
# 2. Online Memory Write Probability:
#    mem_write_p(t) = σ(3V + 2·relevance + urgency + U_epi - A)
#    (This is duplicated from L3 for self-contained memory logic)
#
# 3. Replay Distribution:
#    p_replay(i) ∝ sal_i · exp(-λ_diss Π_q(i)) · exp(-λ_id Δidentity(i))
#    Replay high-salience, low-dissipation, identity-consistent experiences
#
# 4. Homeostatic Personalization Regularizer:
#    ℒ_homeo = λ_homeo · ½ ||Δn||²_Σ
#    Penalize adaptations that cause large homeostatic shifts

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import torch
import torch.nn as nn

from .config import L4Config


@dataclass
class MemoryItem:
    """
    One stored experience for replay scheduling.

    All fields are scalars (not tensors) for storage efficiency.
    """
    drive_norm: float            # ||d(t)|| at storage time
    appraisal_norm: float        # ||a(t)|| at storage time
    U_epi: float                 # Epistemic uncertainty at storage time
    valence: float               # V(t) at storage time
    relevance: float             # Appraisal relevance dimension
    urgency: float               # Appraisal urgency dimension
    arousal: float               # A(t) at storage time
    Pi_q: float                  # Thermodynamic cost for this experience
    delta_identity: float        # Δidentity(i) - hyperbolic distance measure
    timestamp: int = 0           # Step number when stored

    # Optional: actual experience data (tensors stored separately)
    experience_id: int = -1      # Index into experience buffer


class MemoryL4(nn.Module):
    """
    L4 - Memory Consolidation, Sleep, and Personalization.

    Manages:
    - Computing salience for experiences
    - Deciding what to write to memory (online filtering)
    - Computing replay distributions for consolidation
    - Homeostatic regularization for personalization

    The memory system implements a biologically-inspired approach where:
    - High-salience experiences are prioritized for storage
    - Replay probability balances salience, thermodynamic cost, and identity
    - Personalization is constrained to maintain homeostatic stability
    """

    def __init__(self, config: L4Config):
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)

        # Memory buffer (list of MemoryItem for now)
        self.memory: List[MemoryItem] = []
        self.max_capacity = config.memory_capacity

    # =========================================================================
    #  Salience Computation
    # =========================================================================

    def salience(
        self,
        drive_norm: torch.Tensor,      # ||d(t)||
        appraisal_norm: torch.Tensor,  # ||a(t)||
        U_epi: torch.Tensor,           # Epistemic uncertainty
    ) -> torch.Tensor:
        """
        Compute salience for an experience.

        sal(t) = ||d(t)|| · ||a(t)|| + β · U_epi(t)

        High salience experiences are:
        - High drive (important for homeostasis)
        - High appraisal (emotionally significant)
        - High epistemic uncertainty (learning opportunity)

        Args:
            drive_norm: (B,) or scalar
            appraisal_norm: (B,) or scalar
            U_epi: (B,) or scalar

        Returns:
            salience: (B,) or scalar
        """
        beta = self.config.beta_epistemic
        return drive_norm * appraisal_norm + beta * U_epi

    def salience_from_state(
        self,
        drive: torch.Tensor,       # d(t) vector (K,)
        appraisal: torch.Tensor,   # a(t) vector (8,)
        U_epi: float,              # Scalar uncertainty
    ) -> float:
        """
        Convenience method to compute scalar salience from state vectors.
        """
        d_norm = drive.norm(p=2).item()
        a_norm = appraisal.norm(p=2).item()
        return d_norm * a_norm + self.config.beta_epistemic * U_epi

    # =========================================================================
    #  Memory Write Probability
    # =========================================================================

    def mem_write_probability(
        self,
        valence: torch.Tensor,         # V(t)
        relevance: torch.Tensor,       # Appraisal relevance dimension
        urgency: torch.Tensor,         # Appraisal urgency dimension
        U_epi: torch.Tensor,           # Epistemic uncertainty
        arousal: torch.Tensor,         # A(t)
    ) -> torch.Tensor:
        """
        Compute probability of writing experience to memory.

        mem_write_p(t) = σ(3V + 2·relevance + urgency + U_epi - A)

        High write probability when:
        - Positive valence (good experiences)
        - High relevance and urgency (important)
        - High epistemic uncertainty (novel)
        - Low arousal (not too busy to encode)

        Args:
            All inputs (B,) or scalar

        Returns:
            probability (B,) in (0, 1)
        """
        logits = 3.0 * valence + 2.0 * relevance + urgency + U_epi - arousal
        return torch.sigmoid(logits)

    # =========================================================================
    #  Replay Distribution
    # =========================================================================

    def replay_distribution(
        self,
        saliences: torch.Tensor,       # (N,) salience values
        Pi_q: torch.Tensor,            # (N,) thermodynamic costs
        delta_identity: torch.Tensor,  # (N,) identity distances
    ) -> torch.Tensor:
        """
        Compute replay probability distribution over stored experiences.

        p_replay(i) ∝ sal_i · exp(-λ_diss Π_q(i)) · exp(-λ_id Δidentity(i))

        This prioritizes:
        - High salience experiences (sal_i)
        - Low thermodynamic cost (exp(-λ_diss Π_q))
        - Identity-consistent experiences (exp(-λ_id Δidentity))

        Args:
            saliences: (N,) salience values for N experiences
            Pi_q: (N,) thermodynamic costs
            delta_identity: (N,) hyperbolic identity distances

        Returns:
            probabilities: (N,) summing to 1
        """
        lam_diss = self.config.lambda_diss
        lam_id = self.config.lambda_id

        # Compute log weights for numerical stability
        log_weights = (
            torch.log(saliences.clamp_min(1e-8))
            - lam_diss * Pi_q
            - lam_id * delta_identity
        )

        # Softmax normalization
        weights = torch.exp(log_weights - log_weights.max())
        probs = weights / weights.sum().clamp_min(1e-8)

        return probs

    def sample_replay(
        self,
        n_samples: int,
        current_identity: Optional[torch.Tensor] = None,
    ) -> List[MemoryItem]:
        """
        Sample experiences from memory for replay.

        Args:
            n_samples: Number of experiences to sample
            current_identity: Optional current identity embedding for Δidentity

        Returns:
            List of MemoryItem samples
        """
        if len(self.memory) == 0:
            return []

        n_samples = min(n_samples, len(self.memory))

        # Extract values from memory
        saliences = torch.tensor([m.drive_norm * m.appraisal_norm +
                                 self.config.beta_epistemic * m.U_epi
                                 for m in self.memory])
        pi_qs = torch.tensor([m.Pi_q for m in self.memory])
        delta_ids = torch.tensor([m.delta_identity for m in self.memory])

        # Compute replay distribution
        probs = self.replay_distribution(saliences, pi_qs, delta_ids)

        # Sample indices
        indices = torch.multinomial(probs, n_samples, replacement=False)

        return [self.memory[i] for i in indices.tolist()]

    # =========================================================================
    #  Memory Management
    # =========================================================================

    def add_experience(
        self,
        drive_norm: float,
        appraisal_norm: float,
        U_epi: float,
        valence: float,
        relevance: float,
        urgency: float,
        arousal: float,
        Pi_q: float,
        delta_identity: float,
        timestamp: int,
        write_prob_threshold: float = 0.5,
    ) -> bool:
        """
        Potentially add an experience to memory.

        Uses the memory write probability to filter experiences.

        Returns:
            True if experience was added, False otherwise
        """
        # Compute write probability
        prob = self.mem_write_probability(
            torch.tensor(valence),
            torch.tensor(relevance),
            torch.tensor(urgency),
            torch.tensor(U_epi),
            torch.tensor(arousal),
        ).item()

        if prob < write_prob_threshold:
            return False

        # Create memory item
        item = MemoryItem(
            drive_norm=drive_norm,
            appraisal_norm=appraisal_norm,
            U_epi=U_epi,
            valence=valence,
            relevance=relevance,
            urgency=urgency,
            arousal=arousal,
            Pi_q=Pi_q,
            delta_identity=delta_identity,
            timestamp=timestamp,
        )

        # Add to memory (with capacity management)
        if len(self.memory) >= self.max_capacity:
            # Remove lowest salience item
            sal = [m.drive_norm * m.appraisal_norm +
                   self.config.beta_epistemic * m.U_epi
                   for m in self.memory]
            min_idx = sal.index(min(sal))
            self.memory.pop(min_idx)

        self.memory.append(item)
        return True

    # =========================================================================
    #  Homeostatic Personalization Regularizer
    # =========================================================================

    def homeostatic_regularizer(
        self,
        delta_n: torch.Tensor,        # Δn induced by adapter (K,)
        inv_sigma_diag: torch.Tensor, # Σ⁻¹ diagonal (K,)
    ) -> torch.Tensor:
        """
        Compute homeostatic regularization loss.

        ℒ_homeo = λ_homeo · ½ ||Δn||²_Σ = λ_homeo · ½ Δn^T Σ⁻¹ Δn

        This penalizes personalization updates that cause large
        shifts in homeostatic needs, preserving core identity.

        Args:
            delta_n: Change in needs induced by adaptation (K,)
            inv_sigma_diag: Inverse precision diagonal (K,)

        Returns:
            Scalar regularization loss
        """
        weighted_sq = delta_n * inv_sigma_diag * delta_n
        return 0.5 * self.config.lambda_homeo * weighted_sq.sum()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get statistics about memory contents."""
        if len(self.memory) == 0:
            return {
                "count": 0,
                "mean_salience": 0.0,
                "mean_pi_q": 0.0,
                "mean_valence": 0.0,
            }

        saliences = [m.drive_norm * m.appraisal_norm +
                    self.config.beta_epistemic * m.U_epi
                    for m in self.memory]

        return {
            "count": len(self.memory),
            "mean_salience": sum(saliences) / len(saliences),
            "mean_pi_q": sum(m.Pi_q for m in self.memory) / len(self.memory),
            "mean_valence": sum(m.valence for m in self.memory) / len(self.memory),
        }


if __name__ == "__main__":
    print("=== MemoryL4 Test ===")

    config = L4Config(
        beta_epistemic=1.0,
        lambda_diss=0.1,
        lambda_id=0.1,
        lambda_homeo=0.1,
        memory_capacity=100,
        device="cpu",
    )
    memory = MemoryL4(config)

    # Test salience computation
    print("\n--- Salience test ---")
    sal = memory.salience(
        drive_norm=torch.tensor(0.5),
        appraisal_norm=torch.tensor(0.8),
        U_epi=torch.tensor(0.3),
    )
    print(f"Salience: {sal.item():.4f}")

    # Test memory write probability
    print("\n--- Memory write probability test ---")
    prob = memory.mem_write_probability(
        valence=torch.tensor(0.5),
        relevance=torch.tensor(0.7),
        urgency=torch.tensor(0.6),
        U_epi=torch.tensor(0.3),
        arousal=torch.tensor(0.4),
    )
    print(f"Write probability: {prob.item():.4f}")

    # Add some experiences
    print("\n--- Adding experiences ---")
    for i in range(20):
        added = memory.add_experience(
            drive_norm=0.3 + 0.1 * (i % 5),
            appraisal_norm=0.5 + 0.05 * i,
            U_epi=0.2 + 0.02 * i,
            valence=0.1 + 0.03 * i - 0.5 * (i % 3 == 0),
            relevance=0.5,
            urgency=0.3,
            arousal=0.4,
            Pi_q=0.1 * i,
            delta_identity=0.05 * i,
            timestamp=i,
            write_prob_threshold=0.3,
        )
    print(f"Memory stats: {memory.get_memory_stats()}")

    # Test replay distribution
    print("\n--- Replay distribution test ---")
    if len(memory.memory) > 0:
        samples = memory.sample_replay(n_samples=5)
        print(f"Sampled {len(samples)} experiences")
        for s in samples:
            print(f"  t={s.timestamp}, valence={s.valence:.3f}, Pi_q={s.Pi_q:.3f}")

    # Test homeostatic regularizer
    print("\n--- Homeostatic regularizer test ---")
    delta_n = torch.randn(8) * 0.1
    inv_sigma = torch.ones(8)
    reg_loss = memory.homeostatic_regularizer(delta_n, inv_sigma)
    print(f"Regularization loss: {reg_loss.item():.6f}")

    print("\nMemoryL4 test passed!")
