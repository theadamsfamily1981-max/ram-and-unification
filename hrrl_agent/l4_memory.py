"""
Layer 4: Memory & Personalization

Implements:
- Salience computation: sal = ||d|| · ||a|| + β U_epi
- Replay distribution with factors (salience, Π_q, identity distance)
- LoRA adapters for personalization
- Homeostatic rejection: reject update if F_int rises too much
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import math
import logging

from .config import L4Config

logger = logging.getLogger(__name__)


@dataclass
class MemoryEntry:
    """Single memory entry with associated metadata."""
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool

    # Affective metadata
    salience: float
    valence: float
    arousal: float
    epistemic: float

    # For replay distribution
    pi_q: float  # Entropy production at storage time
    identity_distance: float  # Distance from identity at storage time

    # Timestamp
    step: int


class ReplayBuffer:
    """
    Experience replay buffer with salience-weighted sampling.

    Stores experiences and samples according to replay distribution:
    p_replay(i) ∝ sal_i · exp(-λ_diss · Π_q) · exp(-λ_id · Δidentity)
    """

    def __init__(self, config: L4Config):
        self.config = config
        self.capacity = config.buffer_capacity
        self.buffer: List[MemoryEntry] = []
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, entry: MemoryEntry):
        """Add entry to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(entry)
        else:
            self.buffer[self.position] = entry

        self.position = (self.position + 1) % self.capacity

    def compute_salience(
        self,
        drive_norm: float,
        appraisal_norm: float,
        epistemic: float
    ) -> float:
        """
        Compute salience for memory storage.

        sal = ||d|| · ||a|| + β U_epi
        """
        return drive_norm * appraisal_norm + self.config.beta_epistemic * epistemic

    def compute_replay_distribution(self) -> torch.Tensor:
        """
        Compute replay sampling distribution.

        p_replay(i) ∝ sal_i · exp(-λ_diss · Π_q) · exp(-λ_id · Δidentity)
        """
        if len(self.buffer) == 0:
            return torch.tensor([])

        saliences = torch.tensor([e.salience for e in self.buffer])
        pi_qs = torch.tensor([e.pi_q for e in self.buffer])
        id_dists = torch.tensor([e.identity_distance for e in self.buffer])

        # Log weights for numerical stability
        log_weights = (
            torch.log(saliences + 1e-8)
            - self.config.lambda_dissipation * pi_qs
            - self.config.lambda_identity * id_dists
        )

        # Apply temperature
        log_weights = log_weights / self.config.replay_temperature

        # Softmax to get distribution
        probs = F.softmax(log_weights, dim=0)

        return probs

    def sample(self, batch_size: int) -> List[MemoryEntry]:
        """Sample batch according to replay distribution."""
        if len(self.buffer) < self.config.min_samples_for_replay:
            return []

        probs = self.compute_replay_distribution()
        indices = torch.multinomial(probs, batch_size, replacement=True)

        return [self.buffer[i] for i in indices.tolist()]

    def sample_uniform(self, batch_size: int) -> List[MemoryEntry]:
        """Sample uniformly (for comparison)."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        indices = torch.randint(0, len(self.buffer), (batch_size,))
        return [self.buffer[i] for i in indices.tolist()]

    def get_statistics(self) -> Dict:
        """Get buffer statistics."""
        if len(self.buffer) == 0:
            return {}

        saliences = [e.salience for e in self.buffer]
        rewards = [e.reward for e in self.buffer]

        return {
            'size': len(self.buffer),
            'mean_salience': sum(saliences) / len(saliences),
            'max_salience': max(saliences),
            'mean_reward': sum(rewards) / len(rewards),
        }


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for efficient personalization.

    W' = W + α/r · BA

    Where B is [out, r], A is [r, in], and r << min(in, out).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (start from base model)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute LoRA delta: Δ = (α/r) · x @ A^T @ B^T
        """
        x = self.dropout(x)
        # x @ A^T gives [batch, rank]
        # then @ B^T gives [batch, out]
        return self.scaling * (x @ self.lora_A.T @ self.lora_B.T)

    def merge_into(self, linear: nn.Linear) -> None:
        """Merge LoRA weights into base linear layer."""
        with torch.no_grad():
            delta = self.scaling * (self.lora_B @ self.lora_A)
            linear.weight.add_(delta)

    def get_delta_weight(self) -> torch.Tensor:
        """Get the weight delta for analysis."""
        return self.scaling * (self.lora_B @ self.lora_A)


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.

    Combines frozen base weights with trainable low-rank adaptation.
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()

        # Freeze base weights
        self.base = base_linear
        for param in self.base.parameters():
            param.requires_grad = False

        # LoRA adaptation
        self.lora = LoRALayer(
            base_linear.in_features,
            base_linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with base + LoRA."""
        return self.base(x) + self.lora(x)

    def merge(self) -> nn.Linear:
        """Merge LoRA into base and return merged layer."""
        self.lora.merge_into(self.base)
        return self.base


class HomeostaticRejectionGate:
    """
    Gate that rejects updates if they would increase F_int too much.

    Implements "reject update if F_int rises too much" from spec.
    """

    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Maximum allowed increase in F_int
        """
        self.threshold = threshold
        self.rejection_count = 0
        self.total_count = 0

    def should_accept(
        self,
        f_int_before: float,
        f_int_after: float
    ) -> bool:
        """
        Check if update should be accepted.

        Returns True if ΔF_int <= threshold.
        """
        self.total_count += 1
        delta_f = f_int_after - f_int_before

        if delta_f > self.threshold:
            self.rejection_count += 1
            logger.debug(
                f"Rejecting update: ΔF_int={delta_f:.4f} > {self.threshold}"
            )
            return False

        return True

    def get_rejection_rate(self) -> float:
        """Get fraction of rejected updates."""
        if self.total_count == 0:
            return 0.0
        return self.rejection_count / self.total_count

    def reset_stats(self):
        """Reset rejection statistics."""
        self.rejection_count = 0
        self.total_count = 0


class PersonalizationModule(nn.Module):
    """
    Full personalization module with LoRA adapters and homeostatic gating.

    Wraps a base model and adds:
    - LoRA adapters for efficient personalization
    - Homeostatic rejection gating
    - Personalization regularization
    """

    def __init__(
        self,
        base_model: nn.Module,
        config: L4Config,
        target_layers: Optional[List[str]] = None
    ):
        """
        Args:
            base_model: Model to adapt
            config: L4 configuration
            target_layers: Names of layers to add LoRA to (None = all Linear)
        """
        super().__init__()
        self.config = config
        self.base_model = base_model

        # Add LoRA to target layers
        self.lora_layers: Dict[str, LoRALayer] = {}
        self._add_lora_adapters(target_layers)

        # Rejection gate
        self.rejection_gate = HomeostaticRejectionGate(
            threshold=config.f_int_rejection_threshold
        )

        # Track original LoRA state for regularization
        self._save_initial_lora_state()

    def _add_lora_adapters(self, target_layers: Optional[List[str]]):
        """Add LoRA adapters to specified layers."""
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if target_layers is None or name in target_layers:
                    # Create LoRA layer
                    lora = LoRALayer(
                        module.in_features,
                        module.out_features,
                        rank=self.config.lora_rank,
                        alpha=self.config.lora_alpha,
                        dropout=self.config.lora_dropout
                    )
                    self.lora_layers[name] = lora
                    # Register as submodule
                    self.add_module(f"lora_{name.replace('.', '_')}", lora)

    def _save_initial_lora_state(self):
        """Save initial LoRA state for regularization."""
        self._initial_lora_state = {
            name: {
                'A': lora.lora_A.clone().detach(),
                'B': lora.lora_B.clone().detach()
            }
            for name, lora in self.lora_layers.items()
        }

    def compute_personalization_loss(self) -> torch.Tensor:
        """
        Compute homeostatic personalization regularization.

        Penalizes drift from initial LoRA state.
        """
        loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for name, lora in self.lora_layers.items():
            initial = self._initial_lora_state[name]

            # L2 distance from initial
            a_diff = torch.norm(lora.lora_A - initial['A'])
            b_diff = torch.norm(lora.lora_B - initial['B'])

            loss = loss + a_diff + b_diff

        return self.config.personalization_lambda * loss

    def forward_with_lora(
        self,
        x: torch.Tensor,
        layer_name: str
    ) -> torch.Tensor:
        """Apply LoRA to a specific layer's output."""
        if layer_name in self.lora_layers:
            return self.lora_layers[layer_name](x)
        return torch.zeros_like(x)

    def get_all_lora_params(self) -> List[nn.Parameter]:
        """Get all LoRA parameters for optimization."""
        params = []
        for lora in self.lora_layers.values():
            params.extend([lora.lora_A, lora.lora_B])
        return params

    def should_accept_update(
        self,
        f_int_before: float,
        f_int_after: float
    ) -> bool:
        """Check if update should be accepted based on homeostatic state."""
        return self.rejection_gate.should_accept(f_int_before, f_int_after)


class MemoryConsolidation(nn.Module):
    """
    Memory consolidation module for sleep/offline learning.

    During "sleep", replays high-salience memories and updates
    LoRA adapters with homeostatic gating.
    """

    def __init__(
        self,
        config: L4Config,
        personalization: PersonalizationModule,
        buffer: ReplayBuffer
    ):
        super().__init__()
        self.config = config
        self.personalization = personalization
        self.buffer = buffer

    def consolidate(
        self,
        optimizer: torch.optim.Optimizer,
        compute_loss_fn,  # Function (batch) -> loss
        get_f_int_fn,  # Function () -> f_int
        num_samples: int = 256,
        epochs: int = 10
    ) -> Dict:
        """
        Run consolidation (sleep) loop.

        Args:
            optimizer: Optimizer for LoRA parameters
            compute_loss_fn: Function to compute loss on a batch
            get_f_int_fn: Function to get current F_int
            num_samples: Number of samples per epoch
            epochs: Number of consolidation epochs

        Returns:
            Statistics dictionary
        """
        stats = {
            'losses': [],
            'rejections': 0,
            'accepts': 0
        }

        for epoch in range(epochs):
            # Sample from replay buffer
            batch = self.buffer.sample(num_samples)
            if len(batch) == 0:
                continue

            # Get F_int before update
            f_int_before = get_f_int_fn()

            # Compute loss
            loss = compute_loss_fn(batch)

            # Add personalization regularization
            loss = loss + self.personalization.compute_personalization_loss()

            # Gradient step
            optimizer.zero_grad()
            loss.backward()

            # Temporarily apply gradients to check F_int
            with torch.no_grad():
                # Save current params
                old_params = {
                    name: p.clone()
                    for name, p in self.personalization.named_parameters()
                }

                # Apply update
                optimizer.step()

                # Check F_int after
                f_int_after = get_f_int_fn()

                # Accept or reject
                if self.personalization.should_accept_update(
                    f_int_before, f_int_after
                ):
                    stats['accepts'] += 1
                else:
                    # Rollback
                    for name, p in self.personalization.named_parameters():
                        p.copy_(old_params[name])
                    stats['rejections'] += 1

            stats['losses'].append(loss.item())

        stats['mean_loss'] = (
            sum(stats['losses']) / len(stats['losses'])
            if stats['losses'] else 0.0
        )
        stats['rejection_rate'] = (
            stats['rejections'] / (stats['accepts'] + stats['rejections'])
            if (stats['accepts'] + stats['rejections']) > 0 else 0.0
        )

        return stats


if __name__ == "__main__":
    # Test memory and personalization
    print("Testing ReplayBuffer...")

    config = L4Config()
    buffer = ReplayBuffer(config)

    # Add some entries
    for i in range(200):
        entry = MemoryEntry(
            state=torch.randn(10),
            action=torch.randn(4),
            reward=float(i % 10) / 10,
            next_state=torch.randn(10),
            done=i % 50 == 49,
            salience=0.1 + 0.9 * (i % 10) / 10,
            valence=0.5,
            arousal=0.5,
            epistemic=0.1,
            pi_q=0.1 * (i % 5),
            identity_distance=0.05 * (i % 3),
            step=i
        )
        buffer.add(entry)

    print(f"  Buffer size: {len(buffer)}")
    print(f"  Stats: {buffer.get_statistics()}")

    # Test replay distribution
    probs = buffer.compute_replay_distribution()
    print(f"  Replay probs shape: {probs.shape}")
    print(f"  Max prob: {probs.max():.4f}, Min prob: {probs.min():.4f}")

    # Test sampling
    batch = buffer.sample(32)
    print(f"  Sampled batch size: {len(batch)}")

    # Test LoRA
    print("\nTesting LoRALayer...")
    lora = LoRALayer(64, 128, rank=8, alpha=16.0)
    x = torch.randn(4, 64)
    delta = lora(x)
    print(f"  LoRA output shape: {delta.shape}")
    print(f"  Delta norm: {torch.norm(delta):.4f}")

    # Test LoRALinear
    print("\nTesting LoRALinear...")
    base = nn.Linear(64, 128)
    lora_linear = LoRALinear(base, rank=8, alpha=16.0)
    y = lora_linear(x)
    print(f"  LoRALinear output shape: {y.shape}")

    # Test homeostatic rejection
    print("\nTesting HomeostaticRejectionGate...")
    gate = HomeostaticRejectionGate(threshold=0.5)

    # Should accept
    assert gate.should_accept(1.0, 1.3) == True
    # Should reject
    assert gate.should_accept(1.0, 2.0) == False

    print(f"  Rejection rate: {gate.get_rejection_rate():.2f}")

    # Test PersonalizationModule
    print("\nTesting PersonalizationModule...")

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)

        def forward(self, x):
            return self.fc2(F.relu(self.fc1(x)))

    model = SimpleModel()
    personalization = PersonalizationModule(model, config)
    print(f"  LoRA layers: {list(personalization.lora_layers.keys())}")

    reg_loss = personalization.compute_personalization_loss()
    print(f"  Initial regularization loss: {reg_loss.item():.6f}")

    print("\nAll tests passed!")
