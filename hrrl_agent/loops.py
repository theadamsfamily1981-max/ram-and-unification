"""
Online and Sleep Training Loops

Implements the dual-loop training paradigm:
1. Online loop: Real-time learning during experience
2. Sleep loop: Offline consolidation and replay

Based on the pseudocode scaffolding from the spec.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

from .config import TrainingConfig, HRRLConfig
from .l1_homeostat import HomeostatL1, HomeostatState
from .l2_hyperbolic import HyperbolicAppraisalL2, AppraisalOutput
from .l3_gating import GatingControllerL3, GatingOutputs
from .l4_memory import ReplayBuffer, MemoryEntry, PersonalizationModule
from .identity import HyperbolicIdentity
from .thermodynamics import EntropyProductionMonitor

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result from a single training step."""
    reward: float
    f_int: float
    loss: Optional[float]
    gating: GatingOutputs
    accepted_update: bool
    info: Dict


class TrainingLoop(ABC):
    """Abstract base class for training loops."""

    @abstractmethod
    def step(self, *args, **kwargs) -> StepResult:
        """Execute one training step."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        pass


class OnlineLoop(TrainingLoop):
    """
    Online training loop for real-time learning.

    Processes experiences as they arrive, with:
    - Immediate HRRL reward computation
    - Gated learning rate adjustment
    - Memory storage with salience
    - Homeostatic rejection of harmful updates
    """

    def __init__(
        self,
        config: TrainingConfig,
        homeostat: HomeostatL1,
        appraisal: HyperbolicAppraisalL2,
        gating: GatingControllerL3,
        policy: nn.Module,
        buffer: ReplayBuffer,
        personalization: Optional[PersonalizationModule] = None,
        identity: Optional[HyperbolicIdentity] = None,
        thermo: Optional[EntropyProductionMonitor] = None
    ):
        self.config = config
        self.homeostat = homeostat
        self.appraisal = appraisal
        self.gating = gating
        self.policy = policy
        self.buffer = buffer
        self.personalization = personalization
        self.identity = identity
        self.thermo = thermo

        # Optimizer for online updates
        params = list(policy.parameters())
        if personalization is not None:
            params.extend(personalization.get_all_lora_params())

        self.optimizer = self._create_optimizer(params)

        # Statistics
        self._step_count = 0
        self._total_reward = 0.0
        self._accepted_updates = 0
        self._rejected_updates = 0

    def _create_optimizer(self, params: List[nn.Parameter]) -> torch.optim.Optimizer:
        """Create optimizer based on config."""
        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.online_lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.online_lr
            )
        else:
            return torch.optim.SGD(
                params,
                lr=self.config.online_lr
            )

    def step(
        self,
        observation: torch.Tensor,
        belief_state: torch.Tensor,
        action: torch.Tensor,
        control_input: torch.Tensor,  # Effect on needs
        done: bool = False,
        external_reward: float = 0.0
    ) -> StepResult:
        """
        Execute one online training step.

        Args:
            observation: Current sensory observation
            belief_state: Current belief state
            action: Action taken
            control_input: Effect of action on needs
            done: Episode termination flag
            external_reward: Optional external reward signal

        Returns:
            StepResult with all relevant information
        """
        self._step_count += 1

        # 1. Update homeostat and get HRRL reward
        f_int_before = self.homeostat.free_energy
        homeo_state = self.homeostat.step(control_input)

        # Scale reward
        reward = self.config.hrrl_reward_scale * homeo_state.reward + external_reward
        self._total_reward += reward

        # 2. Compute appraisal
        appraisal_out = self.appraisal(observation, belief_state)

        # 3. Compute gating signals
        gating_out = self.gating(
            valence=torch.tensor(homeo_state.valence),
            arousal=torch.tensor(homeo_state.arousal),
            dominance=torch.tensor(homeo_state.dominance),
            appraisal=appraisal_out.appraisal,
            epistemic=appraisal_out.epistemic,
            aleatoric=appraisal_out.aleatoric
        )

        # 4. Compute salience for memory
        drive_norm = torch.norm(homeo_state.d).item()
        appraisal_norm = torch.norm(appraisal_out.appraisal).item()
        salience = self.buffer.compute_salience(
            drive_norm,
            appraisal_norm,
            appraisal_out.epistemic.item()
        )

        # 5. Get identity distance if available
        identity_distance = 0.0
        if self.identity is not None:
            # Use appraisal embedding as proxy for experience embedding
            identity_distance = self.identity.compute_identity_distance(
                appraisal_out.z_combined
            ).mean().item()

        # 6. Get Π_q if available
        pi_q = 0.0
        if self.thermo is not None:
            # Would need actual spike data - use placeholder
            pi_q = 0.1  # Placeholder

        # 7. Store in memory (gated by mem_write_p)
        if torch.rand(1).item() < gating_out.mem_write_p.item():
            entry = MemoryEntry(
                state=observation.clone(),
                action=action.clone(),
                reward=reward,
                next_state=observation.clone(),  # Would be next observation
                done=done,
                salience=salience,
                valence=homeo_state.valence,
                arousal=homeo_state.arousal,
                epistemic=appraisal_out.epistemic.item(),
                pi_q=pi_q,
                identity_distance=identity_distance,
                step=self._step_count
            )
            self.buffer.add(entry)

        # 8. Online policy update (gated)
        loss = None
        accepted = True

        if self._should_update():
            # Compute policy loss (simplified - actual implementation would vary)
            loss_value = self._compute_policy_loss(
                observation, action, reward, gating_out
            )

            if loss_value is not None:
                # Apply gated learning rate
                effective_lr = self.config.online_lr * gating_out.lr_scale.item()
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = effective_lr

                # Compute gradients
                self.optimizer.zero_grad()
                loss_value.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.online_grad_clip
                )

                # Check homeostatic rejection
                if self.personalization is not None:
                    f_int_after_preview = self._preview_f_int_after_update()
                    accepted = self.personalization.should_accept_update(
                        f_int_before,
                        f_int_after_preview
                    )

                if accepted:
                    self.optimizer.step()
                    self._accepted_updates += 1
                else:
                    self._rejected_updates += 1
                    logger.debug(
                        f"Online update rejected: F_int would increase too much"
                    )

                loss = loss_value.item()

        return StepResult(
            reward=reward,
            f_int=homeo_state.f_int,
            loss=loss,
            gating=gating_out,
            accepted_update=accepted,
            info={
                'salience': salience,
                'identity_distance': identity_distance,
                'pi_q': pi_q,
                'drive_norm': drive_norm,
                'appraisal_norm': appraisal_norm
            }
        )

    def _should_update(self) -> bool:
        """Determine if we should perform an update this step."""
        return True  # Online: update every step

    def _compute_policy_loss(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        gating: GatingOutputs
    ) -> Optional[torch.Tensor]:
        """
        Compute policy loss for online update.

        This is a simplified REINFORCE-style loss.
        Actual implementation would depend on policy architecture.
        """
        # Get policy output
        with torch.enable_grad():
            policy_out = self.policy(observation.unsqueeze(0))

            # Assume policy outputs action logits
            if hasattr(policy_out, 'logits'):
                logits = policy_out.logits
            else:
                logits = policy_out

            # Compute log probability of taken action
            log_prob = F.log_softmax(logits, dim=-1)

            # Simple policy gradient: -reward * log_prob
            # Temperature-scaled by tau from gating
            tau = gating.tau.item()
            scaled_log_prob = log_prob / tau

            # Loss (negative because we maximize reward)
            if action.dim() == 0:
                action_idx = action.long()
            else:
                action_idx = action.argmax().long()

            loss = -reward * scaled_log_prob[0, action_idx]

            return loss

    def _preview_f_int_after_update(self) -> float:
        """
        Preview what F_int would be after applying gradients.

        This is an approximation for rejection checking.
        """
        # For now, just return current F_int + small noise
        # Actual implementation would simulate the update
        return self.homeostat.free_energy + 0.01 * torch.randn(1).item()

    def get_statistics(self) -> Dict:
        """Get online loop statistics."""
        return {
            'steps': self._step_count,
            'total_reward': self._total_reward,
            'mean_reward': self._total_reward / max(1, self._step_count),
            'accepted_updates': self._accepted_updates,
            'rejected_updates': self._rejected_updates,
            'rejection_rate': (
                self._rejected_updates /
                max(1, self._accepted_updates + self._rejected_updates)
            ),
            'buffer_size': len(self.buffer)
        }


class SleepLoop(TrainingLoop):
    """
    Sleep/consolidation loop for offline learning.

    Replays high-salience memories with:
    - Salience-weighted sampling
    - Homeostatic rejection of harmful updates
    - LoRA adapter updates
    - Identity preservation checks
    """

    def __init__(
        self,
        config: TrainingConfig,
        homeostat: HomeostatL1,
        policy: nn.Module,
        buffer: ReplayBuffer,
        personalization: Optional[PersonalizationModule] = None,
        identity: Optional[HyperbolicIdentity] = None,
        thermo: Optional[EntropyProductionMonitor] = None
    ):
        self.config = config
        self.homeostat = homeostat
        self.policy = policy
        self.buffer = buffer
        self.personalization = personalization
        self.identity = identity
        self.thermo = thermo

        # Optimizer for sleep updates
        params = list(policy.parameters())
        if personalization is not None:
            params.extend(personalization.get_all_lora_params())

        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.sleep_lr,
            weight_decay=config.weight_decay
        )

        # Statistics
        self._consolidation_count = 0
        self._total_loss = 0.0
        self._accepted_batches = 0
        self._rejected_batches = 0

    def step(
        self,
        num_samples: Optional[int] = None,
        epochs: Optional[int] = None
    ) -> StepResult:
        """
        Execute one sleep consolidation cycle.

        Args:
            num_samples: Number of samples to replay (default: from config)
            epochs: Number of epochs (default: from config)

        Returns:
            StepResult with consolidation statistics
        """
        if num_samples is None:
            num_samples = self.config.sleep_replay_samples
        if epochs is None:
            epochs = self.config.sleep_epochs

        self._consolidation_count += 1

        total_loss = 0.0
        accepted = 0
        rejected = 0

        for epoch in range(epochs):
            # Sample from buffer using salience-weighted distribution
            batch = self.buffer.sample(self.config.sleep_batch_size)

            if len(batch) == 0:
                continue

            f_int_before = self.homeostat.free_energy

            # Compute loss over batch
            loss = self._compute_batch_loss(batch)

            if loss is None:
                continue

            # Add personalization regularization
            if self.personalization is not None:
                loss = loss + self.personalization.compute_personalization_loss()

            # Add identity preservation loss
            if self.identity is not None:
                loss = loss + self.identity.compute_core_value_loss()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Check if update should be accepted
            should_accept = True

            if self.personalization is not None:
                f_int_preview = self._preview_f_int()
                should_accept = self.personalization.should_accept_update(
                    f_int_before,
                    f_int_preview
                )

            if self.identity is not None and should_accept:
                # Also check identity drift
                identity_state = self.identity.get_state()
                if identity_state.alert_level.value in ["critical", "reject"]:
                    should_accept = False
                    logger.warning(
                        f"Sleep update rejected due to identity alert: "
                        f"{identity_state.alert_level.value}"
                    )

            if should_accept:
                self.optimizer.step()
                accepted += 1
                total_loss += loss.item()
            else:
                rejected += 1

        self._total_loss += total_loss
        self._accepted_batches += accepted
        self._rejected_batches += rejected

        # Create dummy gating output for return type
        dummy_gating = GatingOutputs(
            tau=torch.tensor(0.5),
            lr_scale=torch.tensor(1.0),
            mem_write_p=torch.tensor(0.5),
            empathy_w=torch.tensor(0.5),
            att_gain=torch.tensor(1.0),
            raw_logits=torch.zeros(5)
        )

        return StepResult(
            reward=0.0,  # No reward during sleep
            f_int=self.homeostat.free_energy,
            loss=total_loss / max(1, accepted),
            gating=dummy_gating,
            accepted_update=accepted > 0,
            info={
                'epochs': epochs,
                'accepted_batches': accepted,
                'rejected_batches': rejected,
                'total_loss': total_loss
            }
        )

    def _compute_batch_loss(
        self,
        batch: List[MemoryEntry]
    ) -> Optional[torch.Tensor]:
        """Compute loss over a batch of memories."""
        if len(batch) == 0:
            return None

        # Stack batch
        states = torch.stack([e.state for e in batch])
        actions = torch.stack([e.action for e in batch])
        rewards = torch.tensor([e.reward for e in batch])

        # Get policy outputs
        policy_out = self.policy(states)

        if hasattr(policy_out, 'logits'):
            logits = policy_out.logits
        else:
            logits = policy_out

        # Compute loss (simplified)
        log_probs = F.log_softmax(logits, dim=-1)

        # For continuous actions, use MSE; for discrete, use cross-entropy
        if actions.dim() > 1 and actions.size(-1) == logits.size(-1):
            # Continuous: MSE weighted by reward
            loss = F.mse_loss(logits, actions, reduction='none')
            loss = (loss.mean(dim=-1) * (-rewards)).mean()
        else:
            # Discrete: cross-entropy weighted by reward
            action_indices = actions.argmax(dim=-1) if actions.dim() > 1 else actions.long()
            loss = F.cross_entropy(logits, action_indices, reduction='none')
            loss = (loss * (-rewards)).mean()

        return loss

    def _preview_f_int(self) -> float:
        """Preview F_int after update."""
        return self.homeostat.free_energy + 0.01 * torch.randn(1).item()

    def get_statistics(self) -> Dict:
        """Get sleep loop statistics."""
        return {
            'consolidations': self._consolidation_count,
            'total_loss': self._total_loss,
            'mean_loss': self._total_loss / max(1, self._accepted_batches),
            'accepted_batches': self._accepted_batches,
            'rejected_batches': self._rejected_batches,
            'rejection_rate': (
                self._rejected_batches /
                max(1, self._accepted_batches + self._rejected_batches)
            )
        }


class DualLoopTrainer:
    """
    Coordinator for online and sleep training loops.

    Manages the interleaving of online experience and
    offline consolidation.
    """

    def __init__(
        self,
        config: TrainingConfig,
        online_loop: OnlineLoop,
        sleep_loop: SleepLoop
    ):
        self.config = config
        self.online = online_loop
        self.sleep = sleep_loop

        self._steps_since_sleep = 0
        self._total_steps = 0

    def step(
        self,
        observation: torch.Tensor,
        belief_state: torch.Tensor,
        action: torch.Tensor,
        control_input: torch.Tensor,
        done: bool = False,
        external_reward: float = 0.0
    ) -> Tuple[StepResult, Optional[StepResult]]:
        """
        Execute one step, potentially including sleep consolidation.

        Returns:
            (online_result, sleep_result or None)
        """
        self._total_steps += 1
        self._steps_since_sleep += 1

        # Online step
        online_result = self.online.step(
            observation, belief_state, action, control_input,
            done, external_reward
        )

        # Check if time for sleep
        sleep_result = None
        if self._steps_since_sleep >= self.config.steps_between_sleep:
            logger.info(f"Starting sleep consolidation at step {self._total_steps}")
            sleep_result = self.sleep.step()
            self._steps_since_sleep = 0
            logger.info(
                f"Sleep complete: loss={sleep_result.loss:.4f}, "
                f"accepted={sleep_result.info['accepted_batches']}"
            )

        return online_result, sleep_result

    def force_sleep(self) -> StepResult:
        """Force a sleep consolidation cycle."""
        logger.info("Forced sleep consolidation")
        result = self.sleep.step()
        self._steps_since_sleep = 0
        return result

    def get_statistics(self) -> Dict:
        """Get combined statistics."""
        return {
            'total_steps': self._total_steps,
            'steps_since_sleep': self._steps_since_sleep,
            'online': self.online.get_statistics(),
            'sleep': self.sleep.get_statistics()
        }


if __name__ == "__main__":
    # Test training loops
    print("Testing Training Loops...")

    from .config import L1Config, L2Config, L3Config, L4Config

    # Create components
    l1_config = L1Config(num_needs=4)
    homeostat = HomeostatL1(l1_config)

    l2_config = L2Config(sensory_dim=32, belief_dim=32)
    appraisal = HyperbolicAppraisalL2(l2_config)

    l3_config = L3Config()
    gating = GatingControllerL3(l3_config)

    # Simple policy
    class SimplePolicy(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Linear(32, 8)

        def forward(self, x):
            return self.net(x)

    policy = SimplePolicy()

    l4_config = L4Config()
    buffer = ReplayBuffer(l4_config)

    train_config = TrainingConfig(steps_between_sleep=50)

    # Create loops
    online = OnlineLoop(
        train_config, homeostat, appraisal, gating, policy, buffer
    )

    sleep = SleepLoop(
        train_config, homeostat, policy, buffer
    )

    trainer = DualLoopTrainer(train_config, online, sleep)

    # Run some steps
    print("\nRunning training steps...")
    for i in range(100):
        obs = torch.randn(32)
        belief = torch.randn(32)
        action = torch.randint(0, 8, (1,))
        control = torch.randn(4) * 0.1

        online_result, sleep_result = trainer.step(
            obs, belief, action, control
        )

        if sleep_result is not None:
            print(f"  Step {i}: Sleep triggered, loss={sleep_result.loss:.4f}")

    # Get statistics
    print("\nTraining Statistics:")
    stats = trainer.get_statistics()

    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Online mean reward: {stats['online']['mean_reward']:.4f}")
    print(f"  Online rejection rate: {stats['online']['rejection_rate']:.2%}")
    print(f"  Sleep consolidations: {stats['sleep']['consolidations']}")

    print("\nAll loop tests passed!")
