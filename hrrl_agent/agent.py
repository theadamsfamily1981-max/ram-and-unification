"""
HRRL Agent: Main Agent Class

Ties together all components:
- L1: Homeostatic Core (needs, drives, F_int, HRRL reward)
- L2: Hyperbolic Appraisal (ℍ^128)
- L3: MLP Gating Controller
- L4: Memory & Personalization (LoRA, homeostatic rejection)
- Identity: Hyperbolic identity manifold (experimental)
- DAU: Dynamic Architecture Update (guarded)
- Thermodynamics: Π_q measurement
- Training: Online and Sleep loops
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from .config import HRRLConfig, L1Config, L2Config, L3Config, L4Config
from .l1_homeostat import HomeostatL1, HomeostatState
from .l2_hyperbolic import HyperbolicAppraisalL2, HyperbolicAppraisalWithDrive, AppraisalOutput
from .l3_gating import GatingControllerL3, GatingOutputs
from .l4_memory import ReplayBuffer, MemoryEntry, PersonalizationModule
from .identity import HyperbolicIdentity, IdentityState
from .dau import DynamicArchitectureUpdate
from .thermodynamics import EntropyProductionMonitor
from .loops import OnlineLoop, SleepLoop, DualLoopTrainer, StepResult

logger = logging.getLogger(__name__)


class PolicyNetwork(nn.Module):
    """
    Default policy network for HRRL agent.

    Takes observation + drive state and outputs action logits.
    """

    def __init__(
        self,
        obs_dim: int,
        drive_dim: int,
        hidden_dim: int,
        action_dim: int
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim + drive_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        drive: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            (action_logits, value_estimate)
        """
        if drive is not None:
            x = torch.cat([obs, drive], dim=-1)
        else:
            x = obs

        features = self.encoder(x)

        logits = self.policy_head(features)
        value = self.value_head(features)

        return logits, value

    @property
    def logits(self):
        """For compatibility with training loops."""
        return None  # Will be set during forward


class HRRLAgent(nn.Module):
    """
    Homeostatic Reinforcement Regulated Learning Agent.

    A complete agent implementation with:
    - Homeostatic core for intrinsic motivation
    - Hyperbolic cognitive appraisal
    - Adaptive gating of learning and behavior
    - Memory consolidation with replay
    - Identity preservation (experimental)
    - Conservative architecture updates (guarded)
    - Thermodynamic monitoring
    """

    def __init__(
        self,
        config: HRRLConfig,
        obs_dim: int,
        action_dim: int,
        belief_dim: Optional[int] = None
    ):
        """
        Initialize HRRL Agent.

        Args:
            config: Complete configuration
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            belief_dim: Belief state dimension (default: same as obs_dim)
        """
        super().__init__()
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.belief_dim = belief_dim or obs_dim

        # Set random seed if specified
        if config.seed is not None:
            torch.manual_seed(config.seed)

        # === Core Components ===

        # L1: Homeostatic Core
        self.homeostat = HomeostatL1(config.l1)

        # L2: Hyperbolic Appraisal
        # Update config dimensions
        l2_config = config.l2
        l2_config.sensory_dim = obs_dim
        l2_config.belief_dim = self.belief_dim
        self.appraisal = HyperbolicAppraisalWithDrive(
            l2_config,
            num_needs=config.l1.num_needs
        )

        # L3: Gating Controller
        self.gating = GatingControllerL3(
            config.l3,
            appraisal_dim=config.l2.appraisal_dim
        )

        # L4: Memory
        self.buffer = ReplayBuffer(config.l4)

        # Policy Network
        self.policy = PolicyNetwork(
            obs_dim=obs_dim,
            drive_dim=config.l1.num_needs,
            hidden_dim=128,
            action_dim=action_dim
        )

        # Personalization (LoRA adapters)
        self.personalization = PersonalizationModule(
            self.policy,
            config.l4
        )

        # === Experimental Components ===

        # Identity (experimental)
        self.identity: Optional[HyperbolicIdentity] = None
        if config.identity.enabled:
            self.identity = HyperbolicIdentity(config.identity)
            logger.info("Identity module ENABLED (experimental)")

        # DAU (guarded, disabled by default)
        self.dau: Optional[DynamicArchitectureUpdate] = None
        if config.dau.enabled:
            self.dau = DynamicArchitectureUpdate(config.dau, self.policy)
            logger.warning("DAU is ENABLED (experimental, guarded)")

        # Thermodynamics
        self.thermo = EntropyProductionMonitor(config.thermo)

        # === Training Loops ===
        self._setup_training()

        # Step counter
        self._step = 0

        logger.info(
            f"HRRLAgent initialized: obs_dim={obs_dim}, action_dim={action_dim}, "
            f"needs={config.l1.num_needs}, hyperbolic_dim={config.l2.hyperbolic_dim}"
        )

    def _setup_training(self):
        """Set up training loops."""
        # Online loop
        self.online_loop = OnlineLoop(
            config=self.config.training,
            homeostat=self.homeostat,
            appraisal=self.appraisal,
            gating=self.gating,
            policy=self.policy,
            buffer=self.buffer,
            personalization=self.personalization,
            identity=self.identity,
            thermo=self.thermo
        )

        # Sleep loop
        self.sleep_loop = SleepLoop(
            config=self.config.training,
            homeostat=self.homeostat,
            policy=self.policy,
            buffer=self.buffer,
            personalization=self.personalization,
            identity=self.identity,
            thermo=self.thermo
        )

        # Combined trainer
        self.trainer = DualLoopTrainer(
            self.config.training,
            self.online_loop,
            self.sleep_loop
        )

    def forward(
        self,
        observation: torch.Tensor,
        belief_state: Optional[torch.Tensor] = None,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass to select action.

        Args:
            observation: Current observation
            belief_state: Optional belief state (uses observation if None)
            deterministic: If True, select argmax action

        Returns:
            (action, info_dict)
        """
        if belief_state is None:
            belief_state = observation

        # Get drive state
        drive = self.homeostat.drive

        # Compute appraisal
        appraisal_out = self.appraisal(observation, belief_state, drive)

        # Compute gating
        gating_out = self.gating(
            valence=torch.tensor(self._get_valence()),
            arousal=torch.tensor(self._get_arousal()),
            dominance=torch.tensor(self._get_dominance()),
            appraisal=appraisal_out.appraisal,
            epistemic=appraisal_out.epistemic,
            aleatoric=appraisal_out.aleatoric
        )

        # Get policy output
        logits, value = self.policy(observation.unsqueeze(0), drive.unsqueeze(0))
        logits = logits.squeeze(0)
        value = value.squeeze(0)

        # Apply temperature from gating
        tau = gating_out.tau.item()
        scaled_logits = logits / tau

        # Select action
        if deterministic:
            action = scaled_logits.argmax()
        else:
            probs = F.softmax(scaled_logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze()

        info = {
            'value': value.item(),
            'tau': tau,
            'lr_scale': gating_out.lr_scale.item(),
            'epistemic': appraisal_out.epistemic.item(),
            'aleatoric': appraisal_out.aleatoric.item(),
            'f_int': self.homeostat.free_energy,
            'drive_norm': torch.norm(drive).item()
        }

        return action, info

    def step(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        control_input: torch.Tensor,
        belief_state: Optional[torch.Tensor] = None,
        done: bool = False,
        external_reward: float = 0.0
    ) -> StepResult:
        """
        Execute one agent step with learning.

        Args:
            observation: Current observation
            action: Action taken
            control_input: Effect on needs
            belief_state: Optional belief state
            done: Episode termination flag
            external_reward: Optional external reward

        Returns:
            StepResult with step information
        """
        self._step += 1

        if belief_state is None:
            belief_state = observation

        # Use trainer for combined online/sleep
        online_result, sleep_result = self.trainer.step(
            observation=observation,
            belief_state=belief_state,
            action=action,
            control_input=control_input,
            done=done,
            external_reward=external_reward
        )

        # Update thermodynamics (placeholder - would need actual spike data)
        # self.thermo.compute_entropy_production(...)

        return online_result

    def _get_valence(self) -> float:
        """Get current valence (would be from last homeostat step)."""
        return 0.0  # Placeholder

    def _get_arousal(self) -> float:
        """Get current arousal."""
        return 0.5  # Placeholder

    def _get_dominance(self) -> float:
        """Get current dominance."""
        return 0.5  # Placeholder

    def consolidate(self) -> StepResult:
        """Force a sleep consolidation cycle."""
        return self.trainer.force_sleep()

    def get_state(self) -> Dict[str, Any]:
        """Get current agent state."""
        state = {
            'step': self._step,
            'f_int': self.homeostat.free_energy,
            'needs': self.homeostat.n.tolist(),
            'drive': self.homeostat.drive.tolist(),
            'buffer_size': len(self.buffer)
        }

        if self.identity is not None:
            id_state = self.identity.get_state()
            state['identity'] = {
                'drift': id_state.drift_from_origin,
                'alert_level': id_state.alert_level.value
            }

        return state

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        stats = {
            'agent': {
                'step': self._step,
                'f_int': self.homeostat.free_energy
            },
            'training': self.trainer.get_statistics(),
            'buffer': self.buffer.get_statistics(),
            'thermo': self.thermo.get_statistics()
        }

        if self.identity is not None:
            stats['identity'] = {
                'drift': self.identity.get_state().drift_from_origin,
                'alert_level': self.identity.get_state().alert_level.value
            }

        if self.dau is not None:
            stats['dau'] = self.dau.get_statistics()

        return stats

    def reset(self):
        """Reset agent state for new episode."""
        self.homeostat.reset()

        if self.identity is not None:
            self.identity.reset_to_original()

    def save(self, path: str):
        """Save agent state."""
        state = {
            'config': self.config,
            'policy_state': self.policy.state_dict(),
            'homeostat_state': self.homeostat.get_state_dict(),
            'step': self._step
        }

        if self.identity is not None:
            state['identity'] = self.identity.state_dict()

        torch.save(state, path)
        logger.info(f"Agent saved to {path}")

    def load(self, path: str):
        """Load agent state."""
        state = torch.load(path)

        self.policy.load_state_dict(state['policy_state'])
        self.homeostat.load_state_dict_custom(state['homeostat_state'])
        self._step = state['step']

        if 'identity' in state and self.identity is not None:
            self.identity.load_state_dict(state['identity'])

        logger.info(f"Agent loaded from {path}")


def create_agent(
    obs_dim: int,
    action_dim: int,
    num_needs: int = 8,
    enable_identity: bool = True,
    enable_dau: bool = False,
    device: str = "cpu"
) -> HRRLAgent:
    """
    Factory function to create an HRRL agent with sensible defaults.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        num_needs: Number of homeostatic needs
        enable_identity: Enable identity module (experimental)
        enable_dau: Enable DAU (experimental, disabled by default)
        device: Device to use

    Returns:
        Configured HRRLAgent
    """
    config = HRRLConfig(
        l1=L1Config(num_needs=num_needs),
        l2=L2Config(),
        l3=L3Config(),
        l4=L4Config(),
        device=device
    )

    config.identity.enabled = enable_identity
    config.dau.enabled = enable_dau

    agent = HRRLAgent(config, obs_dim, action_dim)

    return agent.to(device) if device != "cpu" else agent


if __name__ == "__main__":
    # Test agent
    print("Testing HRRLAgent...")
    print("=" * 60)

    # Create agent
    agent = create_agent(
        obs_dim=64,
        action_dim=8,
        num_needs=4,
        enable_identity=True,
        enable_dau=False
    )

    print(f"\nAgent created:")
    print(f"  Observation dim: 64")
    print(f"  Action dim: 8")
    print(f"  Needs: 4")
    print(f"  Identity enabled: {agent.identity is not None}")
    print(f"  DAU enabled: {agent.dau is not None}")

    # Test forward pass
    print("\nTesting forward pass...")
    obs = torch.randn(64)
    action, info = agent(obs)
    print(f"  Selected action: {action.item()}")
    print(f"  Value estimate: {info['value']:.4f}")
    print(f"  Temperature: {info['tau']:.4f}")
    print(f"  F_int: {info['f_int']:.4f}")

    # Test training steps
    print("\nRunning training steps...")
    for i in range(100):
        obs = torch.randn(64)
        action, _ = agent(obs, deterministic=False)
        control = torch.randn(4) * 0.1

        result = agent.step(
            observation=obs,
            action=torch.tensor(action),
            control_input=control
        )

        if i % 25 == 0:
            print(f"  Step {i}: reward={result.reward:.4f}, f_int={result.f_int:.4f}")

    # Get statistics
    print("\nAgent Statistics:")
    stats = agent.get_statistics()
    print(f"  Total steps: {stats['agent']['step']}")
    print(f"  F_int: {stats['agent']['f_int']:.4f}")
    print(f"  Buffer size: {stats['buffer'].get('size', 0)}")
    print(f"  Online mean reward: {stats['training']['online']['mean_reward']:.4f}")

    if 'identity' in stats:
        print(f"  Identity drift: {stats['identity']['drift']:.4f}")
        print(f"  Identity alert: {stats['identity']['alert_level']}")

    # Test consolidation
    print("\nTesting sleep consolidation...")
    result = agent.consolidate()
    print(f"  Sleep loss: {result.loss:.4f}")

    # Test save/load
    print("\nTesting save/load...")
    agent.save("/tmp/test_agent.pt")
    agent.load("/tmp/test_agent.pt")
    print("  Save/load successful!")

    print("\n" + "=" * 60)
    print("All agent tests passed!")
