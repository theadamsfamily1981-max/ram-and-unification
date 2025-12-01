"""Layer 4: Policy/Regulator - Affective action selection.

This module implements the PFC-analogue policy that:
- Takes observations from Layers 1-3
- Selects control actions (gates) to minimize homeostatic drive
- Can be backed by heuristics, RL policy, or SNN hardware

The action space consists of "gates" that modulate:
- Attention allocation
- Learning rate / plasticity
- Memory write probability
- Exploration temperature
- Sleep/consolidation triggers
- User clarification requests

Reference: HRRL (Homeostatic Reinforcement Learning), Active Inference
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
import time

from .homeostatic import HomeostaticCore, HomeostaticState
from .interoceptive import InteroceptiveEvent
from .appraisal import AppraisalResult, DiscreteEmotion


# =============================================================================
# Action Space
# =============================================================================

@dataclass
class AffectiveAction:
    """Control action (gate settings) from Layer 4 policy.

    These gates modulate system behavior in response to affective state.
    """
    # Attention
    attention_gain: float = 1.0         # [0.5, 2.0] scale attention
    attention_social_bias: float = 0.0  # [-1, 1] bias toward social streams

    # Learning
    optimizer_lr_mult: float = 1.0      # [0.1, 2.0] scale learning rate
    plasticity_gate: float = 1.0        # [0, 1] enable/disable learning

    # Memory
    memory_write_prob: float = 0.5      # [0, 1] probability of writing to memory
    memory_retrieval_k: int = 5         # [1, 20] number of memories to retrieve

    # Exploration
    exploration_temp: float = 1.0       # [0.1, 3.0] temperature for sampling
    exploration_bonus: float = 0.0      # [0, 1] intrinsic reward bonus

    # System
    sleep_trigger: bool = False         # Whether to trigger consolidation
    alert_level: int = 0                # [0, 3] 0=normal, 1=elevated, 2=high, 3=critical

    # Interaction
    request_clarification: bool = False # Ask user for more info
    verbose_mode: bool = False          # Increase output detail

    def to_dict(self) -> Dict[str, Any]:
        return {
            "attention_gain": self.attention_gain,
            "attention_social_bias": self.attention_social_bias,
            "optimizer_lr_mult": self.optimizer_lr_mult,
            "plasticity_gate": self.plasticity_gate,
            "memory_write_prob": self.memory_write_prob,
            "memory_retrieval_k": self.memory_retrieval_k,
            "exploration_temp": self.exploration_temp,
            "exploration_bonus": self.exploration_bonus,
            "sleep_trigger": self.sleep_trigger,
            "alert_level": self.alert_level,
            "request_clarification": self.request_clarification,
            "verbose_mode": self.verbose_mode,
        }

    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor for neural network output."""
        return torch.tensor([
            self.attention_gain,
            self.attention_social_bias,
            self.optimizer_lr_mult,
            self.plasticity_gate,
            self.memory_write_prob,
            self.exploration_temp,
            self.exploration_bonus,
            float(self.alert_level) / 3.0,
        ], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "AffectiveAction":
        """Create from neural network output tensor."""
        return cls(
            attention_gain=float(np.clip(t[0].item(), 0.5, 2.0)),
            attention_social_bias=float(np.clip(t[1].item(), -1, 1)),
            optimizer_lr_mult=float(np.clip(t[2].item(), 0.1, 2.0)),
            plasticity_gate=float(np.clip(t[3].item(), 0, 1)),
            memory_write_prob=float(np.clip(t[4].item(), 0, 1)),
            exploration_temp=float(np.clip(t[5].item(), 0.1, 3.0)),
            exploration_bonus=float(np.clip(t[6].item(), 0, 1)),
            alert_level=int(np.clip(t[7].item() * 3, 0, 3)),
        )


# =============================================================================
# Observation Builder
# =============================================================================

def build_policy_observation(
    homeostatic_core: HomeostaticCore,
    interoceptive_event: InteroceptiveEvent,
    appraisal_result: AppraisalResult,
) -> torch.Tensor:
    """Build observation vector for policy network.

    Concatenates:
    - Layer 1: homeostatic state + drive + PAD [11]
    - Layer 2: interoceptive features [25]
    - Layer 3: appraisal dims + PAD [13]

    Returns:
        Observation tensor [49]
    """
    # Layer 1
    l1 = homeostatic_core.get_observation()  # [11]

    # Layer 2
    l2 = interoceptive_event.to_observation_vector()  # [25]

    # Layer 3
    appraisal_dims = appraisal_result.appraisal_dims.to_tensor()  # [10]
    generated_pad = appraisal_result.generated_pad.to_tensor()  # [3]
    l3 = torch.cat([appraisal_dims, generated_pad])  # [13]

    return torch.cat([l1, l2, l3])


# =============================================================================
# Heuristic Policy
# =============================================================================

def heuristic_policy(
    homeostatic_core: HomeostaticCore,
    interoceptive_event: InteroceptiveEvent,
    appraisal_result: AppraisalResult,
) -> AffectiveAction:
    """Simple heuristic policy for baseline/testing.

    Maps affective state to actions using hand-crafted rules.
    """
    action = AffectiveAction()
    state = homeostatic_core.state
    drive = homeostatic_core.current_drive
    emotion = appraisal_result.discrete_emotion
    pad = appraisal_result.generated_pad
    uncertainty = appraisal_result.epistemic_uncertainty

    # High drive + high uncertainty → explore more, ask for clarification
    if drive > 0.5 and uncertainty > 0.6:
        action.exploration_temp = 1.5
        action.request_clarification = True
        action.attention_gain = 1.3

    # Negative valence → reduce plasticity, increase attention
    if pad.pleasure < -0.3:
        action.plasticity_gate = 0.5
        action.attention_gain = 1.4
        action.memory_write_prob = 0.3  # Don't store negative experiences deeply

    # Fear/distress → high alert, conservative learning
    if emotion in [DiscreteEmotion.FEAR, DiscreteEmotion.DISTRESS]:
        action.alert_level = 2
        action.optimizer_lr_mult = 0.5
        action.exploration_temp = 0.7  # Be more conservative
        action.attention_gain = 1.5

    # Anger/frustration → moderate alert, social attention
    if emotion in [DiscreteEmotion.ANGER, DiscreteEmotion.FRUSTRATION]:
        action.alert_level = 1
        action.attention_social_bias = 0.5
        action.exploration_temp = 0.8

    # Curiosity/interest → high exploration, high plasticity
    if emotion in [DiscreteEmotion.CURIOSITY, DiscreteEmotion.INTEREST]:
        action.exploration_temp = 1.5
        action.exploration_bonus = 0.3
        action.plasticity_gate = 1.0
        action.memory_write_prob = 0.8

    # Confusion → ask for help, reduce cognitive load
    if emotion == DiscreteEmotion.CONFUSION:
        action.request_clarification = True
        action.verbose_mode = True
        action.exploration_temp = 0.8

    # Boredom → increase novelty seeking
    if emotion == DiscreteEmotion.BOREDOM or state.novelty < 0.2:
        action.exploration_temp = 1.8
        action.exploration_bonus = 0.5

    # High cognitive load → consolidate
    if state.cogload > 0.8:
        action.sleep_trigger = True
        action.memory_write_prob = 0.2
        action.attention_gain = 0.8

    # Low energy → conservative mode
    if state.energy < 0.3:
        action.optimizer_lr_mult = 0.5
        action.exploration_temp = 0.6
        action.alert_level = max(action.alert_level, 1)

    # Safety concern → maximum alert
    if state.safety < 0.5:
        action.alert_level = 3
        action.plasticity_gate = 0.0  # Freeze learning
        action.attention_gain = 2.0

    return action


# =============================================================================
# Neural Policy Network
# =============================================================================

class AffectivePolicyNetwork(nn.Module):
    """Neural network policy for affective action selection.

    Takes observation from all layers and outputs action vector.
    Can be trained with PPO, SAC, or distillation from heuristics.
    """

    def __init__(
        self,
        obs_dim: int = 49,
        action_dim: int = 8,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Actor (policy)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))

        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action mean and value.

        Args:
            obs: Observation tensor [batch, obs_dim]

        Returns:
            action_mean: [batch, action_dim]
            value: [batch, 1]
        """
        h = self.actor(obs)
        action_mean = torch.tanh(self.actor_mean(h))
        value = self.critic(obs)
        return action_mean, value

    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.

        Returns:
            action: [batch, action_dim]
            log_prob: [batch]
            value: [batch, 1]
        """
        action_mean, value = self(obs)

        if deterministic:
            return action_mean, torch.zeros(obs.shape[0]), value

        std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(action_mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        return action, log_prob, value


# =============================================================================
# Policy Regulator (main interface)
# =============================================================================

class PolicyRegulator:
    """Layer 4: Affective policy/regulator.

    Selects control actions (gates) based on affective state to minimize
    long-term homeostatic drive.

    Can use:
    - Heuristic policy (default)
    - Neural network (learned)
    - SNN hardware (FPGA-accelerated)
    """

    def __init__(
        self,
        method: str = "heuristic",
        neural_model: Optional[AffectivePolicyNetwork] = None,
        snn_interface: Optional[Any] = None,  # Interface to FPGA SNN
    ):
        self.method = method
        self.neural_model = neural_model
        self.snn_interface = snn_interface

        self._action_history: List[AffectiveAction] = []
        self._reward_history: List[float] = []
        self._history_len = 100

    def select_action(
        self,
        homeostatic_core: HomeostaticCore,
        interoceptive_event: InteroceptiveEvent,
        appraisal_result: AppraisalResult,
        deterministic: bool = False,
    ) -> AffectiveAction:
        """Select action based on current affective state.

        Args:
            homeostatic_core: Layer 1 state
            interoceptive_event: Layer 2 features
            appraisal_result: Layer 3 appraisal
            deterministic: Whether to use deterministic action

        Returns:
            AffectiveAction with gate settings
        """
        if self.method == "neural" and self.neural_model is not None:
            action = self._neural_action(
                homeostatic_core, interoceptive_event, appraisal_result, deterministic
            )
        elif self.method == "snn" and self.snn_interface is not None:
            action = self._snn_action(
                homeostatic_core, interoceptive_event, appraisal_result
            )
        else:
            action = heuristic_policy(
                homeostatic_core, interoceptive_event, appraisal_result
            )

        # Track history
        self._action_history.append(action)
        if len(self._action_history) > self._history_len:
            self._action_history = self._action_history[-self._history_len:]

        return action

    def _neural_action(
        self,
        homeostatic_core: HomeostaticCore,
        interoceptive_event: InteroceptiveEvent,
        appraisal_result: AppraisalResult,
        deterministic: bool,
    ) -> AffectiveAction:
        """Select action using neural policy."""
        obs = build_policy_observation(
            homeostatic_core, interoceptive_event, appraisal_result
        ).unsqueeze(0)

        with torch.no_grad():
            action_t, _, _ = self.neural_model.get_action(obs, deterministic)

        return AffectiveAction.from_tensor(action_t[0])

    def _snn_action(
        self,
        homeostatic_core: HomeostaticCore,
        interoceptive_event: InteroceptiveEvent,
        appraisal_result: AppraisalResult,
    ) -> AffectiveAction:
        """Select action using SNN hardware.

        This interfaces with the Kitten FPGA fabric.
        """
        # TODO: Implement SNN interface
        # Would encode observation as spikes, run through FPGA,
        # decode output spikes to action
        return heuristic_policy(
            homeostatic_core, interoceptive_event, appraisal_result
        )

    def update_reward(self, reward: float):
        """Record reward for RL training."""
        self._reward_history.append(reward)
        if len(self._reward_history) > self._history_len:
            self._reward_history = self._reward_history[-self._history_len:]

    def get_mean_reward(self, window: int = 20) -> float:
        """Get mean reward over recent window."""
        recent = self._reward_history[-window:]
        return np.mean(recent) if recent else 0.0


# =============================================================================
# SNN Policy Interface (for FPGA hardware)
# =============================================================================

class SNNPolicyInterface:
    """Interface to SNN hardware for Layer 4 policy.

    Connects the 4-layer affect architecture to the Kitten FPGA fabric,
    allowing hardware-accelerated policy execution.

    The SNN receives:
    - Encoded observations as spike trains
    - Runs through trained fabric
    - Decodes output spikes to action

    This enables sub-millisecond affective decisions on FPGA.
    """

    def __init__(
        self,
        fabric_topology_path: str,
        weights_path: str,
        neurons_path: str,
        num_timesteps: int = 64,
    ):
        self.fabric_topology_path = fabric_topology_path
        self.weights_path = weights_path
        self.neurons_path = neurons_path
        self.num_timesteps = num_timesteps

        # Load topology
        self._topology = None
        self._num_input_neurons = 64
        self._num_output_neurons = 16
        self._load_topology()

    def _load_topology(self):
        """Load fabric topology from JSON."""
        import json
        try:
            with open(self.fabric_topology_path, "r") as f:
                self._topology = json.load(f)
            self._num_input_neurons = self._topology.get("input_neurons", 64)
            self._num_output_neurons = self._topology.get("output_neurons", 16)
        except Exception as e:
            print(f"[SNNPolicy] Could not load topology: {e}")

    def encode_observation(
        self,
        obs: torch.Tensor,
        encoding_rate: float = 100.0,
    ) -> np.ndarray:
        """Encode observation tensor as spike trains.

        Args:
            obs: Observation tensor [obs_dim]
            encoding_rate: Max firing rate (Hz)

        Returns:
            Spike array [num_timesteps, num_input_neurons]
        """
        # Normalize observation to [0, 1]
        obs_np = obs.detach().cpu().numpy()
        obs_norm = (obs_np - obs_np.min()) / (obs_np.max() - obs_np.min() + 1e-8)

        # Truncate or pad to match input neurons
        if len(obs_norm) > self._num_input_neurons:
            obs_norm = obs_norm[:self._num_input_neurons]
        elif len(obs_norm) < self._num_input_neurons:
            obs_norm = np.pad(obs_norm, (0, self._num_input_neurons - len(obs_norm)))

        # Rate coding
        spike_probs = obs_norm * (encoding_rate / 1000.0)  # Prob per ms
        spikes = (np.random.random((self.num_timesteps, self._num_input_neurons)) < spike_probs).astype(np.uint8)

        return spikes

    def decode_action(self, output_spikes: np.ndarray) -> AffectiveAction:
        """Decode output spikes to action.

        Output neurons are organized as 4 actions × 4 voters (like safety reflex).
        Uses spike count voting.

        Args:
            output_spikes: [num_timesteps, num_output_neurons] or [num_output_neurons]

        Returns:
            AffectiveAction
        """
        if output_spikes.ndim == 2:
            spike_counts = output_spikes.sum(axis=0)
        else:
            spike_counts = output_spikes

        # Reshape to [4, 4] for voting (4 action groups × 4 neurons each)
        # Map to 8 continuous actions
        if len(spike_counts) >= 16:
            votes = spike_counts[:16].reshape(4, 4).sum(axis=1)
            votes_norm = votes / (votes.sum() + 1e-8)

            # Map votes to action parameters
            action = AffectiveAction(
                attention_gain=1.0 + votes_norm[0] * 0.5,
                exploration_temp=0.5 + votes_norm[1] * 1.5,
                memory_write_prob=votes_norm[2],
                alert_level=int(votes_norm[3] * 3),
            )
        else:
            action = AffectiveAction()

        return action

    def run_inference(self, obs: torch.Tensor) -> AffectiveAction:
        """Run SNN inference (software simulation).

        For actual FPGA deployment, this would call the XRT/SYCL host.

        Args:
            obs: Observation tensor

        Returns:
            AffectiveAction
        """
        # Encode
        input_spikes = self.encode_observation(obs)

        # TODO: Call actual FPGA kernel
        # For now, simple software LIF simulation
        output_spikes = self._software_snn_simulate(input_spikes)

        # Decode
        return self.decode_action(output_spikes)

    def _software_snn_simulate(self, input_spikes: np.ndarray) -> np.ndarray:
        """Placeholder software SNN simulation."""
        # Simple pass-through for testing
        output_spikes = np.zeros((self.num_timesteps, self._num_output_neurons), dtype=np.uint8)

        # Random spikes based on input activity
        input_activity = input_spikes.mean()
        for t in range(self.num_timesteps):
            output_spikes[t] = (np.random.random(self._num_output_neurons) < input_activity * 0.1).astype(np.uint8)

        return output_spikes


__all__ = [
    # Action
    "AffectiveAction",
    # Observation
    "build_policy_observation",
    # Policies
    "heuristic_policy",
    "AffectivePolicyNetwork",
    # Main interface
    "PolicyRegulator",
    # SNN interface
    "SNNPolicyInterface",
]
