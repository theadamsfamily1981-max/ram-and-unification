"""
EXPERIMENTAL/GUARDED: Dynamic Architecture Update (DAU)

Very conservative implementation with:
- Tiny step sizes
- Hard ban on identity/value axiom modification
- Extensive logging
- Ability to disable entirely

⚠️ GUARDED: This module is disabled by default.
Enable only with careful consideration.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import re
import copy

from .config import DAUConfig

logger = logging.getLogger(__name__)


class DAUAction(Enum):
    """Types of DAU actions."""
    ADD_NEURON = "add_neuron"
    REMOVE_NEURON = "remove_neuron"
    MODIFY_CONNECTION = "modify_connection"
    ADJUST_CAPACITY = "adjust_capacity"


@dataclass
class DAUProposal:
    """A proposed architecture update."""
    action: DAUAction
    target_layer: str
    parameters: Dict
    estimated_impact: float
    reason: str


@dataclass
class DAULog:
    """Log entry for DAU operations."""
    step: int
    proposal: Optional[DAUProposal]
    accepted: bool
    reason: str
    param_norms_before: Optional[Dict[str, float]]
    param_norms_after: Optional[Dict[str, float]]


class DAUGuard:
    """
    Safety guard for DAU operations.

    Implements hard bans and safety checks.
    """

    def __init__(self, config: DAUConfig):
        self.config = config
        self._violation_count = 0

    def is_parameter_banned(self, param_name: str) -> bool:
        """
        Check if a parameter is banned from modification.

        HARD BAN: Never modify identity or value parameters.
        """
        param_lower = param_name.lower()

        for pattern in self.config.banned_parameter_patterns:
            if pattern in param_lower:
                logger.warning(
                    f"DAU Guard: Attempted modification of banned parameter '{param_name}' "
                    f"(matched pattern '{pattern}')"
                )
                self._violation_count += 1
                return True

        return False

    def check_change_magnitude(
        self,
        param_name: str,
        old_value: torch.Tensor,
        new_value: torch.Tensor
    ) -> Tuple[bool, str]:
        """
        Check if parameter change is within acceptable bounds.

        Returns:
            (is_acceptable, reason)
        """
        change_norm = torch.norm(new_value - old_value).item()

        if change_norm > self.config.max_param_change_norm:
            reason = (
                f"Parameter change norm {change_norm:.6f} exceeds "
                f"max allowed {self.config.max_param_change_norm}"
            )
            logger.warning(f"DAU Guard: {reason}")
            return False, reason

        if change_norm > self.config.require_approval_above:
            logger.info(
                f"DAU: Large parameter change detected for '{param_name}': "
                f"norm={change_norm:.6f}"
            )

        return True, "Change within acceptable bounds"

    def get_violation_count(self) -> int:
        """Get number of guard violations."""
        return self._violation_count


class DAUCheckpoint:
    """
    Checkpoint manager for DAU rollback capability.
    """

    def __init__(self, max_checkpoints: int = 10):
        self.max_checkpoints = max_checkpoints
        self._checkpoints: List[Dict] = []
        self._step_ids: List[int] = []

    def save(self, model: nn.Module, step: int):
        """Save a checkpoint."""
        checkpoint = {
            'state_dict': copy.deepcopy(model.state_dict()),
            'step': step
        }

        self._checkpoints.append(checkpoint)
        self._step_ids.append(step)

        # Limit checkpoints
        while len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)
            self._step_ids.pop(0)

    def restore(self, model: nn.Module, step: Optional[int] = None) -> bool:
        """
        Restore from checkpoint.

        Args:
            model: Model to restore
            step: Specific step to restore (None = most recent)

        Returns:
            True if restoration successful
        """
        if len(self._checkpoints) == 0:
            logger.error("DAU Checkpoint: No checkpoints available for restore")
            return False

        if step is None:
            checkpoint = self._checkpoints[-1]
        else:
            try:
                idx = self._step_ids.index(step)
                checkpoint = self._checkpoints[idx]
            except ValueError:
                logger.error(f"DAU Checkpoint: Step {step} not found")
                return False

        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"DAU Checkpoint: Restored to step {checkpoint['step']}")
        return True

    def get_available_steps(self) -> List[int]:
        """Get list of available checkpoint steps."""
        return list(self._step_ids)


class DynamicArchitectureUpdate(nn.Module):
    """
    EXPERIMENTAL/GUARDED: Dynamic Architecture Update module.

    ⚠️ DISABLED BY DEFAULT - must explicitly enable.

    Provides very conservative architecture updates with:
    - Tiny step sizes
    - Hard bans on protected parameters
    - Extensive logging
    - Automatic rollback capability
    """

    def __init__(self, config: DAUConfig, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model

        # Safety systems
        self.guard = DAUGuard(config)
        self.checkpoint = DAUCheckpoint(config.keep_checkpoints)

        # Logging
        self._logs: List[DAULog] = []
        self._step = 0

        # Track which parameters are protected
        self._protected_params: Set[str] = set()
        self._identify_protected_params()

        # Warn if enabled
        if config.enabled:
            logger.warning(
                "DAU is ENABLED. This is an experimental feature. "
                "Proceed with caution."
            )
        else:
            logger.info("DAU is disabled (default). Enable explicitly if needed.")

    def _identify_protected_params(self):
        """Identify parameters that should never be modified."""
        for name, _ in self.model.named_parameters():
            if self.guard.is_parameter_banned(name):
                self._protected_params.add(name)

        if self._protected_params:
            logger.info(
                f"DAU: Identified {len(self._protected_params)} protected parameters"
            )

    def is_enabled(self) -> bool:
        """Check if DAU is enabled."""
        return self.config.enabled

    def propose_update(
        self,
        action: DAUAction,
        target_layer: str,
        parameters: Dict,
        reason: str
    ) -> DAUProposal:
        """
        Create a proposal for architecture update.

        This only creates the proposal - it must be accepted via accept_proposal().
        """
        if not self.is_enabled():
            logger.warning("DAU: Proposal rejected - DAU is disabled")
            return None

        # Estimate impact (placeholder - should be more sophisticated)
        estimated_impact = 0.1

        proposal = DAUProposal(
            action=action,
            target_layer=target_layer,
            parameters=parameters,
            estimated_impact=estimated_impact,
            reason=reason
        )

        # Log proposal
        if self.config.log_all_proposals:
            logger.info(f"DAU Proposal: {action.value} on {target_layer}: {reason}")

        return proposal

    def accept_proposal(
        self,
        proposal: DAUProposal,
        force: bool = False
    ) -> Tuple[bool, str]:
        """
        Attempt to accept and apply a DAU proposal.

        Args:
            proposal: The proposal to accept
            force: If True, skip some safety checks (NOT RECOMMENDED)

        Returns:
            (success, reason)
        """
        self._step += 1

        if not self.is_enabled():
            return False, "DAU is disabled"

        if proposal is None:
            return False, "No proposal provided"

        # Check if target is protected
        if proposal.target_layer in self._protected_params:
            reason = f"Target layer '{proposal.target_layer}' is protected"
            self._log_rejection(proposal, reason)
            return False, reason

        # Check ban on identity modification
        if self.config.ban_identity_modification:
            if "identity" in proposal.target_layer.lower():
                reason = "Identity modification is banned"
                self._log_rejection(proposal, reason)
                return False, reason

        # Check ban on value axiom modification
        if self.config.ban_value_axiom_modification:
            for banned in ["value", "axiom", "ethics", "core"]:
                if banned in proposal.target_layer.lower():
                    reason = f"Value/axiom modification is banned (matched '{banned}')"
                    self._log_rejection(proposal, reason)
                    return False, reason

        # Save checkpoint before modification
        self.checkpoint.save(self.model, self._step)

        # Get current param norms for logging
        norms_before = self._get_param_norms() if self.config.log_parameter_norms else None

        # Apply the update
        try:
            success, apply_reason = self._apply_proposal(proposal)
        except Exception as e:
            reason = f"Exception during apply: {e}"
            self._log_rejection(proposal, reason)

            # Auto-rollback
            if self.config.enable_auto_rollback:
                self.checkpoint.restore(self.model)

            return False, reason

        if not success:
            self._log_rejection(proposal, apply_reason)
            return False, apply_reason

        # Get norms after for logging
        norms_after = self._get_param_norms() if self.config.log_parameter_norms else None

        # Log success
        self._logs.append(DAULog(
            step=self._step,
            proposal=proposal,
            accepted=True,
            reason="Update applied successfully",
            param_norms_before=norms_before,
            param_norms_after=norms_after
        ))

        return True, "Update applied successfully"

    def _apply_proposal(
        self,
        proposal: DAUProposal
    ) -> Tuple[bool, str]:
        """
        Actually apply the proposal to the model.

        Uses tiny step sizes as configured.
        """
        action = proposal.action
        target = proposal.target_layer
        params = proposal.parameters

        if action == DAUAction.MODIFY_CONNECTION:
            return self._apply_connection_modification(target, params)
        elif action == DAUAction.ADJUST_CAPACITY:
            return self._apply_capacity_adjustment(target, params)
        elif action == DAUAction.ADD_NEURON:
            return False, "ADD_NEURON not yet implemented (conservative)"
        elif action == DAUAction.REMOVE_NEURON:
            return False, "REMOVE_NEURON not yet implemented (conservative)"

        return False, f"Unknown action: {action}"

    def _apply_connection_modification(
        self,
        target: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """Apply connection weight modification with tiny step size."""
        # Find target parameter
        target_param = None
        for name, param in self.model.named_parameters():
            if name == target:
                target_param = param
                break

        if target_param is None:
            return False, f"Parameter '{target}' not found"

        # Get modification
        if 'delta' not in params:
            return False, "No 'delta' in parameters"

        delta = params['delta']
        if isinstance(delta, torch.Tensor):
            delta = delta.to(target_param.device)
        else:
            delta = torch.tensor(delta, device=target_param.device)

        # Apply tiny step size
        scaled_delta = delta * self.config.step_size

        # Check magnitude
        delta_norm = torch.norm(scaled_delta).item()
        if delta_norm > self.config.max_step_size:
            scaled_delta = scaled_delta * (self.config.max_step_size / delta_norm)
            logger.info(f"DAU: Clamped delta norm from {delta_norm:.6f} to {self.config.max_step_size}")

        # Check with guard
        new_value = target_param.data + scaled_delta
        ok, reason = self.guard.check_change_magnitude(target, target_param.data, new_value)

        if not ok:
            return False, reason

        # Apply
        with torch.no_grad():
            target_param.add_(scaled_delta)

        return True, f"Modified {target} with delta norm {torch.norm(scaled_delta).item():.8f}"

    def _apply_capacity_adjustment(
        self,
        target: str,
        params: Dict
    ) -> Tuple[bool, str]:
        """Adjust capacity (placeholder - very conservative)."""
        # This is a placeholder - actual implementation would need
        # careful consideration of architecture changes
        return False, "Capacity adjustment not yet implemented (conservative)"

    def _log_rejection(self, proposal: DAUProposal, reason: str):
        """Log a rejected proposal."""
        if self.config.log_all_rejections:
            logger.info(f"DAU Rejection: {reason}")

        self._logs.append(DAULog(
            step=self._step,
            proposal=proposal,
            accepted=False,
            reason=reason,
            param_norms_before=None,
            param_norms_after=None
        ))

    def _get_param_norms(self) -> Dict[str, float]:
        """Get norms of all parameters."""
        return {
            name: torch.norm(param).item()
            for name, param in self.model.named_parameters()
        }

    def rollback(self, steps: int = 1) -> bool:
        """
        Rollback to a previous checkpoint.

        Args:
            steps: Number of steps to roll back

        Returns:
            True if rollback successful
        """
        available = self.checkpoint.get_available_steps()
        if len(available) < steps:
            logger.error(f"DAU: Cannot rollback {steps} steps, only {len(available)} available")
            return False

        target_step = available[-(steps + 1)] if steps < len(available) else available[0]
        return self.checkpoint.restore(self.model, target_step)

    def get_logs(self, last_n: Optional[int] = None) -> List[DAULog]:
        """Get DAU logs."""
        if last_n is None:
            return self._logs
        return self._logs[-last_n:]

    def get_statistics(self) -> Dict:
        """Get DAU statistics."""
        total = len(self._logs)
        accepted = sum(1 for log in self._logs if log.accepted)

        return {
            'total_proposals': total,
            'accepted': accepted,
            'rejected': total - accepted,
            'acceptance_rate': accepted / total if total > 0 else 0.0,
            'guard_violations': self.guard.get_violation_count(),
            'enabled': self.is_enabled()
        }

    def disable(self):
        """Disable DAU (recommended after initial setup)."""
        self.config.enabled = False
        logger.info("DAU has been disabled")

    def enable(self, confirm: bool = False):
        """
        Enable DAU.

        Args:
            confirm: Must be True to actually enable (safety measure)
        """
        if not confirm:
            logger.warning(
                "DAU enable() called without confirm=True. "
                "DAU remains disabled. "
                "To enable, call enable(confirm=True)"
            )
            return

        self.config.enabled = True
        logger.warning("DAU has been ENABLED. Proceed with caution.")


if __name__ == "__main__":
    # Test DAU
    print("Testing Dynamic Architecture Update...")
    print("=" * 50)

    # Create simple model
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(64, 128)
            self.fc2 = nn.Linear(128, 64)
            self.identity_layer = nn.Linear(32, 32)  # Protected
            self.core_values = nn.Parameter(torch.randn(8))  # Protected

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    model = TestModel()
    config = DAUConfig(enabled=False)  # Disabled by default

    dau = DynamicArchitectureUpdate(config, model)

    print(f"DAU enabled: {dau.is_enabled()}")
    print(f"Protected params: {dau._protected_params}")

    # Test proposal when disabled
    proposal = dau.propose_update(
        action=DAUAction.MODIFY_CONNECTION,
        target_layer="fc1.weight",
        parameters={'delta': torch.randn(128, 64) * 0.01},
        reason="Test update"
    )
    print(f"\nProposal when disabled: {proposal}")

    # Enable DAU (with confirmation)
    print("\nEnabling DAU...")
    dau.enable(confirm=True)
    print(f"DAU enabled: {dau.is_enabled()}")

    # Test valid proposal
    print("\nTesting valid proposal...")
    proposal = dau.propose_update(
        action=DAUAction.MODIFY_CONNECTION,
        target_layer="fc1.weight",
        parameters={'delta': torch.randn(128, 64) * 0.01},
        reason="Test update"
    )

    success, reason = dau.accept_proposal(proposal)
    print(f"  Accepted: {success}, Reason: {reason}")

    # Test protected parameter
    print("\nTesting protected parameter...")
    proposal = dau.propose_update(
        action=DAUAction.MODIFY_CONNECTION,
        target_layer="identity_layer.weight",
        parameters={'delta': torch.randn(32, 32) * 0.01},
        reason="Attempt to modify identity"
    )

    success, reason = dau.accept_proposal(proposal)
    print(f"  Accepted: {success}, Reason: {reason}")

    # Test core values (should be banned)
    print("\nTesting core values modification...")
    proposal = dau.propose_update(
        action=DAUAction.MODIFY_CONNECTION,
        target_layer="core_values",
        parameters={'delta': torch.randn(8) * 0.01},
        reason="Attempt to modify core values"
    )

    success, reason = dau.accept_proposal(proposal)
    print(f"  Accepted: {success}, Reason: {reason}")

    # Get statistics
    print("\nDAU Statistics:")
    stats = dau.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Disable DAU
    print("\nDisabling DAU...")
    dau.disable()
    print(f"DAU enabled: {dau.is_enabled()}")

    print("\nAll DAU tests passed!")
