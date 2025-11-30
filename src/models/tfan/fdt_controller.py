"""FDT (Fluctuation-Dissipation Theorem) Controller for TF-A-N Training.

Implements adaptive learning rate and gradient clipping based on
training dynamics monitoring using a PI-D control scheme.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple


@dataclass
class FDTMetrics:
    """Metrics tracked by the FDT controller."""
    grad_norm: float = 0.0
    loss: float = 0.0
    lr: float = 0.0
    clip_value: float = 1.0
    fluctuation: float = 0.0
    dissipation: float = 0.0
    fdt_ratio: float = 1.0
    step: int = 0


@dataclass
class FDTConfig:
    """Configuration for FDT controller.

    Args:
        target_fdt_ratio: Target ratio of fluctuation to dissipation (default: 1.0)
        kp: Proportional gain (default: 0.1)
        ki: Integral gain (default: 0.01)
        kd: Derivative gain (default: 0.001)
        window_size: Size of moving average window (default: 100)
        min_lr_scale: Minimum learning rate scaling factor (default: 0.1)
        max_lr_scale: Maximum learning rate scaling factor (default: 2.0)
        min_clip: Minimum gradient clip value (default: 0.1)
        max_clip: Maximum gradient clip value (default: 10.0)
        warmup_steps: Steps before FDT control activates (default: 1000)
        update_interval: Steps between controller updates (default: 10)
    """
    target_fdt_ratio: float = 1.0
    kp: float = 0.1
    ki: float = 0.01
    kd: float = 0.001
    window_size: int = 100
    min_lr_scale: float = 0.1
    max_lr_scale: float = 2.0
    min_clip: float = 0.1
    max_clip: float = 10.0
    warmup_steps: int = 1000
    update_interval: int = 10


class FDTController:
    """Fluctuation-Dissipation Theorem controller for adaptive training.

    Monitors the relationship between gradient fluctuations (variance)
    and energy dissipation (loss decrease) to maintain stable training.

    The FDT ratio = fluctuation / dissipation should be approximately 1
    at equilibrium. Deviations indicate:
    - ratio > 1: Too much noise, reduce LR or increase clip
    - ratio < 1: Can potentially increase LR for faster training

    Args:
        config: FDTConfig instance
        base_lr: Base learning rate
        base_clip: Base gradient clipping value
    """

    def __init__(
        self,
        config: Optional[FDTConfig] = None,
        base_lr: float = 1e-4,
        base_clip: float = 1.0,
    ):
        self.config = config or FDTConfig()
        self.base_lr = base_lr
        self.base_clip = base_clip

        # Current state
        self.lr_scale = 1.0
        self.clip_value = base_clip

        # Moving windows for metrics
        self.grad_norms = deque(maxlen=self.config.window_size)
        self.losses = deque(maxlen=self.config.window_size)
        self.grad_variances = deque(maxlen=self.config.window_size)

        # PID controller state
        self.integral_error = 0.0
        self.prev_error = 0.0

        # Step counter
        self.step = 0

        # History for analysis
        self.history: List[FDTMetrics] = []

    def update(
        self,
        model: nn.Module,
        loss: torch.Tensor,
        grad_norm: Optional[float] = None,
    ) -> FDTMetrics:
        """Update controller with current training state.

        Args:
            model: The model being trained
            loss: Current loss value
            grad_norm: Pre-computed gradient norm (optional)

        Returns:
            FDTMetrics with current state
        """
        self.step += 1

        # Compute gradient norm if not provided
        if grad_norm is None:
            grad_norm = self._compute_grad_norm(model)

        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss

        # Update moving windows
        self.grad_norms.append(grad_norm)
        self.losses.append(loss_value)

        # Compute gradient variance (fluctuation proxy)
        if len(self.grad_norms) >= 2:
            grad_variance = self._compute_variance(list(self.grad_norms))
            self.grad_variances.append(grad_variance)

        # Compute metrics
        fluctuation = self._compute_fluctuation()
        dissipation = self._compute_dissipation()

        # FDT ratio
        fdt_ratio = fluctuation / max(dissipation, 1e-10)

        # Apply PID control after warmup
        if self.step > self.config.warmup_steps and self.step % self.config.update_interval == 0:
            self._pid_update(fdt_ratio)

        # Create metrics
        metrics = FDTMetrics(
            grad_norm=grad_norm,
            loss=loss_value,
            lr=self.base_lr * self.lr_scale,
            clip_value=self.clip_value,
            fluctuation=fluctuation,
            dissipation=dissipation,
            fdt_ratio=fdt_ratio,
            step=self.step,
        )

        self.history.append(metrics)

        return metrics

    def _compute_grad_norm(self, model: nn.Module) -> float:
        """Compute total gradient norm across model parameters."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm)

    def _compute_variance(self, values: List[float]) -> float:
        """Compute variance of a list of values."""
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance

    def _compute_fluctuation(self) -> float:
        """Compute fluctuation metric (gradient variance)."""
        if len(self.grad_variances) < 1:
            return 1.0
        return sum(self.grad_variances) / len(self.grad_variances) + 1e-10

    def _compute_dissipation(self) -> float:
        """Compute dissipation metric (loss decrease rate)."""
        if len(self.losses) < 2:
            return 1.0

        losses = list(self.losses)
        # Compute average loss decrease
        loss_decrease = sum(
            max(0, losses[i] - losses[i+1])
            for i in range(len(losses) - 1)
        ) / (len(losses) - 1)

        return loss_decrease + 1e-10

    def _pid_update(self, fdt_ratio: float):
        """Apply PID control to adjust learning rate scale."""
        # Error: deviation from target ratio
        error = fdt_ratio - self.config.target_fdt_ratio

        # Proportional term
        p_term = self.config.kp * error

        # Integral term (with anti-windup)
        self.integral_error += error
        self.integral_error = max(-10, min(10, self.integral_error))  # Anti-windup
        i_term = self.config.ki * self.integral_error

        # Derivative term
        d_term = self.config.kd * (error - self.prev_error)
        self.prev_error = error

        # Combined adjustment
        adjustment = p_term + i_term + d_term

        # Update learning rate scale
        # High FDT ratio -> reduce LR (negative adjustment)
        self.lr_scale *= math.exp(-adjustment)
        self.lr_scale = max(self.config.min_lr_scale, min(self.config.max_lr_scale, self.lr_scale))

        # Update gradient clipping
        # High FDT ratio -> increase clipping (reduce effective gradients)
        clip_adjustment = adjustment * 0.5  # Less aggressive
        self.clip_value *= math.exp(clip_adjustment)
        self.clip_value = max(self.config.min_clip, min(self.config.max_clip, self.clip_value))

    def get_lr(self) -> float:
        """Get current effective learning rate."""
        return self.base_lr * self.lr_scale

    def get_clip_value(self) -> float:
        """Get current gradient clipping value."""
        return self.clip_value

    def reset(self):
        """Reset controller state."""
        self.lr_scale = 1.0
        self.clip_value = self.base_clip
        self.grad_norms.clear()
        self.losses.clear()
        self.grad_variances.clear()
        self.integral_error = 0.0
        self.prev_error = 0.0
        self.step = 0
        self.history.clear()

    def state_dict(self) -> Dict[str, Any]:
        """Get controller state for checkpointing."""
        return {
            "lr_scale": self.lr_scale,
            "clip_value": self.clip_value,
            "integral_error": self.integral_error,
            "prev_error": self.prev_error,
            "step": self.step,
            "grad_norms": list(self.grad_norms),
            "losses": list(self.losses),
            "grad_variances": list(self.grad_variances),
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load controller state from checkpoint."""
        self.lr_scale = state_dict["lr_scale"]
        self.clip_value = state_dict["clip_value"]
        self.integral_error = state_dict["integral_error"]
        self.prev_error = state_dict["prev_error"]
        self.step = state_dict["step"]
        self.grad_norms = deque(state_dict["grad_norms"], maxlen=self.config.window_size)
        self.losses = deque(state_dict["losses"], maxlen=self.config.window_size)
        self.grad_variances = deque(state_dict["grad_variances"], maxlen=self.config.window_size)


class GradientClipper:
    """Gradient clipping utility with FDT integration.

    Args:
        max_norm: Maximum gradient norm
        norm_type: Norm type (default: 2.0)
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, model: nn.Module, max_norm: Optional[float] = None) -> float:
        """Clip gradients and return the total norm before clipping.

        Args:
            model: Model with gradients
            max_norm: Override max norm (e.g., from FDT controller)

        Returns:
            Total gradient norm before clipping
        """
        max_norm = max_norm or self.max_norm
        parameters = [p for p in model.parameters() if p.grad is not None]

        if len(parameters) == 0:
            return 0.0

        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, max_norm, norm_type=self.norm_type
        )

        return total_norm.item() if isinstance(total_norm, torch.Tensor) else total_norm


class LearningRateScheduler:
    """Learning rate scheduler with warmup and FDT integration.

    Implements linear warmup followed by cosine decay, with
    optional FDT-based learning rate scaling.

    Args:
        base_lr: Base learning rate
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum LR as ratio of base (default: 0.1)
        fdt_controller: Optional FDT controller for adaptive scaling
    """

    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
        fdt_controller: Optional[FDTController] = None,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio
        self.fdt_controller = fdt_controller

    def get_lr(self, step: int) -> float:
        """Get learning rate for given step.

        Args:
            step: Current training step

        Returns:
            Learning rate value
        """
        if step < self.warmup_steps:
            # Linear warmup
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # Cosine decay
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

        # Apply FDT scaling if available
        if self.fdt_controller is not None:
            lr *= self.fdt_controller.lr_scale

        return lr

    def step(self, optimizer: torch.optim.Optimizer, step: int):
        """Update optimizer learning rate.

        Args:
            optimizer: PyTorch optimizer
            step: Current training step
        """
        lr = self.get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


__all__ = [
    "FDTController",
    "FDTConfig",
    "FDTMetrics",
    "GradientClipper",
    "LearningRateScheduler",
]
