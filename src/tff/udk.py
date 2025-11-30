"""UDKController - Unified Dynamics and Knowledge Controller.

This module implements the UDK controller with three proxies:
1. σ (Sigma) - Belief revision cost proxy
2. ε (Epsilon) - Dissipation/turbulence proxy
3. κ (Kappa) - Fisher Information Matrix curvature proxy

These proxies compose into the UTCF (Unified Topological Control Field)
that guides learning dynamics and knowledge integration.

Key insight: By tracking these information-theoretic quantities, we can
adaptively control learning rate, memory consolidation, and exploration.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from collections import deque
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlSignal(Enum):
    """Control signals produced by UDK."""

    ACCELERATE = "accelerate"  # Increase learning rate
    DECELERATE = "decelerate"  # Decrease learning rate
    STABILIZE = "stabilize"  # Hold steady
    EXPLORE = "explore"  # Increase temperature
    EXPLOIT = "exploit"  # Decrease temperature
    CONSOLIDATE = "consolidate"  # Memory write
    FORGET = "forget"  # Memory pruning


@dataclass
class SigmaConfig:
    """Configuration for σ (belief revision cost) proxy."""

    eps: float = 1e-6  # Numerical stability
    ema_decay: float = 0.95  # EMA for smoothing
    threshold_high: float = 0.8  # High revision cost threshold
    threshold_low: float = 0.2  # Low revision cost threshold


@dataclass
class EpsilonConfig:
    """Configuration for ε (dissipation/turbulence) proxy."""

    window_size: int = 100  # History window for variance
    eps: float = 1e-6
    ema_decay: float = 0.9
    threshold_turbulent: float = 0.5
    threshold_laminar: float = 0.1


@dataclass
class KappaConfig:
    """Configuration for κ (FIM curvature) proxy."""

    num_samples: int = 10  # Samples for FIM estimation
    eps: float = 1e-6
    ema_decay: float = 0.95
    damping: float = 1e-4  # For numerical stability
    threshold_high_curv: float = 10.0
    threshold_low_curv: float = 0.1


@dataclass
class UDKConfig:
    """Configuration for UDKController."""

    sigma_config: SigmaConfig = field(default_factory=SigmaConfig)
    epsilon_config: EpsilonConfig = field(default_factory=EpsilonConfig)
    kappa_config: KappaConfig = field(default_factory=KappaConfig)

    # UTCF composition weights
    sigma_weight: float = 1.0
    epsilon_weight: float = 1.0
    kappa_weight: float = 1.0

    # Control thresholds
    utcf_high: float = 0.7
    utcf_low: float = 0.3

    # Adaptive learning rate bounds
    lr_min_mult: float = 0.1
    lr_max_mult: float = 10.0


class SigmaProxy(nn.Module):
    """σ (Sigma) - Belief Revision Cost Proxy.

    Measures the cost of updating beliefs given new evidence.

    σ = log(1 + cost_R) / (ε + precision_gain)

    Where:
    - cost_R: KL divergence from prior to posterior
    - precision_gain: reduction in uncertainty

    High σ: expensive updates, be cautious
    Low σ: cheap updates, can be aggressive
    """

    def __init__(self, config: SigmaConfig):
        super().__init__()
        self.config = config

        # EMA state
        self.register_buffer("ema_sigma", torch.tensor(0.5))
        self.register_buffer("ema_cost", torch.tensor(0.0))
        self.register_buffer("ema_gain", torch.tensor(1.0))

    def compute_kl_cost(
        self,
        prior_logits: torch.Tensor,
        posterior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence from prior to posterior."""
        prior_probs = F.softmax(prior_logits, dim=-1)
        posterior_probs = F.softmax(posterior_logits, dim=-1)

        kl = F.kl_div(
            prior_probs.log(),
            posterior_probs,
            reduction='batchmean',
            log_target=False,
        )

        return kl

    def compute_precision_gain(
        self,
        prior_uncertainty: torch.Tensor,
        posterior_uncertainty: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reduction in uncertainty."""
        # Uncertainty can be entropy, variance, or calibration error
        gain = prior_uncertainty - posterior_uncertainty
        return torch.clamp(gain, min=0)

    def forward(
        self,
        cost_R: torch.Tensor,
        precision_gain: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute σ proxy value.

        Args:
            cost_R: Belief revision cost (e.g., KL divergence)
            precision_gain: Uncertainty reduction

        Returns:
            Dict with sigma value and diagnostics
        """
        # Core computation
        sigma = torch.log1p(cost_R) / (self.config.eps + precision_gain)

        # Update EMA
        if self.training:
            self.ema_sigma = (
                self.config.ema_decay * self.ema_sigma +
                (1 - self.config.ema_decay) * sigma.mean()
            )
            self.ema_cost = (
                self.config.ema_decay * self.ema_cost +
                (1 - self.config.ema_decay) * cost_R.mean()
            )
            self.ema_gain = (
                self.config.ema_decay * self.ema_gain +
                (1 - self.config.ema_decay) * precision_gain.mean()
            )

        # Classify regime
        if sigma.mean() > self.config.threshold_high:
            regime = "high_cost"
        elif sigma.mean() < self.config.threshold_low:
            regime = "low_cost"
        else:
            regime = "moderate"

        return {
            "sigma": sigma,
            "sigma_mean": sigma.mean(),
            "sigma_ema": self.ema_sigma,
            "cost_R": cost_R.mean(),
            "precision_gain": precision_gain.mean(),
            "regime": regime,
        }


class EpsilonProxy(nn.Module):
    """ε (Epsilon) - Dissipation/Turbulence Proxy.

    Measures the level of turbulence in gradient flow.

    ε = Var(∇L) / (mean(|∇L|) + δ)

    High ε: turbulent dynamics, need stabilization
    Low ε: laminar flow, can accelerate
    """

    def __init__(self, config: EpsilonConfig):
        super().__init__()
        self.config = config

        # Gradient history
        self.grad_history: deque = deque(maxlen=config.window_size)

        # EMA state
        self.register_buffer("ema_epsilon", torch.tensor(0.5))
        self.register_buffer("ema_grad_var", torch.tensor(0.0))
        self.register_buffer("ema_grad_mean", torch.tensor(1.0))

    def update_history(self, gradients: torch.Tensor) -> None:
        """Update gradient history."""
        self.grad_history.append(gradients.detach().clone())

    def compute_from_history(self) -> torch.Tensor:
        """Compute ε from gradient history."""
        if len(self.grad_history) < 2:
            return torch.tensor(0.5)

        # Stack history
        grads = torch.stack(list(self.grad_history), dim=0)

        # Compute variance and mean
        grad_var = grads.var(dim=0).mean()
        grad_mean = grads.abs().mean()

        epsilon = grad_var / (grad_mean + self.config.eps)

        return epsilon

    def forward(
        self,
        gradients: Optional[torch.Tensor] = None,
        loss_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute ε proxy value.

        Can use either raw gradients or loss trajectory.

        Args:
            gradients: Gradient tensor (flattened)
            loss_values: Recent loss values

        Returns:
            Dict with epsilon value and diagnostics
        """
        if gradients is not None:
            self.update_history(gradients.flatten())

        epsilon = self.compute_from_history()

        # Alternative: compute from loss values
        if loss_values is not None and len(loss_values) >= 2:
            loss_var = loss_values.var()
            loss_mean = loss_values.abs().mean()
            epsilon_loss = loss_var / (loss_mean + self.config.eps)
            # Blend with gradient-based estimate
            epsilon = 0.7 * epsilon + 0.3 * epsilon_loss

        # Update EMA
        if self.training:
            self.ema_epsilon = (
                self.config.ema_decay * self.ema_epsilon +
                (1 - self.config.ema_decay) * epsilon
            )

        # Classify regime
        if epsilon > self.config.threshold_turbulent:
            regime = "turbulent"
        elif epsilon < self.config.threshold_laminar:
            regime = "laminar"
        else:
            regime = "transitional"

        return {
            "epsilon": epsilon,
            "epsilon_ema": self.ema_epsilon,
            "regime": regime,
            "history_size": len(self.grad_history),
        }


class KappaProxy(nn.Module):
    """κ (Kappa) - Fisher Information Matrix Curvature Proxy.

    Measures the curvature of the loss landscape via FIM.

    κ = trace(F) / dim(θ)

    Where F is the Fisher Information Matrix.

    High κ: sharp curvature, small steps needed
    Low κ: flat region, can take larger steps
    """

    def __init__(self, config: KappaConfig):
        super().__init__()
        self.config = config

        # EMA state
        self.register_buffer("ema_kappa", torch.tensor(1.0))
        self.register_buffer("ema_trace", torch.tensor(0.0))

    def compute_fim_diagonal(
        self,
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute diagonal of FIM via gradient outer product.

        Uses empirical Fisher approximation:
        F_ii ≈ E[(∂L/∂θ_i)²]
        """
        model.zero_grad()

        fim_diag = []

        for _ in range(self.config.num_samples):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward(retain_graph=True)

            # Collect squared gradients
            sample_diag = []
            for param in model.parameters():
                if param.grad is not None:
                    sample_diag.append(param.grad.flatten() ** 2)

            if sample_diag:
                fim_diag.append(torch.cat(sample_diag))

            model.zero_grad()

        if not fim_diag:
            return torch.tensor(0.0)

        # Average over samples
        fim_diag_mean = torch.stack(fim_diag).mean(dim=0)

        return fim_diag_mean

    def compute_kappa_from_fim(
        self,
        fim_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute κ from FIM diagonal."""
        # Add damping for stability
        fim_diag = fim_diag + self.config.damping

        # Trace / dimension
        trace = fim_diag.sum()
        dim = fim_diag.numel()

        kappa = trace / (dim + self.config.eps)

        return kappa

    def forward(
        self,
        fim_diag: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
        hessian_diag: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute κ proxy value.

        Can use FIM diagonal directly or approximate from gradients.

        Args:
            fim_diag: Diagonal of FIM
            grad_norm: Gradient norm (rough proxy)
            hessian_diag: Diagonal of Hessian (alternative)

        Returns:
            Dict with kappa value and diagnostics
        """
        if fim_diag is not None:
            kappa = self.compute_kappa_from_fim(fim_diag)
        elif hessian_diag is not None:
            # Use Hessian diagonal as proxy
            kappa = hessian_diag.abs().mean()
        elif grad_norm is not None:
            # Very rough proxy: grad norm² ≈ trace(F)
            kappa = grad_norm ** 2
        else:
            kappa = self.ema_kappa

        # Update EMA
        if self.training:
            self.ema_kappa = (
                self.config.ema_decay * self.ema_kappa +
                (1 - self.config.ema_decay) * kappa
            )

        # Classify regime
        if kappa > self.config.threshold_high_curv:
            regime = "high_curvature"
        elif kappa < self.config.threshold_low_curv:
            regime = "low_curvature"
        else:
            regime = "moderate_curvature"

        return {
            "kappa": kappa,
            "kappa_ema": self.ema_kappa,
            "regime": regime,
        }


class UDKController(nn.Module):
    """Unified Dynamics and Knowledge Controller.

    Composes σ, ε, κ proxies into a unified control field (UTCF)
    that provides adaptive signals for learning dynamics.

    UTCF = w_σ · σ + w_ε · ε + w_κ · κ (normalized)

    Control outputs:
    - learning_rate_mult: Multiplier for base LR
    - temperature: Exploration temperature
    - memory_write_prob: Probability of memory consolidation
    - control_signals: Discrete action recommendations
    """

    def __init__(self, config: UDKConfig):
        super().__init__()
        self.config = config

        # Proxies
        self.sigma_proxy = SigmaProxy(config.sigma_config)
        self.epsilon_proxy = EpsilonProxy(config.epsilon_config)
        self.kappa_proxy = KappaProxy(config.kappa_config)

        # Learnable composition weights (optional)
        self.use_learned_weights = False
        self.sigma_w = nn.Parameter(torch.tensor(config.sigma_weight))
        self.epsilon_w = nn.Parameter(torch.tensor(config.epsilon_weight))
        self.kappa_w = nn.Parameter(torch.tensor(config.kappa_weight))

        # Control history
        self.utcf_history: deque = deque(maxlen=100)

    def compute_utcf(
        self,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        kappa: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Unified Topological Control Field.

        Normalizes and combines the three proxies.
        """
        # Normalize each to [0, 1] range (roughly)
        sigma_norm = torch.sigmoid(sigma)
        epsilon_norm = torch.sigmoid(epsilon)
        kappa_norm = torch.sigmoid(torch.log1p(kappa))

        # Weighted sum
        if self.use_learned_weights:
            w_sigma = torch.softplus(self.sigma_w)
            w_epsilon = torch.softplus(self.epsilon_w)
            w_kappa = torch.softplus(self.kappa_w)
        else:
            w_sigma = self.config.sigma_weight
            w_epsilon = self.config.epsilon_weight
            w_kappa = self.config.kappa_weight

        total_w = w_sigma + w_epsilon + w_kappa + 1e-8

        utcf = (
            w_sigma * sigma_norm +
            w_epsilon * epsilon_norm +
            w_kappa * kappa_norm
        ) / total_w

        return utcf

    def compute_lr_multiplier(self, utcf: torch.Tensor) -> torch.Tensor:
        """Compute learning rate multiplier from UTCF.

        High UTCF → low LR (careful)
        Low UTCF → high LR (aggressive)
        """
        # Inverse mapping: high UTCF = low LR
        lr_mult = self.config.lr_max_mult - (
            (self.config.lr_max_mult - self.config.lr_min_mult) * utcf
        )

        # Clamp to bounds
        lr_mult = torch.clamp(
            lr_mult,
            self.config.lr_min_mult,
            self.config.lr_max_mult,
        )

        return lr_mult

    def compute_temperature(self, utcf: torch.Tensor) -> torch.Tensor:
        """Compute exploration temperature from UTCF.

        High UTCF → high temperature (explore)
        Low UTCF → low temperature (exploit)
        """
        # Direct mapping
        base_temp = 1.0
        temp_range = 2.0  # [0.5, 2.5]

        temperature = base_temp + (utcf - 0.5) * temp_range

        return torch.clamp(temperature, 0.1, 5.0)

    def compute_memory_write_prob(
        self,
        utcf: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Compute memory write probability.

        Consolidate when:
        - UTCF is stable (moderate)
        - Epsilon is low (laminar flow)
        """
        # High write prob in stable regions
        stability = 1 - torch.abs(utcf - 0.5) * 2  # Peak at utcf=0.5
        laminar_bonus = 1 - epsilon

        write_prob = 0.5 * stability + 0.5 * laminar_bonus

        return torch.clamp(write_prob, 0.0, 1.0)

    def get_control_signals(
        self,
        utcf: torch.Tensor,
        sigma_regime: str,
        epsilon_regime: str,
        kappa_regime: str,
    ) -> List[ControlSignal]:
        """Generate discrete control signals from proxies."""
        signals = []

        # UTCF-based signals
        if utcf > self.config.utcf_high:
            signals.append(ControlSignal.DECELERATE)
            signals.append(ControlSignal.EXPLORE)
        elif utcf < self.config.utcf_low:
            signals.append(ControlSignal.ACCELERATE)
            signals.append(ControlSignal.EXPLOIT)
        else:
            signals.append(ControlSignal.STABILIZE)

        # Regime-specific signals
        if epsilon_regime == "turbulent":
            signals.append(ControlSignal.DECELERATE)

        if epsilon_regime == "laminar" and kappa_regime == "low_curvature":
            signals.append(ControlSignal.CONSOLIDATE)

        if sigma_regime == "high_cost":
            signals.append(ControlSignal.EXPLORE)

        return signals

    def forward(
        self,
        # Sigma inputs
        cost_R: Optional[torch.Tensor] = None,
        precision_gain: Optional[torch.Tensor] = None,
        # Epsilon inputs
        gradients: Optional[torch.Tensor] = None,
        loss_values: Optional[torch.Tensor] = None,
        # Kappa inputs
        fim_diag: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run full UDK control cycle.

        Args:
            cost_R: Belief revision cost
            precision_gain: Uncertainty reduction
            gradients: Gradient tensor
            loss_values: Recent losses
            fim_diag: FIM diagonal
            grad_norm: Gradient norm

        Returns:
            Dict with control outputs and diagnostics
        """
        # Compute individual proxies
        sigma_out = self.sigma_proxy(
            cost_R if cost_R is not None else torch.tensor(0.0),
            precision_gain if precision_gain is not None else torch.tensor(1.0),
        )

        epsilon_out = self.epsilon_proxy(
            gradients=gradients,
            loss_values=loss_values,
        )

        kappa_out = self.kappa_proxy(
            fim_diag=fim_diag,
            grad_norm=grad_norm,
        )

        # Compute UTCF
        utcf = self.compute_utcf(
            sigma_out["sigma_ema"],
            epsilon_out["epsilon_ema"],
            kappa_out["kappa_ema"],
        )

        # Update history
        self.utcf_history.append(utcf.item())

        # Compute control outputs
        lr_mult = self.compute_lr_multiplier(utcf)
        temperature = self.compute_temperature(utcf)
        memory_write_prob = self.compute_memory_write_prob(
            utcf, epsilon_out["epsilon_ema"]
        )

        # Generate discrete signals
        signals = self.get_control_signals(
            utcf,
            sigma_out["regime"],
            epsilon_out["regime"],
            kappa_out["regime"],
        )

        return {
            # Control outputs
            "utcf": utcf,
            "learning_rate_mult": lr_mult,
            "temperature": temperature,
            "memory_write_prob": memory_write_prob,
            "control_signals": signals,
            # Proxy values
            "sigma": sigma_out,
            "epsilon": epsilon_out,
            "kappa": kappa_out,
            # Diagnostics
            "utcf_history_mean": sum(self.utcf_history) / len(self.utcf_history) if self.utcf_history else 0.5,
        }


class UDKOptimizer:
    """Optimizer wrapper that uses UDK for adaptive control.

    Wraps a base optimizer and adjusts learning rate based on UDK signals.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        udk_controller: UDKController,
        base_lr: float = 1e-3,
    ):
        self.optimizer = base_optimizer
        self.udk = udk_controller
        self.base_lr = base_lr

        # Track state
        self.step_count = 0
        self.current_lr_mult = 1.0

    def step(
        self,
        loss: torch.Tensor,
        **udk_kwargs,
    ) -> Dict[str, Any]:
        """Perform optimization step with UDK control.

        Args:
            loss: Current loss value
            **udk_kwargs: Arguments for UDK forward pass

        Returns:
            Dict with step info
        """
        # Get gradients
        gradients = []
        grad_norm_sq = 0.0

        for param_group in self.optimizer.param_groups:
            for param in param_group['params']:
                if param.grad is not None:
                    gradients.append(param.grad.flatten())
                    grad_norm_sq += param.grad.norm() ** 2

        if gradients:
            gradients = torch.cat(gradients)
            grad_norm = torch.sqrt(torch.tensor(grad_norm_sq))
        else:
            gradients = None
            grad_norm = None

        # Run UDK
        udk_out = self.udk(
            gradients=gradients,
            grad_norm=grad_norm,
            **udk_kwargs,
        )

        # Update learning rate
        lr_mult = udk_out["learning_rate_mult"].item()
        self.current_lr_mult = lr_mult

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.base_lr * lr_mult

        # Perform optimizer step
        self.optimizer.step()
        self.step_count += 1

        return {
            "step": self.step_count,
            "lr": self.base_lr * lr_mult,
            "lr_mult": lr_mult,
            **udk_out,
        }

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()


def create_udk_controller() -> UDKController:
    """Factory function for UDKController with defaults."""
    return UDKController(UDKConfig())
