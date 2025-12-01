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

    # UTCF composition weights (per T-FAN spec)
    # UTCF = α_σ·σ + α_ε·ε + α_topo·L_topo + α_κ·κ
    alpha_sigma: float = 1.0
    alpha_epsilon: float = 1.0
    alpha_topo: float = 1.0
    alpha_kappa: float = 1.0

    # Legacy aliases
    sigma_weight: float = 1.0
    epsilon_weight: float = 1.0
    kappa_weight: float = 1.0

    # Control thresholds
    utcf_high: float = 0.7
    utcf_low: float = 0.3

    # Adaptive learning rate bounds
    lr_min_mult: float = 0.1
    lr_max_mult: float = 10.0

    # Dynamic lambda_topo policy
    lambda_topo_base: float = 0.1
    lambda_topo_instability_scale: float = 0.1
    kappa_instability_scale: float = 0.5

    # Turbulence dampening policy
    epsilon_limit: float = 0.5
    lr_dampen_factor: float = 0.8
    momentum_boost: float = 0.05


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


class DissipationMonitor:
    """Tracks Temporal Weight Variance for Epsilon Proxy.

    Per T-FAN spec: ε_proxy = Var_t(||ΔW_t||)

    This monitors the variance of weight delta norms over a sliding window,
    measuring the "turbulence" in the optimization trajectory.
    """

    def __init__(self, window: int = 100):
        self.window = window
        self.delta_norms: deque = deque(maxlen=window)
        self.prev_weights: Optional[torch.Tensor] = None

    def update(self, current_weights: torch.Tensor) -> None:
        """Update with current weights, computing delta norm."""
        current_flat = current_weights.detach().flatten()

        if self.prev_weights is not None:
            # Compute ||ΔW_t|| = ||W_t - W_{t-1}||
            delta = current_flat - self.prev_weights
            delta_norm = delta.norm().item()
            self.delta_norms.append(delta_norm)

        # Store for next step
        self.prev_weights = current_flat.clone()

    def epsilon_proxy(self) -> float:
        """Compute ε_proxy = Var_t(||ΔW_t||)."""
        if len(self.delta_norms) < 2:
            return 0.0

        arr = torch.tensor(list(self.delta_norms))
        return float(arr.var().item())

    def mean_delta(self) -> float:
        """Get mean of delta norms (for diagnostics)."""
        if len(self.delta_norms) < 1:
            return 0.0
        return float(sum(self.delta_norms) / len(self.delta_norms))


class EpsilonProxy(nn.Module):
    """ε (Epsilon) - Dissipation/Turbulence Proxy.

    Per T-FAN spec: ε_proxy = Var_t(||ΔW_t||)

    Measures temporal variance of weight update magnitudes.
    High ε: turbulent dynamics, need stabilization
    Low ε: laminar flow, can accelerate

    Also supports gradient-based computation for compatibility.
    """

    def __init__(self, config: EpsilonConfig):
        super().__init__()
        self.config = config

        # Weight-based dissipation monitor (primary per T-FAN spec)
        self.dissipation_monitor = DissipationMonitor(window=config.window_size)

        # Gradient history (secondary/fallback)
        self.grad_history: deque = deque(maxlen=config.window_size)

        # EMA state
        self.register_buffer("ema_epsilon", torch.tensor(0.5))
        self.register_buffer("ema_grad_var", torch.tensor(0.0))
        self.register_buffer("ema_grad_mean", torch.tensor(1.0))

    def update_weights(self, weights: torch.Tensor) -> None:
        """Update weight-based dissipation monitor."""
        self.dissipation_monitor.update(weights)

    def update_history(self, gradients: torch.Tensor) -> None:
        """Update gradient history (fallback method)."""
        self.grad_history.append(gradients.detach().clone())

    def compute_from_weights(self) -> float:
        """Compute ε from weight delta variance (T-FAN spec)."""
        return self.dissipation_monitor.epsilon_proxy()

    def compute_from_gradients(self) -> torch.Tensor:
        """Compute ε from gradient history (fallback)."""
        if len(self.grad_history) < 2:
            return torch.tensor(0.5)

        grads = torch.stack(list(self.grad_history), dim=0)
        grad_var = grads.var(dim=0).mean()
        grad_mean = grads.abs().mean()

        return grad_var / (grad_mean + self.config.eps)

    def forward(
        self,
        weights: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        loss_values: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute ε proxy value.

        Primary: Weight delta variance (T-FAN spec)
        Fallback: Gradient variance

        Args:
            weights: Current weight tensor to track
            gradients: Gradient tensor (fallback)
            loss_values: Recent loss values (blending)

        Returns:
            Dict with epsilon value and diagnostics
        """
        # Primary: weight-based (T-FAN spec: Var_t(||ΔW_t||))
        if weights is not None:
            self.update_weights(weights)
            epsilon = torch.tensor(self.compute_from_weights())
        elif gradients is not None:
            # Fallback: gradient-based
            self.update_history(gradients.flatten())
            epsilon = self.compute_from_gradients()
        else:
            epsilon = self.ema_epsilon

        # Blend with loss trajectory if available
        if loss_values is not None and len(loss_values) >= 2:
            loss_var = loss_values.var()
            loss_mean = loss_values.abs().mean()
            epsilon_loss = loss_var / (loss_mean + self.config.eps)
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
            "history_size": len(self.dissipation_monitor.delta_norms),
            "mean_delta_norm": self.dissipation_monitor.mean_delta(),
        }


class KappaProxy(nn.Module):
    """κ (Kappa) - Fisher Information Matrix Curvature Proxy.

    Per T-FAN spec: κ_proxy = max eigvals(FIM)

    Measures the curvature of the loss landscape via FIM maximum eigenvalue.
    Uses power iteration for efficient computation.

    High κ: sharp curvature, small steps needed
    Low κ: flat region, can take larger steps
    """

    def __init__(self, config: KappaConfig):
        super().__init__()
        self.config = config

        # EMA state
        self.register_buffer("ema_kappa", torch.tensor(1.0))
        self.register_buffer("ema_trace", torch.tensor(0.0))
        self.register_buffer("ema_max_eigval", torch.tensor(1.0))

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

    def estimate_fim_max_eigenvalue(
        self,
        model: nn.Module,
        loss_fn: Callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        n_power_iter: int = 5,
    ) -> torch.Tensor:
        """Estimate max eigenvalue of FIM using power iteration.

        Per T-FAN spec: κ_proxy = max eigvals(FIM)

        Uses the relation: FIM * v ≈ E[g * (g^T v)] where g = ∇log p

        Args:
            model: The model to compute FIM for
            loss_fn: Loss function (negative log likelihood)
            inputs: Input batch
            targets: Target batch
            n_power_iter: Number of power iteration steps

        Returns:
            Estimated maximum eigenvalue
        """
        # Gather parameters
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            return torch.tensor(0.0)

        # Initialize random vector
        v = [torch.randn_like(p) for p in params]
        v_norm = math.sqrt(sum((vi ** 2).sum().item() for vi in v))
        v = [vi / (v_norm + 1e-8) for vi in v]

        for _ in range(n_power_iter):
            # Compute gradient
            model.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward(create_graph=True)

            # Compute FIM * v via gradient-vector product
            # For empirical Fisher: F * v = E[g * (g^T v)]
            gv = sum((p.grad * vi).sum() for p, vi in zip(params, v) if p.grad is not None)

            # Compute gradient of (g^T v) to get F * v
            model.zero_grad()
            if gv.requires_grad:
                gv.backward()

                # New v = F * v (normalized)
                fv = [p.grad.clone() if p.grad is not None else torch.zeros_like(p)
                      for p in params]
            else:
                # Fallback: use g * (g^T v) approximation
                fv = [p.grad * gv if p.grad is not None else torch.zeros_like(p)
                      for p in params]

            # Normalize
            fv_norm = math.sqrt(sum((fi ** 2).sum().item() for fi in fv) + 1e-8)
            v = [fi / fv_norm for fi in fv]

        # Rayleigh quotient: λ_max ≈ v^T F v / v^T v
        model.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # v^T g
        vtg = sum((p.grad * vi).sum() for p, vi in zip(params, v) if p.grad is not None)
        # λ_max ≈ (v^T g)^2 for empirical Fisher
        lambda_max = vtg ** 2

        model.zero_grad()

        return lambda_max.detach()

    def compute_kappa_from_fim(
        self,
        fim_diag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute κ from FIM diagonal (trace-based fallback)."""
        fim_diag = fim_diag + self.config.damping
        trace = fim_diag.sum()
        dim = fim_diag.numel()
        return trace / (dim + self.config.eps)

    def forward(
        self,
        fim_diag: Optional[torch.Tensor] = None,
        fim_max_eigval: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
        hessian_diag: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute κ proxy value.

        Primary: FIM max eigenvalue (T-FAN spec)
        Fallback: FIM diagonal trace or gradient norm

        Args:
            fim_diag: Diagonal of FIM
            fim_max_eigval: Pre-computed max eigenvalue
            grad_norm: Gradient norm (rough proxy)
            hessian_diag: Diagonal of Hessian (alternative)

        Returns:
            Dict with kappa value and diagnostics
        """
        # Primary: max eigenvalue (T-FAN spec)
        if fim_max_eigval is not None:
            kappa = fim_max_eigval
        elif fim_diag is not None:
            # Use max of diagonal as proxy for max eigenvalue
            kappa = fim_diag.max()
        elif hessian_diag is not None:
            kappa = hessian_diag.abs().max()
        elif grad_norm is not None:
            # Rough proxy: grad norm² ≈ trace(F)
            kappa = grad_norm ** 2
        else:
            kappa = self.ema_kappa

        # Update EMA
        if self.training:
            self.ema_kappa = (
                self.config.ema_decay * self.ema_kappa +
                (1 - self.config.ema_decay) * kappa
            )
            self.ema_max_eigval = (
                self.config.ema_decay * self.ema_max_eigval +
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
            "kappa_max_eigval": self.ema_max_eigval,
            "regime": regime,
        }


class UDKController(nn.Module):
    """Unified Dynamics and Knowledge Controller.

    Per T-FAN spec, composes σ, ε, L_topo, κ proxies into UTCF:

    UTCF_metrics = α_σ·σ_proxy + α_ε·ε_proxy + α_topo·L_topo + α_κ·κ_proxy

    Governance policies:
    1. Dynamic λ_topo scaling based on structural instability
    2. Optimizer dampening when ε (turbulence) is high
    3. Adaptive learning rate and temperature control

    Control outputs:
    - learning_rate_mult: Multiplier for base LR
    - lambda_topo: Dynamic topological loss coefficient
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
        self.sigma_w = nn.Parameter(torch.tensor(config.alpha_sigma))
        self.epsilon_w = nn.Parameter(torch.tensor(config.alpha_epsilon))
        self.topo_w = nn.Parameter(torch.tensor(config.alpha_topo))
        self.kappa_w = nn.Parameter(torch.tensor(config.alpha_kappa))

        # Control history
        self.utcf_history: deque = deque(maxlen=100)

        # State tracking for T-FAN metrics
        self.state: Dict[str, float] = {
            "sigma_proxy": 0.0,
            "epsilon_proxy": 0.0,
            "L_topo": 0.0,
            "kappa_proxy": 0.0,
        }

    def compute_utcf(
        self,
        sigma: torch.Tensor,
        epsilon: torch.Tensor,
        kappa: torch.Tensor,
        l_topo: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute Unified Topo-Thermodynamic Cost Field.

        Per T-FAN spec:
        UTCF = α_σ·σ + α_ε·ε + α_topo·L_topo + α_κ·κ
        """
        # Normalize each to [0, 1] range (roughly)
        sigma_norm = torch.sigmoid(sigma)
        epsilon_norm = torch.sigmoid(epsilon)
        kappa_norm = torch.sigmoid(torch.log1p(kappa))

        if l_topo is not None:
            topo_norm = torch.sigmoid(l_topo)
        else:
            topo_norm = torch.tensor(0.0)

        # Get weights
        if self.use_learned_weights:
            w_sigma = torch.softplus(self.sigma_w)
            w_epsilon = torch.softplus(self.epsilon_w)
            w_topo = torch.softplus(self.topo_w)
            w_kappa = torch.softplus(self.kappa_w)
        else:
            w_sigma = self.config.alpha_sigma
            w_epsilon = self.config.alpha_epsilon
            w_topo = self.config.alpha_topo
            w_kappa = self.config.alpha_kappa

        total_w = w_sigma + w_epsilon + w_topo + w_kappa + 1e-8

        utcf = (
            w_sigma * sigma_norm +
            w_epsilon * epsilon_norm +
            w_topo * topo_norm +
            w_kappa * kappa_norm
        ) / total_w

        return utcf

    def get_lambda_topo(self) -> float:
        """Dynamic λ_topo scaling based on structural instability.

        Per T-FAN spec:
        λ_topo = base * (1 + L_topo * scale + κ * kappa_scale)

        High instability requires stronger topological inductive bias.
        """
        l_topo = self.state["L_topo"]
        kappa = self.state["kappa_proxy"]

        instability_factor = (
            1.0 +
            l_topo * self.config.lambda_topo_instability_scale +
            kappa * self.config.kappa_instability_scale
        )

        return max(0.01, self.config.lambda_topo_base * instability_factor)

    def adjust_optimizer_config(self, optimizer_config: Dict[str, Any]) -> Dict[str, Any]:
        """Dampening policy when ε (turbulence) is high.

        Per T-FAN spec:
        - Reduce LR when ε > threshold
        - Increase momentum for stability
        """
        epsilon = self.state["epsilon_proxy"]

        if epsilon > self.config.epsilon_limit:
            # Dampening: reduce LR, boost momentum
            optimizer_config["lr"] = optimizer_config.get("lr", 1e-3) * self.config.lr_dampen_factor
            current_momentum = optimizer_config.get("momentum", 0.9)
            optimizer_config["momentum"] = min(0.95, current_momentum + self.config.momentum_boost)

        return optimizer_config

    def update_tff_metrics(
        self,
        model: Optional[nn.Module] = None,
        l_topo: float = 0.0,
        kappa_proxy: Optional[float] = None,
        target_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Update T-FAN metrics from TFF training loop.

        Args:
            model: Model to track weights from
            l_topo: Topological regularization loss
            kappa_proxy: Pre-computed κ value
            target_weights: Specific weights to track for ε
        """
        # Update epsilon from weights
        if target_weights is not None:
            self.epsilon_proxy.update_weights(target_weights)
            self.state["epsilon_proxy"] = self.epsilon_proxy.dissipation_monitor.epsilon_proxy()
        elif model is not None:
            # Default: track first Linear layer weights
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    self.epsilon_proxy.update_weights(module.weight)
                    self.state["epsilon_proxy"] = self.epsilon_proxy.dissipation_monitor.epsilon_proxy()
                    break

        # Update topological loss
        self.state["L_topo"] = l_topo

        # Update curvature
        if kappa_proxy is not None:
            self.state["kappa_proxy"] = kappa_proxy

    def update_dac_metrics(self, cost_R: float, precision_gain: float) -> None:
        """Update metrics from DAC belief revision.

        Args:
            cost_R: FLOPs/time cost of revision
            precision_gain: Reduction in DAC mismatch loss
        """
        sigma = self.sigma_proxy.config.eps  # Prevent div by zero
        sigma = math.log1p(cost_R) / (sigma + precision_gain)
        self.state["sigma_proxy"] = sigma

    def utcf_cost_metrics(self) -> float:
        """Compute core UTCF metric value from current state."""
        s = self.state
        return (
            self.config.alpha_sigma * s["sigma_proxy"] +
            self.config.alpha_epsilon * s["epsilon_proxy"] +
            self.config.alpha_topo * s["L_topo"] +
            self.config.alpha_kappa * s["kappa_proxy"]
        )

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
        # Epsilon inputs (T-FAN spec: weight-based)
        weights: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
        loss_values: Optional[torch.Tensor] = None,
        # Kappa inputs
        fim_diag: Optional[torch.Tensor] = None,
        fim_max_eigval: Optional[torch.Tensor] = None,
        grad_norm: Optional[torch.Tensor] = None,
        # Topological inputs
        l_topo: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Run full UDK control cycle.

        Per T-FAN spec, computes UTCF from all four proxies:
        UTCF = α_σ·σ + α_ε·ε + α_topo·L_topo + α_κ·κ

        Args:
            cost_R: Belief revision cost (σ input)
            precision_gain: Uncertainty reduction (σ input)
            weights: Weight tensor to track for ε (primary)
            gradients: Gradient tensor (ε fallback)
            loss_values: Recent losses (ε blending)
            fim_diag: FIM diagonal (κ fallback)
            fim_max_eigval: Pre-computed max eigenvalue (κ primary)
            grad_norm: Gradient norm (κ rough proxy)
            l_topo: Topological regularization loss

        Returns:
            Dict with control outputs and diagnostics
        """
        # Compute individual proxies
        sigma_out = self.sigma_proxy(
            cost_R if cost_R is not None else torch.tensor(0.0),
            precision_gain if precision_gain is not None else torch.tensor(1.0),
        )

        epsilon_out = self.epsilon_proxy(
            weights=weights,
            gradients=gradients,
            loss_values=loss_values,
        )

        kappa_out = self.kappa_proxy(
            fim_diag=fim_diag,
            fim_max_eigval=fim_max_eigval,
            grad_norm=grad_norm,
        )

        # Update state for policy methods
        self.state["sigma_proxy"] = sigma_out["sigma_ema"].item() if isinstance(sigma_out["sigma_ema"], torch.Tensor) else sigma_out["sigma_ema"]
        self.state["epsilon_proxy"] = epsilon_out["epsilon_ema"].item() if isinstance(epsilon_out["epsilon_ema"], torch.Tensor) else epsilon_out["epsilon_ema"]
        self.state["kappa_proxy"] = kappa_out["kappa_ema"].item() if isinstance(kappa_out["kappa_ema"], torch.Tensor) else kappa_out["kappa_ema"]
        if l_topo is not None:
            self.state["L_topo"] = l_topo.item() if isinstance(l_topo, torch.Tensor) else l_topo

        # Compute UTCF (with L_topo per T-FAN spec)
        utcf = self.compute_utcf(
            sigma_out["sigma_ema"],
            epsilon_out["epsilon_ema"],
            kappa_out["kappa_ema"],
            l_topo=l_topo,
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

        # Dynamic lambda_topo (T-FAN policy)
        lambda_topo = self.get_lambda_topo()

        return {
            # Control outputs
            "utcf": utcf,
            "utcf_cost_metrics": self.utcf_cost_metrics(),
            "learning_rate_mult": lr_mult,
            "lambda_topo": lambda_topo,
            "temperature": temperature,
            "memory_write_prob": memory_write_prob,
            "control_signals": signals,
            # Proxy values
            "sigma": sigma_out,
            "epsilon": epsilon_out,
            "kappa": kappa_out,
            "l_topo": l_topo.item() if l_topo is not None and isinstance(l_topo, torch.Tensor) else (l_topo if l_topo is not None else 0.0),
            # State snapshot
            "state": dict(self.state),
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
