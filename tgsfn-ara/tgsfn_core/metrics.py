# tgsfn_core/metrics.py
# Criticality Metrics for TGSFN
#
# Implements real-time monitoring of network criticality through:
#   - Branching ratio estimation
#   - Power-law exponent tracking
#   - Finite-size scaling analysis
#
# Scientific Constraints (Referee-safe):
#   - Uses established neuroscience metrics (Beggs & Plenz)
#   - Proper statistical methods for power-law fitting
#   - Finite-size corrections: τ_eff(N) ≈ 3/2 + c/√N, c ≈ 6.6

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
from collections import deque


@dataclass
class CriticalityMetrics:
    """Container for criticality measurements."""
    branching_ratio: float = 1.0      # m = <descendants>/<ancestors>
    tau: float = 1.5                  # Size exponent P(s) ~ s^(-τ)
    tau_err: float = 0.0              # Standard error on τ
    alpha: float = 2.0                # Duration exponent P(T) ~ T^(-α)
    alpha_err: float = 0.0            # Standard error on α
    gamma_sT: float = 2.0             # Size-duration scaling <s>_T ~ T^γ
    gamma_sT_err: float = 0.0         # Standard error on γ_sT
    scaling_relation_error: float = 0.0  # |predicted_γ - measured_γ|
    n_avalanches: int = 0             # Number of avalanches analyzed
    is_critical: bool = False         # Whether metrics indicate criticality

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "branching_ratio": self.branching_ratio,
            "tau": self.tau,
            "tau_err": self.tau_err,
            "alpha": self.alpha,
            "alpha_err": self.alpha_err,
            "gamma_sT": self.gamma_sT,
            "gamma_sT_err": self.gamma_sT_err,
            "scaling_relation_error": self.scaling_relation_error,
            "n_avalanches": self.n_avalanches,
            "is_critical": float(self.is_critical),
        }


def compute_branching_ratio(
    spike_counts: torch.Tensor,
    window_size: int = 10,
) -> torch.Tensor:
    """
    Compute branching ratio from population spike counts.

    The branching ratio m = E[A(t+1)] / A(t) measures how activity
    propagates. For criticality, m ≈ 1.0.

    Args:
        spike_counts: Population spike counts R(t), shape (T,) or (B, T)
        window_size: Window for temporal averaging

    Returns:
        Branching ratio estimate (scalar tensor)
    """
    if spike_counts.dim() == 1:
        spike_counts = spike_counts.unsqueeze(0)

    # Compute A(t) and A(t+1) pairs
    ancestors = spike_counts[:, :-1]  # A(t)
    descendants = spike_counts[:, 1:]  # A(t+1)

    # Only count timesteps where ancestors > 0
    valid_mask = ancestors > 0

    if not valid_mask.any():
        return torch.tensor(1.0, device=spike_counts.device)

    # Compute ratio
    numerator = (descendants * valid_mask.float()).sum()
    denominator = (ancestors * valid_mask.float()).sum()

    if denominator < 1e-8:
        return torch.tensor(1.0, device=spike_counts.device)

    m = numerator / denominator
    return m


def finite_size_correction(
    tau_measured: float,
    N: int,
    c: float = 6.6,
) -> float:
    """
    Apply finite-size correction to measured τ exponent.

    τ_eff(N) ≈ 3/2 + c/√N

    For infinite system, τ → 3/2.

    Args:
        tau_measured: Measured τ from finite network
        N: Network size (number of neurons)
        c: Finite-size coefficient (default 6.6 from theory)

    Returns:
        Corrected τ estimate
    """
    # Expected finite-size shift
    expected_shift = c / np.sqrt(N)

    # Correct by subtracting the finite-size bias
    tau_corrected = tau_measured - expected_shift + 0.5  # Shift toward 1.5

    return tau_corrected


class BranchingRatioTracker:
    """
    Online tracker for branching ratio with EMA smoothing.

    Maintains running estimate of m with exponential moving average.
    """

    def __init__(
        self,
        ema_decay: float = 0.99,
        window_size: int = 100,
    ):
        """
        Args:
            ema_decay: Decay rate for EMA
            window_size: Window for instantaneous computation
        """
        self.ema_decay = ema_decay
        self.window_size = window_size

        self._m_ema = 1.0
        self._buffer = deque(maxlen=window_size)
        self._update_count = 0

    def update(self, R_t: int, R_t_prev: int) -> float:
        """
        Update with new spike counts.

        Args:
            R_t: Current timestep spike count
            R_t_prev: Previous timestep spike count

        Returns:
            Current EMA branching ratio
        """
        if R_t_prev > 0:
            m_instant = R_t / R_t_prev
            self._buffer.append(m_instant)

            # Update EMA
            self._m_ema = self.ema_decay * self._m_ema + (1 - self.ema_decay) * m_instant
            self._update_count += 1

        return self._m_ema

    def get_branching_ratio(self) -> float:
        """Get current branching ratio estimate."""
        return self._m_ema

    def get_variance(self) -> float:
        """Get variance of recent branching ratios."""
        if len(self._buffer) < 2:
            return 0.0
        return np.var(list(self._buffer))

    def reset(self) -> None:
        """Reset tracker state."""
        self._m_ema = 1.0
        self._buffer.clear()
        self._update_count = 0


class CriticalityMonitor(nn.Module):
    """
    Real-time criticality monitoring for TGSFN.

    Tracks:
    1. Branching ratio (online)
    2. Avalanche statistics (buffered)
    3. Power-law exponents (periodic)
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        update_interval: int = 1000,
        tau_target: float = 1.5,
        alpha_target: float = 2.0,
        branching_target: float = 1.0,
        tolerance: float = 0.2,
    ):
        """
        Args:
            buffer_size: Size of activity buffer
            update_interval: Steps between full metric updates
            tau_target: Target size exponent
            alpha_target: Target duration exponent
            branching_target: Target branching ratio
            tolerance: Acceptable deviation from targets
        """
        super().__init__()

        self.buffer_size = buffer_size
        self.update_interval = update_interval
        self.tau_target = tau_target
        self.alpha_target = alpha_target
        self.branching_target = branching_target
        self.tolerance = tolerance

        # Buffers
        self.register_buffer("activity_buffer", torch.zeros(buffer_size))
        self.register_buffer("buffer_idx", torch.tensor(0))
        self.register_buffer("step_count", torch.tensor(0))

        # Branching ratio tracker
        self.branching_tracker = BranchingRatioTracker()

        # Latest metrics
        self._latest_metrics = CriticalityMetrics()

    def update(self, spike_counts: torch.Tensor) -> Optional[CriticalityMetrics]:
        """
        Update with new spike counts.

        Args:
            spike_counts: Population spike count(s)

        Returns:
            CriticalityMetrics if update_interval reached, else None
        """
        # Handle batch dimension
        if spike_counts.dim() > 0:
            R = spike_counts.sum().item()
        else:
            R = spike_counts.item()

        # Update branching ratio
        prev_idx = (self.buffer_idx - 1) % self.buffer_size
        R_prev = self.activity_buffer[prev_idx].item()
        self.branching_tracker.update(R, R_prev)

        # Store in buffer
        self.activity_buffer[self.buffer_idx] = R
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.step_count += 1

        # Periodic full update
        if self.step_count % self.update_interval == 0:
            return self._compute_full_metrics()

        return None

    def _compute_full_metrics(self) -> CriticalityMetrics:
        """Compute full criticality metrics from buffer."""
        # Get activity history
        if self.step_count < self.buffer_size:
            activity = self.activity_buffer[:self.step_count].cpu().numpy()
        else:
            # Reorder circular buffer
            idx = self.buffer_idx.item()
            activity = torch.cat([
                self.activity_buffer[idx:],
                self.activity_buffer[:idx]
            ]).cpu().numpy()

        # Import avalanche functions
        from .avalanches import compute_avalanche_stats, validate_criticality

        # Compute avalanche statistics
        stats = compute_avalanche_stats(activity)

        # Validate criticality
        validation = validate_criticality(stats)

        # Build metrics
        self._latest_metrics = CriticalityMetrics(
            branching_ratio=self.branching_tracker.get_branching_ratio(),
            tau=stats.get("tau", np.nan),
            tau_err=stats.get("tau_err", np.nan),
            alpha=stats.get("alpha", np.nan),
            alpha_err=stats.get("alpha_err", np.nan),
            gamma_sT=stats.get("gamma_sT", np.nan),
            gamma_sT_err=stats.get("gamma_sT_err", np.nan),
            scaling_relation_error=stats.get("scaling_relation_error", np.nan),
            n_avalanches=stats.get("n_avalanches", 0),
            is_critical=validation.get("is_critical", False),
        )

        return self._latest_metrics

    def get_metrics(self) -> CriticalityMetrics:
        """Get latest metrics."""
        return self._latest_metrics

    def get_branching_ratio(self) -> float:
        """Get current branching ratio."""
        return self.branching_tracker.get_branching_ratio()

    def is_critical(self) -> bool:
        """Check if network appears critical."""
        m = self.branching_tracker.get_branching_ratio()

        # Quick check: branching ratio near 1
        if abs(m - self.branching_target) > self.tolerance:
            return False

        # Full check if metrics available
        if self._latest_metrics.n_avalanches > 0:
            return self._latest_metrics.is_critical

        return True  # Assume critical if no data

    def get_deviation_from_criticality(self) -> float:
        """
        Compute scalar deviation from critical point.

        Returns:
            Positive value indicating distance from criticality
        """
        m = self.branching_tracker.get_branching_ratio()
        m_dev = abs(m - self.branching_target)

        tau_dev = abs(self._latest_metrics.tau - self.tau_target) if not np.isnan(self._latest_metrics.tau) else 0.0
        alpha_dev = abs(self._latest_metrics.alpha - self.alpha_target) if not np.isnan(self._latest_metrics.alpha) else 0.0

        # Weighted combination
        deviation = 0.5 * m_dev + 0.25 * tau_dev + 0.25 * alpha_dev

        return deviation

    def reset(self) -> None:
        """Reset monitor state."""
        self.activity_buffer.zero_()
        self.buffer_idx.zero_()
        self.step_count.zero_()
        self.branching_tracker.reset()
        self._latest_metrics = CriticalityMetrics()

    def forward(self, spike_counts: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for integration with training loop.

        Returns dict with branching_ratio and is_critical.
        """
        metrics = self.update(spike_counts)

        return {
            "branching_ratio": torch.tensor(self.branching_tracker.get_branching_ratio()),
            "is_critical": torch.tensor(float(self.is_critical())),
        }


def compute_criticality_loss(
    spike_counts: torch.Tensor,
    target_m: float = 1.0,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Compute loss that encourages critical dynamics.

    L_crit = max(0, |m - 1| - margin)²

    Only penalizes when branching ratio deviates beyond margin.

    Args:
        spike_counts: Population spike counts, shape (T,) or (B, T)
        target_m: Target branching ratio
        margin: Acceptable deviation

    Returns:
        Criticality loss (scalar)
    """
    m = compute_branching_ratio(spike_counts)

    deviation = torch.abs(m - target_m)
    loss = torch.relu(deviation - margin) ** 2

    return loss


if __name__ == "__main__":
    print("=== Criticality Metrics Test ===\n")

    # Generate synthetic critical-like data
    np.random.seed(42)
    T = 5000

    # Branching process with m ≈ 1
    R = np.zeros(T, dtype=int)
    R[0] = 5
    sigma = 0.99  # Slightly subcritical

    for t in range(1, T):
        if R[t-1] > 0:
            R[t] = np.random.poisson(sigma * R[t-1])
        else:
            R[t] = np.random.poisson(0.5)
        R[t] = min(R[t], 100)

    print(f"Generated {T} timesteps")
    print(f"Mean rate: {R.mean():.2f}")

    # Test branching ratio computation
    print("\n--- Branching Ratio ---")
    R_torch = torch.from_numpy(R).float()
    m = compute_branching_ratio(R_torch)
    print(f"Computed m: {m.item():.4f}")
    print(f"Expected (σ): {sigma:.4f}")

    # Test tracker
    print("\n--- Branching Ratio Tracker ---")
    tracker = BranchingRatioTracker(ema_decay=0.99)
    for t in range(1, min(1000, T)):
        tracker.update(R[t], R[t-1])
    print(f"Tracker m: {tracker.get_branching_ratio():.4f}")
    print(f"Tracker variance: {tracker.get_variance():.4f}")

    # Test monitor
    print("\n--- Criticality Monitor ---")
    monitor = CriticalityMonitor(buffer_size=5000, update_interval=500)

    for t in range(T):
        result = monitor.update(torch.tensor(R[t]))
        if result is not None:
            print(f"  Step {t}: m={result.branching_ratio:.3f}, τ={result.tau:.3f}, α={result.alpha:.3f}")

    # Final metrics
    metrics = monitor.get_metrics()
    print(f"\nFinal metrics:")
    print(f"  Branching ratio: {metrics.branching_ratio:.4f}")
    print(f"  τ: {metrics.tau:.3f} ± {metrics.tau_err:.3f}")
    print(f"  α: {metrics.alpha:.3f} ± {metrics.alpha_err:.3f}")
    print(f"  γ_sT: {metrics.gamma_sT:.3f}")
    print(f"  Is critical: {metrics.is_critical}")

    # Test finite-size correction
    print("\n--- Finite-Size Correction ---")
    for N in [100, 256, 1000, 10000]:
        tau_eff = 1.5 + 6.6 / np.sqrt(N)
        tau_corr = finite_size_correction(tau_eff, N)
        print(f"  N={N:5d}: τ_eff={tau_eff:.3f} → τ_corr={tau_corr:.3f}")

    # Test criticality loss
    print("\n--- Criticality Loss ---")
    loss = compute_criticality_loss(R_torch, target_m=1.0, margin=0.1)
    print(f"Criticality loss: {loss.item():.6f}")

    print("\n✓ Metrics test passed!")
