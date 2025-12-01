# tgsfn_core/avalanches.py
# Neuronal Avalanche Detection and Analysis
#
# Implements online avalanche detection from population spike activity.
# Used to validate that TGSFN operates near criticality.
#
# Key metrics:
#   - τ (tau): Size distribution exponent, P(s) ~ s^(-τ)
#   - α (alpha): Duration distribution exponent, P(T) ~ T^(-α)
#   - γ_sT: Size-duration scaling, <s>_T ~ T^(γ_sT)
#
# For C-DP (Conserved Directed Percolation) universality class:
#   τ ≈ 1.5, α ≈ 2.0, γ_sT ≈ 2.0, and (τ-1)/(α-1) ≈ γ_sT

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
from collections import deque


class Avalanche(NamedTuple):
    """Single avalanche record."""
    start_time: int
    end_time: int
    size: int
    duration: int
    peak_activity: int


def detect_avalanches(
    rate_series: np.ndarray,
    min_size: int = 1,
) -> List[Avalanche]:
    """
    Detect avalanches from population activity time series.

    An avalanche is a contiguous period where R(t) > 0,
    bounded by silence (R(t) = 0).

    Args:
        rate_series: 1D array of spike counts per time bin R(t)
        min_size: Minimum total spikes to count as avalanche

    Returns:
        List of Avalanche named tuples
    """
    avalanches = []
    in_avalanche = False
    current_start = 0
    current_size = 0
    current_peak = 0

    for t, r in enumerate(rate_series):
        r = int(r)
        if r > 0:
            if not in_avalanche:
                # Start new avalanche
                in_avalanche = True
                current_start = t
                current_size = 0
                current_peak = 0
            current_size += r
            current_peak = max(current_peak, r)
        else:
            if in_avalanche:
                # End current avalanche
                duration = t - current_start
                if current_size >= min_size:
                    avalanches.append(Avalanche(
                        start_time=current_start,
                        end_time=t,
                        size=current_size,
                        duration=duration,
                        peak_activity=current_peak,
                    ))
                in_avalanche = False

    # Handle avalanche at end of series
    if in_avalanche:
        duration = len(rate_series) - current_start
        if current_size >= min_size:
            avalanches.append(Avalanche(
                start_time=current_start,
                end_time=len(rate_series),
                size=current_size,
                duration=duration,
                peak_activity=current_peak,
            ))

    return avalanches


def log_bin_histogram(
    data: np.ndarray,
    num_bins: int = 20,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Compute logarithmically-binned histogram.

    Args:
        data: Array of values to histogram
        num_bins: Number of log-spaced bins

    Returns:
        (bin_centers, counts) or (None, None) if insufficient data
    """
    data = np.asarray(data)
    data = data[data > 0]

    if len(data) < 10:
        return None, None

    # Create log-spaced bins
    min_val = data.min()
    max_val = data.max()

    if min_val >= max_val:
        return None, None

    bins = np.logspace(np.log10(min_val), np.log10(max_val), num_bins + 1)
    counts, edges = np.histogram(data, bins=bins)

    # Bin centers (geometric mean)
    centers = np.sqrt(edges[:-1] * edges[1:])

    # Filter out empty bins
    mask = counts > 0
    return centers[mask], counts[mask]


def estimate_power_law_exponent(
    data: np.ndarray,
    x_min: Optional[float] = None,
    x_max: Optional[float] = None,
    num_bins: int = 20,
) -> Tuple[float, float]:
    """
    Estimate power-law exponent via log-log linear regression.

    For P(x) ~ x^(-τ), the log-binned histogram gives:
        log(P) ≈ -τ log(x) + const

    Args:
        data: Array of values
        x_min: Minimum value for fitting range
        x_max: Maximum value for fitting range
        num_bins: Number of bins for histogram

    Returns:
        (exponent, standard_error)
    """
    data = np.asarray(data, dtype=float)

    # Apply range filter
    if x_min is not None:
        data = data[data >= x_min]
    if x_max is not None:
        data = data[data <= x_max]

    if len(data) < 20:
        return np.nan, np.nan

    # Log-binned histogram
    centers, counts = log_bin_histogram(data, num_bins)

    if centers is None or len(centers) < 3:
        return np.nan, np.nan

    # Log transform
    log_x = np.log10(centers)
    log_y = np.log10(counts)

    # Linear regression
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    result = np.linalg.lstsq(A, log_y, rcond=None)
    slope = result[0][0]

    # Estimate standard error from residuals
    if len(result) > 1 and len(result[1]) > 0:
        residuals = result[1][0]
        n = len(log_x)
        stderr = np.sqrt(residuals / (n - 2)) / np.sqrt(((log_x - log_x.mean()) ** 2).sum())
    else:
        stderr = np.nan

    # Exponent is negative of slope
    exponent = -slope

    return exponent, stderr


def compute_size_duration_scaling(
    sizes: np.ndarray,
    durations: np.ndarray,
    min_samples: int = 5,
) -> Tuple[float, float]:
    """
    Compute size-duration scaling exponent γ_sT.

    For <s>_T ~ T^(γ_sT), we compute mean size for each duration
    and fit a power law.

    Args:
        sizes: Array of avalanche sizes
        durations: Array of avalanche durations
        min_samples: Minimum samples per duration bin

    Returns:
        (gamma_sT, standard_error)
    """
    sizes = np.asarray(sizes, dtype=float)
    durations = np.asarray(durations, dtype=float)

    if len(sizes) < 20:
        return np.nan, np.nan

    # Compute mean size for each unique duration
    unique_T = np.unique(durations)
    T_vals = []
    mean_sizes = []

    for T in unique_T:
        mask = durations == T
        if mask.sum() >= min_samples:
            T_vals.append(T)
            mean_sizes.append(sizes[mask].mean())

    if len(T_vals) < 3:
        return np.nan, np.nan

    T_vals = np.array(T_vals)
    mean_sizes = np.array(mean_sizes)

    # Log-log regression
    log_T = np.log10(T_vals)
    log_S = np.log10(mean_sizes)

    A = np.vstack([log_T, np.ones_like(log_T)]).T
    result = np.linalg.lstsq(A, log_S, rcond=None)
    gamma_sT = result[0][0]

    # Standard error
    if len(result) > 1 and len(result[1]) > 0:
        residuals = result[1][0]
        n = len(log_T)
        stderr = np.sqrt(residuals / (n - 2)) / np.sqrt(((log_T - log_T.mean()) ** 2).sum())
    else:
        stderr = np.nan

    return gamma_sT, stderr


def compute_avalanche_stats(
    rate_series: np.ndarray,
    s_min: int = 5,
    s_max: int = 300,
    T_min: int = 2,
    T_max: int = 100,
) -> Dict[str, float]:
    """
    Compute complete avalanche statistics.

    Args:
        rate_series: Population rate time series R(t)
        s_min, s_max: Size range for τ estimation
        T_min, T_max: Duration range for α estimation

    Returns:
        Dict with tau, alpha, gamma_sT, n_avalanches, etc.
    """
    # Detect avalanches
    avalanches = detect_avalanches(rate_series, min_size=1)

    if len(avalanches) == 0:
        return {
            "n_avalanches": 0,
            "tau": np.nan,
            "tau_err": np.nan,
            "alpha": np.nan,
            "alpha_err": np.nan,
            "gamma_sT": np.nan,
            "gamma_sT_err": np.nan,
            "mean_size": np.nan,
            "mean_duration": np.nan,
            "max_size": 0,
            "max_duration": 0,
            "scaling_relation_error": np.nan,
        }

    sizes = np.array([a.size for a in avalanches])
    durations = np.array([a.duration for a in avalanches])

    # Estimate exponents
    tau, tau_err = estimate_power_law_exponent(sizes, x_min=s_min, x_max=s_max)
    alpha, alpha_err = estimate_power_law_exponent(durations, x_min=T_min, x_max=T_max)
    gamma_sT, gamma_sT_err = compute_size_duration_scaling(sizes, durations)

    # Check scaling relation: (τ-1)/(α-1) ≈ γ_sT
    if not np.isnan(tau) and not np.isnan(alpha) and alpha != 1:
        predicted_gamma = (tau - 1) / (alpha - 1)
        scaling_error = abs(predicted_gamma - gamma_sT) if not np.isnan(gamma_sT) else np.nan
    else:
        scaling_error = np.nan

    return {
        "n_avalanches": len(avalanches),
        "tau": tau,
        "tau_err": tau_err,
        "alpha": alpha,
        "alpha_err": alpha_err,
        "gamma_sT": gamma_sT,
        "gamma_sT_err": gamma_sT_err,
        "mean_size": float(sizes.mean()),
        "mean_duration": float(durations.mean()),
        "max_size": int(sizes.max()),
        "max_duration": int(durations.max()),
        "scaling_relation_error": scaling_error,
    }


class OnlineAvalancheDetector:
    """
    Online avalanche detector for streaming spike data.

    Maintains a buffer of recent activity and detects avalanches
    in real-time as new data arrives.
    """

    def __init__(
        self,
        buffer_size: int = 10000,
        min_size: int = 1,
    ):
        """
        Initialize online detector.

        Args:
            buffer_size: Maximum history to keep
            min_size: Minimum avalanche size to record
        """
        self.buffer_size = buffer_size
        self.min_size = min_size

        self.rate_buffer = deque(maxlen=buffer_size)
        self.avalanches: List[Avalanche] = []

        # Current avalanche state
        self._in_avalanche = False
        self._current_start = 0
        self._current_size = 0
        self._current_peak = 0
        self._time = 0

    def update(self, R: int) -> Optional[Avalanche]:
        """
        Update with new population activity.

        Args:
            R: Spike count at current timestep

        Returns:
            Completed Avalanche if one just ended, else None
        """
        self.rate_buffer.append(R)
        completed = None

        if R > 0:
            if not self._in_avalanche:
                self._in_avalanche = True
                self._current_start = self._time
                self._current_size = 0
                self._current_peak = 0
            self._current_size += R
            self._current_peak = max(self._current_peak, R)
        else:
            if self._in_avalanche:
                duration = self._time - self._current_start
                if self._current_size >= self.min_size:
                    completed = Avalanche(
                        start_time=self._current_start,
                        end_time=self._time,
                        size=self._current_size,
                        duration=duration,
                        peak_activity=self._current_peak,
                    )
                    self.avalanches.append(completed)
                self._in_avalanche = False

        self._time += 1
        return completed

    def get_stats(self) -> Dict[str, float]:
        """Get current avalanche statistics."""
        if len(self.avalanches) == 0:
            return {"n_avalanches": 0, "tau": np.nan, "alpha": np.nan}

        sizes = np.array([a.size for a in self.avalanches])
        durations = np.array([a.duration for a in self.avalanches])

        tau, _ = estimate_power_law_exponent(sizes)
        alpha, _ = estimate_power_law_exponent(durations)

        return {
            "n_avalanches": len(self.avalanches),
            "tau": tau,
            "alpha": alpha,
            "mean_size": float(sizes.mean()),
            "mean_duration": float(durations.mean()),
        }

    def reset(self) -> None:
        """Reset detector state."""
        self.rate_buffer.clear()
        self.avalanches.clear()
        self._in_avalanche = False
        self._time = 0


def validate_criticality(
    stats: Dict[str, float],
    tau_range: Tuple[float, float] = (1.3, 1.8),
    alpha_range: Tuple[float, float] = (1.7, 2.3),
    gamma_range: Tuple[float, float] = (1.5, 2.5),
) -> Dict[str, bool]:
    """
    Validate whether statistics are consistent with criticality.

    Args:
        stats: Output from compute_avalanche_stats
        tau_range: Acceptable range for τ
        alpha_range: Acceptable range for α
        gamma_range: Acceptable range for γ_sT

    Returns:
        Dict with validation results
    """
    tau = stats.get("tau", np.nan)
    alpha = stats.get("alpha", np.nan)
    gamma = stats.get("gamma_sT", np.nan)

    tau_ok = tau_range[0] < tau < tau_range[1] if not np.isnan(tau) else False
    alpha_ok = alpha_range[0] < alpha < alpha_range[1] if not np.isnan(alpha) else False
    gamma_ok = gamma_range[0] < gamma < gamma_range[1] if not np.isnan(gamma) else False

    # Scaling relation
    if not np.isnan(tau) and not np.isnan(alpha) and alpha != 1:
        predicted_gamma = (tau - 1) / (alpha - 1)
        scaling_ok = abs(predicted_gamma - gamma) < 0.3 if not np.isnan(gamma) else False
    else:
        scaling_ok = False

    return {
        "tau_valid": tau_ok,
        "alpha_valid": alpha_ok,
        "gamma_valid": gamma_ok,
        "scaling_valid": scaling_ok,
        "is_critical": tau_ok and alpha_ok and gamma_ok,
    }


if __name__ == "__main__":
    print("=== Avalanche Detection Test ===\n")

    # Generate synthetic critical-like data
    np.random.seed(42)
    T = 50000
    N = 256

    # Branching process simulation
    R = np.zeros(T, dtype=int)
    R[0] = 3
    sigma = 0.98  # Slightly subcritical

    for t in range(1, T):
        if R[t-1] > 0:
            R[t] = np.random.poisson(sigma * R[t-1])
        else:
            R[t] = np.random.poisson(0.3)
        R[t] = min(R[t], N)

    print(f"Generated {T} timesteps")
    print(f"Total spikes: {R.sum()}")
    print(f"Mean rate: {R.mean():.3f}")

    # Compute statistics
    print("\n--- Avalanche Statistics ---")
    stats = compute_avalanche_stats(R, s_min=5, s_max=500)

    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Validate criticality
    print("\n--- Criticality Validation ---")
    validation = validate_criticality(stats)
    for k, v in validation.items():
        print(f"  {k}: {v}")

    # Test online detector
    print("\n--- Online Detector Test ---")
    detector = OnlineAvalancheDetector()

    for r in R[:10000]:
        detector.update(r)

    online_stats = detector.get_stats()
    print(f"  Online n_avalanches: {online_stats['n_avalanches']}")
    print(f"  Online τ: {online_stats['tau']:.3f}")
    print(f"  Online α: {online_stats['alpha']:.3f}")

    print("\n✓ Avalanche detection test passed!")
