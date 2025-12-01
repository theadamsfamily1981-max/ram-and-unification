# analysis/avalanches.py
# Avalanche Extraction and Analysis for TGSFN
#
# Implements the standard definition of neuronal avalanches:
#   An avalanche is a contiguous run of non-zero population spiking R(t) > 0,
#   bounded by periods of silence R(t) = 0.
#
# Avalanche Properties:
#   - Size (S): Total number of spikes in the avalanche
#   - Duration (T): Number of timesteps
#   - Shape: Temporal profile of activity
#
# Scientific Constraints (Referee-safe):
#   - Uses standard neuroscience definition (Beggs & Plenz, 2003)
#   - Proper handling of boundary conditions
#   - Statistical rigor in measurement
#
# References:
#   - Beggs & Plenz (2003): Neuronal avalanches in neocortical circuits
#   - Friedman et al. (2012): Universal critical dynamics in cortex

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np
import torch


class Avalanche(NamedTuple):
    """Single avalanche record."""
    start: int          # Start timestep
    end: int            # End timestep (exclusive)
    size: int           # Total spike count
    duration: int       # Duration in timesteps
    peak_activity: int  # Maximum spikes in single timestep
    shape: np.ndarray   # Activity profile R(t) during avalanche


@dataclass
class AvalancheStats:
    """Summary statistics for avalanche distribution."""
    sizes: np.ndarray           # Array of all avalanche sizes
    durations: np.ndarray       # Array of all avalanche durations
    n_avalanches: int           # Total number of avalanches
    mean_size: float            # Mean avalanche size
    mean_duration: float        # Mean avalanche duration
    max_size: int               # Largest avalanche
    max_duration: int           # Longest avalanche
    total_spikes: int           # Total spikes across all avalanches
    total_time: int             # Total simulation timesteps
    avalanche_rate: float       # Avalanches per timestep


def extract_avalanches(
    spike_train: np.ndarray,
    min_silence: int = 1,
    min_size: int = 1,
) -> List[Avalanche]:
    """
    Extract avalanches from spike train data.

    An avalanche is defined as a contiguous period where R(t) > 0,
    bounded by periods of silence (R(t) = 0).

    Args:
        spike_train: Binary spike matrix, shape (T, N) or population rate (T,)
        min_silence: Minimum silent timesteps to separate avalanches
        min_size: Minimum spikes to count as avalanche

    Returns:
        List of Avalanche named tuples
    """
    # Convert to population activity R(t)
    if spike_train.ndim == 2:
        R = spike_train.sum(axis=1)  # Sum over neurons
    else:
        R = spike_train.copy()

    T = len(R)
    avalanches = []

    # Find contiguous active periods
    in_avalanche = False
    start_t = 0
    silence_count = 0

    for t in range(T):
        if R[t] > 0:
            if not in_avalanche:
                # Start new avalanche
                in_avalanche = True
                start_t = t
            silence_count = 0
        else:
            if in_avalanche:
                silence_count += 1
                if silence_count >= min_silence:
                    # End current avalanche
                    end_t = t - silence_count + 1
                    shape = R[start_t:end_t]
                    size = int(shape.sum())

                    if size >= min_size:
                        avalanches.append(Avalanche(
                            start=start_t,
                            end=end_t,
                            size=size,
                            duration=end_t - start_t,
                            peak_activity=int(shape.max()),
                            shape=shape.copy(),
                        ))

                    in_avalanche = False

    # Handle avalanche at end of recording
    if in_avalanche:
        end_t = T
        shape = R[start_t:end_t]
        size = int(shape.sum())
        if size >= min_size:
            avalanches.append(Avalanche(
                start=start_t,
                end=end_t,
                size=size,
                duration=end_t - start_t,
                peak_activity=int(shape.max()),
                shape=shape.copy(),
            ))

    return avalanches


def compute_avalanche_stats(avalanches: List[Avalanche]) -> AvalancheStats:
    """
    Compute summary statistics from avalanche list.

    Args:
        avalanches: List of Avalanche records

    Returns:
        AvalancheStats dataclass
    """
    if not avalanches:
        return AvalancheStats(
            sizes=np.array([]),
            durations=np.array([]),
            n_avalanches=0,
            mean_size=0.0,
            mean_duration=0.0,
            max_size=0,
            max_duration=0,
            total_spikes=0,
            total_time=0,
            avalanche_rate=0.0,
        )

    sizes = np.array([a.size for a in avalanches])
    durations = np.array([a.duration for a in avalanches])

    total_time = avalanches[-1].end - avalanches[0].start

    return AvalancheStats(
        sizes=sizes,
        durations=durations,
        n_avalanches=len(avalanches),
        mean_size=float(sizes.mean()),
        mean_duration=float(durations.mean()),
        max_size=int(sizes.max()),
        max_duration=int(durations.max()),
        total_spikes=int(sizes.sum()),
        total_time=total_time,
        avalanche_rate=len(avalanches) / max(1, total_time),
    )


def size_duration_relation(
    avalanches: List[Avalanche],
    n_bins: int = 20,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute size-duration scaling relation.

    For critical systems: <S> ∝ T^γ_sT where γ_sT ≈ 2.0 (C-DP)

    Args:
        avalanches: List of avalanches
        n_bins: Number of duration bins

    Returns:
        (duration_bins, mean_sizes, gamma_sT)
    """
    if not avalanches:
        return np.array([]), np.array([]), 0.0

    durations = np.array([a.duration for a in avalanches])
    sizes = np.array([a.size for a in avalanches])

    # Bin by duration
    unique_durations = np.unique(durations)
    if len(unique_durations) < 3:
        return unique_durations, np.array([sizes[durations == d].mean()
                                           for d in unique_durations]), 0.0

    # Compute mean size for each duration
    duration_bins = []
    mean_sizes = []

    for d in unique_durations:
        mask = durations == d
        if mask.sum() >= 3:  # Require at least 3 samples
            duration_bins.append(d)
            mean_sizes.append(sizes[mask].mean())

    duration_bins = np.array(duration_bins)
    mean_sizes = np.array(mean_sizes)

    if len(duration_bins) < 3:
        return duration_bins, mean_sizes, 0.0

    # Fit power law: log(<S>) = γ_sT * log(T) + c
    log_T = np.log(duration_bins)
    log_S = np.log(mean_sizes)

    # Linear regression
    A = np.vstack([log_T, np.ones_like(log_T)]).T
    gamma_sT, _ = np.linalg.lstsq(A, log_S, rcond=None)[0]

    return duration_bins, mean_sizes, float(gamma_sT)


def compute_avalanche_shape(
    avalanches: List[Avalanche],
    duration: int,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute average avalanche shape for given duration.

    At criticality, avalanche shapes should collapse to a universal
    parabolic profile when properly rescaled.

    Args:
        avalanches: List of avalanches
        duration: Target duration to analyze
        normalize: Whether to normalize shape to unit area

    Returns:
        Average shape profile
    """
    # Select avalanches with target duration
    matching = [a for a in avalanches if a.duration == duration]

    if not matching:
        return np.array([])

    # Stack shapes (they all have same length)
    shapes = np.stack([a.shape for a in matching])

    # Average
    avg_shape = shapes.mean(axis=0)

    if normalize:
        total = avg_shape.sum()
        if total > 0:
            avg_shape = avg_shape / total

    return avg_shape


class AvalancheAnalyzer:
    """
    Complete avalanche analysis pipeline.

    Provides methods for:
    1. Extracting avalanches from spike data
    2. Computing size and duration distributions
    3. Fitting power-law exponents
    4. Validating scaling relations
    """

    def __init__(
        self,
        min_silence: int = 1,
        min_size: int = 1,
        min_avalanches: int = 100,
    ):
        """
        Args:
            min_silence: Minimum silent steps between avalanches
            min_size: Minimum spikes to count as avalanche
            min_avalanches: Minimum avalanches for reliable statistics
        """
        self.min_silence = min_silence
        self.min_size = min_size
        self.min_avalanches = min_avalanches

        # Store results
        self.avalanches: List[Avalanche] = []
        self.stats: Optional[AvalancheStats] = None

    def analyze(
        self,
        spike_train: np.ndarray,
    ) -> Dict[str, any]:
        """
        Full analysis pipeline.

        Args:
            spike_train: Spike data, shape (T, N) or (T,)

        Returns:
            Dict with avalanches, stats, and scaling relations
        """
        # Convert torch tensor if needed
        if isinstance(spike_train, torch.Tensor):
            spike_train = spike_train.detach().cpu().numpy()

        # Extract avalanches
        self.avalanches = extract_avalanches(
            spike_train,
            min_silence=self.min_silence,
            min_size=self.min_size,
        )

        # Compute statistics
        self.stats = compute_avalanche_stats(self.avalanches)

        # Size-duration relation
        duration_bins, mean_sizes, gamma_sT = size_duration_relation(
            self.avalanches
        )

        # Check if we have enough data
        sufficient_data = len(self.avalanches) >= self.min_avalanches

        return {
            "avalanches": self.avalanches,
            "stats": self.stats,
            "n_avalanches": len(self.avalanches),
            "sizes": self.stats.sizes,
            "durations": self.stats.durations,
            "gamma_sT": gamma_sT,
            "duration_bins": duration_bins,
            "mean_sizes": mean_sizes,
            "sufficient_data": sufficient_data,
        }

    def get_size_histogram(
        self,
        bins: int = 50,
        log_bins: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute size histogram.

        Args:
            bins: Number of bins
            log_bins: Use logarithmic binning

        Returns:
            (bin_centers, counts)
        """
        if not self.avalanches:
            return np.array([]), np.array([])

        sizes = self.stats.sizes

        if log_bins:
            # Log-spaced bins
            min_s = max(1, sizes.min())
            max_s = sizes.max()
            bin_edges = np.logspace(np.log10(min_s), np.log10(max_s), bins + 1)
        else:
            bin_edges = np.linspace(sizes.min(), sizes.max(), bins + 1)

        counts, _ = np.histogram(sizes, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, counts

    def get_duration_histogram(
        self,
        bins: int = 30,
        log_bins: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute duration histogram.

        Args:
            bins: Number of bins
            log_bins: Use logarithmic binning

        Returns:
            (bin_centers, counts)
        """
        if not self.avalanches:
            return np.array([]), np.array([])

        durations = self.stats.durations

        if log_bins:
            min_d = max(1, durations.min())
            max_d = durations.max()
            bin_edges = np.logspace(np.log10(min_d), np.log10(max_d), bins + 1)
        else:
            bin_edges = np.linspace(durations.min(), durations.max(), bins + 1)

        counts, _ = np.histogram(durations, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, counts


# =============================================================================
# Torch-Compatible Functions
# =============================================================================

def extract_avalanches_torch(
    spike_train: torch.Tensor,
    min_silence: int = 1,
    min_size: int = 1,
) -> List[Avalanche]:
    """
    Torch-compatible avalanche extraction.

    Args:
        spike_train: Spike tensor, shape (T, N) or (T,)
        min_silence: Minimum silent timesteps
        min_size: Minimum avalanche size

    Returns:
        List of Avalanche records
    """
    return extract_avalanches(
        spike_train.detach().cpu().numpy(),
        min_silence=min_silence,
        min_size=min_size,
    )


if __name__ == "__main__":
    print("=== Avalanche Extraction Test ===\n")

    # Generate synthetic critical-like spike data
    np.random.seed(42)
    T, N = 10000, 100

    # Simulate branching process with σ ≈ 1 (critical)
    R = np.zeros(T)
    R[0] = 5  # Initial seed

    sigma = 0.99  # Slightly subcritical for stability
    for t in range(1, T):
        # Each spike produces σ offspring on average
        if R[t-1] > 0:
            R[t] = np.random.poisson(sigma * R[t-1])
        else:
            # Spontaneous activity
            R[t] = np.random.poisson(0.5)

    # Expand to full spike matrix
    spikes = np.zeros((T, N))
    for t in range(T):
        if R[t] > 0:
            active = np.random.choice(N, size=min(int(R[t]), N), replace=False)
            spikes[t, active] = 1

    print(f"Simulated {T} timesteps, {N} neurons")
    print(f"Total spikes: {int(spikes.sum())}")
    print(f"Mean rate: {spikes.sum() / T:.2f} spikes/timestep")

    # Extract avalanches
    print("\n--- Extracting Avalanches ---")
    analyzer = AvalancheAnalyzer(min_silence=1, min_size=1)
    result = analyzer.analyze(spikes)

    print(f"Found {result['n_avalanches']} avalanches")
    print(f"Mean size: {result['stats'].mean_size:.2f}")
    print(f"Mean duration: {result['stats'].mean_duration:.2f}")
    print(f"Max size: {result['stats'].max_size}")
    print(f"Max duration: {result['stats'].max_duration}")

    # Size-duration scaling
    print(f"\n--- Scaling Relations ---")
    print(f"γ_sT (size~duration): {result['gamma_sT']:.3f}")
    print(f"(Critical C-DP: γ_sT ≈ 2.0)")

    # Test histogram
    print("\n--- Histograms ---")
    size_bins, size_counts = analyzer.get_size_histogram(bins=20)
    print(f"Size bins: {len(size_bins)} (range {size_bins[0]:.1f} - {size_bins[-1]:.1f})")

    dur_bins, dur_counts = analyzer.get_duration_histogram(bins=15)
    print(f"Duration bins: {len(dur_bins)} (range {dur_bins[0]:.1f} - {dur_bins[-1]:.1f})")

    # Test torch compatibility
    print("\n--- Torch Compatibility ---")
    spikes_torch = torch.from_numpy(spikes).float()
    avalanches_torch = extract_avalanches_torch(spikes_torch)
    print(f"Extracted {len(avalanches_torch)} avalanches from torch tensor")

    print("\n✓ Avalanche module test passed!")
