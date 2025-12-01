#!/usr/bin/env python3
"""
measure_avalanches.py
Avalanche measurement and criticality validation

Runs a TGSFN network with spontaneous activity and measures
avalanche statistics to validate criticality.

Expected results for C-DP universality class:
    - tau ≈ 1.5 (size distribution exponent)
    - alpha ≈ 2.0 (duration distribution exponent)
    - gamma_sT ≈ 2.0 (size-duration scaling)
    - (tau-1)/(alpha-1) ≈ gamma_sT (scaling relation)

Usage:
    python measure_avalanches.py --n_neurons 256 --timesteps 50000
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tgsfn_core import TGSFNNetwork, detect_avalanches, compute_avalanche_stats
from tgsfn_core.avalanches import validate_criticality, OnlineAvalancheDetector
from tgsfn_core.metrics import (
    CriticalityMonitor,
    compute_branching_ratio,
    finite_size_correction,
)


def run_spontaneous_activity(
    model: TGSFNNetwork,
    n_timesteps: int = 50000,
    input_rate: float = 0.01,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run network with spontaneous activity and record spikes.

    Args:
        model: TGSFN network
        n_timesteps: Number of timesteps
        input_rate: Spontaneous input rate (Poisson)
        device: Device to run on

    Returns:
        (population_rates, spike_matrix)
    """
    model.eval()
    model.reset_state()

    n_neurons = model.n_neurons
    input_dim = model.input_dim

    # Storage
    population_rates = np.zeros(n_timesteps)
    all_spikes = []

    print(f"Running {n_timesteps} timesteps...")

    with torch.no_grad():
        for t in range(n_timesteps):
            # Spontaneous Poisson input
            x = (torch.rand(1, input_dim, device=device) < input_rate).float()

            # Forward pass
            _, spikes = model(x)

            # Record
            R = spikes.sum().item()
            population_rates[t] = R
            all_spikes.append(spikes.cpu().numpy())

            # Progress
            if (t + 1) % 10000 == 0:
                print(f"  Timestep {t+1}/{n_timesteps}, mean R = {population_rates[:t+1].mean():.2f}")

    spike_matrix = np.concatenate(all_spikes, axis=0)

    return population_rates, spike_matrix


def analyze_avalanches(
    population_rates: np.ndarray,
    n_neurons: int,
) -> Dict:
    """
    Full avalanche analysis.

    Args:
        population_rates: R(t) time series
        n_neurons: Network size (for finite-size correction)

    Returns:
        Dict with all statistics
    """
    print("\n" + "=" * 60)
    print("AVALANCHE ANALYSIS")
    print("=" * 60)

    # Basic statistics
    print(f"\nPopulation activity statistics:")
    print(f"  Total timesteps: {len(population_rates)}")
    print(f"  Mean R(t): {population_rates.mean():.3f}")
    print(f"  Max R(t): {population_rates.max():.0f}")
    print(f"  Fraction active: {(population_rates > 0).mean():.3f}")

    # Compute avalanche statistics
    print(f"\nComputing avalanche statistics...")
    stats = compute_avalanche_stats(
        population_rates,
        s_min=5,
        s_max=500,
        T_min=2,
        T_max=100,
    )

    print(f"\nAvalanche statistics:")
    print(f"  Number of avalanches: {stats['n_avalanches']}")
    print(f"  Mean size: {stats['mean_size']:.2f}")
    print(f"  Max size: {stats['max_size']}")
    print(f"  Mean duration: {stats['mean_duration']:.2f}")
    print(f"  Max duration: {stats['max_duration']}")

    # Power-law exponents
    print(f"\nPower-law exponents:")
    print(f"  tau (size):     {stats['tau']:.3f} +/- {stats['tau_err']:.3f} (expected ~1.5)")
    print(f"  alpha (duration): {stats['alpha']:.3f} +/- {stats['alpha_err']:.3f} (expected ~2.0)")
    print(f"  gamma_sT:       {stats['gamma_sT']:.3f} +/- {stats['gamma_sT_err']:.3f} (expected ~2.0)")

    # Finite-size correction for tau
    tau_corrected = finite_size_correction(stats['tau'], n_neurons)
    print(f"  tau (corrected): {tau_corrected:.3f} (after finite-size correction)")

    # Scaling relation check
    if not np.isnan(stats['tau']) and not np.isnan(stats['alpha']) and stats['alpha'] != 1:
        predicted_gamma = (stats['tau'] - 1) / (stats['alpha'] - 1)
        print(f"\nScaling relation check:")
        print(f"  (tau-1)/(alpha-1) = {predicted_gamma:.3f}")
        print(f"  gamma_sT measured = {stats['gamma_sT']:.3f}")
        print(f"  Scaling error: {stats['scaling_relation_error']:.3f}")

    # Validate criticality
    print(f"\nCriticality validation:")
    validation = validate_criticality(stats)
    for k, v in validation.items():
        status = "PASS" if v else "FAIL"
        print(f"  {k}: {status}")

    # Overall assessment
    print("\n" + "-" * 40)
    if validation['is_critical']:
        print("RESULT: Network appears to operate near criticality")
    else:
        print("RESULT: Network may not be critical")
        if stats['tau'] < 1.3:
            print("  -> tau too low (supercritical?)")
        elif stats['tau'] > 1.8:
            print("  -> tau too high (subcritical?)")
    print("-" * 40)

    return {
        "stats": stats,
        "validation": validation,
        "tau_corrected": tau_corrected,
    }


def run_branching_analysis(
    population_rates: np.ndarray,
) -> Dict:
    """
    Analyze branching ratio dynamics.

    Args:
        population_rates: R(t) time series

    Returns:
        Branching analysis results
    """
    print("\n" + "=" * 60)
    print("BRANCHING RATIO ANALYSIS")
    print("=" * 60)

    # Compute branching ratio
    R = torch.from_numpy(population_rates).float()
    m = compute_branching_ratio(R)

    print(f"\nOverall branching ratio: {m.item():.4f}")
    print(f"  (Expected m = 1.0 for criticality)")

    # Time-varying branching ratio
    window = 1000
    n_windows = len(population_rates) // window

    m_series = []
    for i in range(n_windows):
        start = i * window
        end = start + window
        R_window = torch.from_numpy(population_rates[start:end]).float()
        m_window = compute_branching_ratio(R_window)
        m_series.append(m_window.item())

    m_series = np.array(m_series)

    print(f"\nTime-varying branching ratio (window={window}):")
    print(f"  Mean: {m_series.mean():.4f}")
    print(f"  Std:  {m_series.std():.4f}")
    print(f"  Min:  {m_series.min():.4f}")
    print(f"  Max:  {m_series.max():.4f}")

    # Check stability
    is_stable = (m_series < 1.1).all()
    is_critical = np.abs(m_series.mean() - 1.0) < 0.1

    print(f"\nBranching analysis:")
    print(f"  Stable (all m < 1.1): {'YES' if is_stable else 'NO'}")
    print(f"  Critical (|m-1| < 0.1): {'YES' if is_critical else 'NO'}")

    return {
        "m_overall": m.item(),
        "m_mean": m_series.mean(),
        "m_std": m_series.std(),
        "is_stable": is_stable,
        "is_critical": is_critical,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure avalanches in TGSFN")
    parser.add_argument("--n_neurons", type=int, default=256)
    parser.add_argument("--input_dim", type=int, default=64)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--input_rate", type=float, default=0.01)
    parser.add_argument("--ei_ratio", type=float, default=0.8)
    parser.add_argument("--connectivity", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("TGSFN Avalanche Measurement")
    print("=" * 60)
    print(f"Network size: N = {args.n_neurons}")
    print(f"Input dim: {args.input_dim}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Input rate: {args.input_rate}")
    print(f"E:I ratio: {args.ei_ratio}")
    print(f"Connectivity: {args.connectivity}")
    print("=" * 60)

    # Create network
    print("\nCreating TGSFN network...")
    model = TGSFNNetwork(
        input_dim=args.input_dim,
        n_neurons=args.n_neurons,
        output_dim=4,  # Dummy output
        ei_ratio=args.ei_ratio,
        connectivity=args.connectivity,
    ).to(args.device)

    # Run spontaneous activity
    population_rates, spike_matrix = run_spontaneous_activity(
        model=model,
        n_timesteps=args.timesteps,
        input_rate=args.input_rate,
        device=args.device,
    )

    # Analyze avalanches
    avalanche_results = analyze_avalanches(
        population_rates=population_rates,
        n_neurons=args.n_neurons,
    )

    # Analyze branching
    branching_results = run_branching_analysis(population_rates)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    stats = avalanche_results['stats']
    val = avalanche_results['validation']

    print(f"\nCriticality indicators:")
    print(f"  tau = {stats['tau']:.3f} (target: 1.5)")
    print(f"  alpha = {stats['alpha']:.3f} (target: 2.0)")
    print(f"  gamma_sT = {stats['gamma_sT']:.3f} (target: 2.0)")
    print(f"  m = {branching_results['m_overall']:.3f} (target: 1.0)")

    overall_critical = val['is_critical'] and branching_results['is_critical']
    print(f"\n{'CRITICAL' if overall_critical else 'NOT CRITICAL'}")

    print("=" * 60)


if __name__ == "__main__":
    main()
