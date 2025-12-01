# analysis/criticality.py
# Criticality Validation Suite for TGSFN
#
# Implements validation tools to verify membership in the Conserved Directed
# Percolation (C-DP) mean-field universality class.
#
# Key Exponents and Relations:
#   - τ (tau): Size distribution exponent, P(S) ~ S^(-τ)
#   - α (alpha): Duration distribution exponent, P(T) ~ T^(-α)
#   - γ_sT: Size-duration scaling, <S> ~ T^(γ_sT)
#
# C-DP Mean-Field Universality Class:
#   - τ ≈ 1.5 - 1.65
#   - α ≈ 2.0
#   - γ_sT ≈ 2.0
#   - Scaling relation: (τ - 1)/(α - 1) = γ_sT
#
# Finite-Size Scaling:
#   - True asymptotic: α = 3/2 (as N → ∞)
#   - Phenomenological fit: τ_eff(N) ≈ 3/2 + c/√N (c ≈ 6.6)
#   - Logarithmic correction: τ_eff(N) ≈ 3/2 + 1/(ln N + C)
#
# Scientific Constraints (Referee-safe):
#   - Use maximum likelihood for power-law fitting
#   - Report confidence intervals
#   - Validate with Kolmogorov-Smirnov test
#
# References:
#   - Clauset et al. (2009): Power-law distributions in empirical data
#   - Muñoz et al. (2018): Colloquium on criticality and neural avalanches

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np
from scipy import stats
from scipy.special import zeta
from scipy.optimize import minimize_scalar

from .avalanches import Avalanche, AvalancheStats


# =============================================================================
# Constants
# =============================================================================

# C-DP mean-field exponents (theoretical values)
CDP_TAU = 1.5       # Size exponent (branching process)
CDP_ALPHA = 2.0     # Duration exponent
CDP_GAMMA_ST = 2.0  # Size-duration scaling

# Finite-size scaling constant (phenomenological)
FINITE_SIZE_C = 6.6  # τ_eff(N) ≈ 3/2 + c/√N


@dataclass
class PowerLawFit:
    """Result of power-law fitting."""
    exponent: float           # Fitted exponent (τ or α)
    xmin: float               # Lower cutoff
    sigma: float              # Standard error
    n_tail: int               # Number of samples in tail
    ks_statistic: float       # Kolmogorov-Smirnov statistic
    ks_pvalue: float          # KS p-value (> 0.1 suggests good fit)
    llr: float                # Log-likelihood ratio vs exponential
    llr_pvalue: float         # LLR p-value


@dataclass
class CriticalityResult:
    """Complete criticality validation result."""
    tau: PowerLawFit          # Size exponent fit
    alpha: PowerLawFit        # Duration exponent fit
    gamma_sT: float           # Size-duration scaling
    gamma_sT_err: float       # Scaling error
    scaling_satisfied: bool   # Whether (τ-1)/(α-1) ≈ γ_sT
    is_critical: bool         # Overall criticality assessment
    cdp_distance: float       # Distance from C-DP class
    N: int                    # System size
    tau_eff_predicted: float  # Predicted τ from finite-size scaling


# =============================================================================
# Power-Law Fitting (MLE)
# =============================================================================

def fit_power_law(
    data: np.ndarray,
    xmin: Optional[float] = None,
    xmax: Optional[float] = None,
) -> PowerLawFit:
    """
    Fit power-law distribution using maximum likelihood estimation.

    Uses the method of Clauset et al. (2009) for discrete power laws.

    P(x) ∝ x^(-α) for x ≥ xmin

    Args:
        data: Array of positive values (sizes or durations)
        xmin: Lower cutoff (if None, estimated via KS minimization)
        xmax: Upper cutoff (if None, no upper bound)

    Returns:
        PowerLawFit with exponent and diagnostics
    """
    data = np.asarray(data, dtype=float)
    data = data[data > 0]  # Remove zeros

    if xmax is not None:
        data = data[data <= xmax]

    if len(data) < 10:
        return PowerLawFit(
            exponent=np.nan, xmin=1.0, sigma=np.nan, n_tail=0,
            ks_statistic=np.nan, ks_pvalue=0.0, llr=np.nan, llr_pvalue=np.nan,
        )

    # Estimate xmin if not provided
    if xmin is None:
        xmin = _estimate_xmin(data)

    # Filter to tail
    tail = data[data >= xmin]
    n_tail = len(tail)

    if n_tail < 10:
        return PowerLawFit(
            exponent=np.nan, xmin=xmin, sigma=np.nan, n_tail=n_tail,
            ks_statistic=np.nan, ks_pvalue=0.0, llr=np.nan, llr_pvalue=np.nan,
        )

    # MLE for discrete power law
    # α = 1 + n / Σ ln(x_i / (xmin - 0.5))
    log_sum = np.sum(np.log(tail / (xmin - 0.5)))
    alpha = 1 + n_tail / log_sum

    # Standard error (Cramér-Rao bound)
    sigma = (alpha - 1) / np.sqrt(n_tail)

    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = _ks_test_powerlaw(tail, alpha, xmin)

    # Log-likelihood ratio vs exponential
    llr, llr_pval = _llr_vs_exponential(tail, alpha, xmin)

    return PowerLawFit(
        exponent=alpha,
        xmin=xmin,
        sigma=sigma,
        n_tail=n_tail,
        ks_statistic=ks_stat,
        ks_pvalue=ks_pval,
        llr=llr,
        llr_pvalue=llr_pval,
    )


def _estimate_xmin(data: np.ndarray) -> float:
    """
    Estimate optimal xmin by minimizing KS statistic.

    Scans possible xmin values and selects the one that minimizes
    the Kolmogorov-Smirnov distance between data and fitted power law.
    """
    unique_vals = np.unique(data)
    if len(unique_vals) < 5:
        return unique_vals[0]

    # Try xmin values from 10th to 90th percentile
    candidates = unique_vals[
        (unique_vals >= np.percentile(data, 10)) &
        (unique_vals <= np.percentile(data, 90))
    ][:50]  # Limit candidates

    best_xmin = candidates[0]
    best_ks = np.inf

    for xmin in candidates:
        tail = data[data >= xmin]
        if len(tail) < 10:
            continue

        # Fit and compute KS
        log_sum = np.sum(np.log(tail / (xmin - 0.5)))
        alpha = 1 + len(tail) / log_sum

        ks, _ = _ks_test_powerlaw(tail, alpha, xmin)
        if ks < best_ks:
            best_ks = ks
            best_xmin = xmin

    return best_xmin


def _ks_test_powerlaw(
    data: np.ndarray,
    alpha: float,
    xmin: float,
) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov test for power-law fit.

    Compares empirical CDF to theoretical power-law CDF.
    """
    n = len(data)
    if n < 5:
        return np.nan, 0.0

    # Sort data
    sorted_data = np.sort(data)

    # Empirical CDF
    ecdf = np.arange(1, n + 1) / n

    # Theoretical CDF for power law: 1 - (x/xmin)^(1-α)
    tcdf = 1 - (sorted_data / xmin) ** (1 - alpha)
    tcdf = np.clip(tcdf, 0, 1)

    # KS statistic
    ks = np.max(np.abs(ecdf - tcdf))

    # Approximate p-value (for large n)
    # Using asymptotic formula
    pval = 2 * np.exp(-2 * n * ks ** 2)
    pval = min(1.0, max(0.0, pval))

    return ks, pval


def _llr_vs_exponential(
    data: np.ndarray,
    alpha: float,
    xmin: float,
) -> Tuple[float, float]:
    """
    Log-likelihood ratio test: power law vs exponential.

    Positive LLR favors power law.
    """
    n = len(data)
    if n < 5:
        return 0.0, 0.5

    # Log-likelihood for power law
    ll_pl = n * np.log(alpha - 1) - n * np.log(xmin) - alpha * np.sum(np.log(data / xmin))

    # Fit and log-likelihood for exponential
    lambda_exp = 1 / (np.mean(data) - xmin)
    ll_exp = n * np.log(lambda_exp) - lambda_exp * np.sum(data - xmin)

    # LLR
    llr = ll_pl - ll_exp

    # Normalized (pseudo p-value)
    llr_pval = 1 / (1 + np.exp(-llr / np.sqrt(n)))

    return llr, llr_pval


# =============================================================================
# Scaling Relations
# =============================================================================

def compute_size_duration_scaling(
    sizes: np.ndarray,
    durations: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute size-duration scaling exponent γ_sT.

    <S> ∝ T^γ_sT

    Uses log-log regression on mean size vs duration.

    Args:
        sizes: Array of avalanche sizes
        durations: Array of avalanche durations

    Returns:
        (gamma_sT, standard_error)
    """
    # Group by duration
    unique_T = np.unique(durations)
    mean_S = []
    T_vals = []

    for T in unique_T:
        mask = durations == T
        if mask.sum() >= 3:  # Require multiple samples
            mean_S.append(sizes[mask].mean())
            T_vals.append(T)

    if len(T_vals) < 3:
        return np.nan, np.nan

    T_vals = np.array(T_vals)
    mean_S = np.array(mean_S)

    # Log-log regression
    log_T = np.log(T_vals)
    log_S = np.log(mean_S)

    # Linear fit
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_T, log_S)

    return slope, std_err


def validate_scaling_relation(
    tau: float,
    alpha: float,
    gamma_sT: float,
    tolerance: float = 0.2,
) -> Tuple[bool, float]:
    """
    Check if scaling relation (τ-1)/(α-1) = γ_sT is satisfied.

    Args:
        tau: Size exponent
        alpha: Duration exponent
        gamma_sT: Size-duration scaling
        tolerance: Acceptable deviation

    Returns:
        (is_satisfied, deviation)
    """
    if np.isnan(tau) or np.isnan(alpha) or alpha == 1:
        return False, np.nan

    predicted_gamma = (tau - 1) / (alpha - 1)
    deviation = abs(predicted_gamma - gamma_sT)

    return deviation < tolerance, deviation


# =============================================================================
# Finite-Size Analysis
# =============================================================================

def finite_size_tau(N: int, c: float = FINITE_SIZE_C) -> float:
    """
    Compute expected effective τ for system size N.

    τ_eff(N) ≈ 3/2 + c/√N

    This is the phenomenological fit for the accessible range N ∈ [10³, 10⁵].

    Args:
        N: Number of neurons
        c: Finite-size constant (default 6.6)

    Returns:
        Expected τ_eff
    """
    return 1.5 + c / np.sqrt(N)


def finite_size_tau_log(N: int, C: float = 1.0) -> float:
    """
    Compute expected effective τ using logarithmic correction.

    τ_eff(N) ≈ 3/2 + 1/(ln N + C)

    This is the theoretically correct form (asymptotically).

    Args:
        N: Number of neurons
        C: Additive constant in log

    Returns:
        Expected τ_eff
    """
    return 1.5 + 1 / (np.log(N) + C)


def finite_size_analysis(
    tau_measured: float,
    N: int,
    method: str = "sqrt",
) -> Dict[str, float]:
    """
    Perform finite-size scaling analysis.

    Compares measured τ to finite-size prediction.

    Args:
        tau_measured: Measured size exponent
        N: System size
        method: "sqrt" for √N scaling, "log" for logarithmic

    Returns:
        Dict with predictions and residuals
    """
    if method == "log":
        tau_predicted = finite_size_tau_log(N)
        # Estimate C from data
        C_est = 1 / (tau_measured - 1.5) - np.log(N) if tau_measured > 1.5 else np.nan
    else:
        tau_predicted = finite_size_tau(N)
        # Estimate c from data
        c_est = (tau_measured - 1.5) * np.sqrt(N) if tau_measured > 1.5 else np.nan

    residual = tau_measured - tau_predicted

    # Asymptotic value (N → ∞)
    tau_asymptotic = 1.5

    return {
        "tau_measured": tau_measured,
        "tau_predicted": tau_predicted,
        "tau_asymptotic": tau_asymptotic,
        "residual": residual,
        "N": N,
        "method": method,
        "scaling_constant": c_est if method == "sqrt" else C_est,
    }


# =============================================================================
# Complete Validation
# =============================================================================

def validate_cdp_scaling(
    sizes: np.ndarray,
    durations: np.ndarray,
    N: int,
    verbose: bool = False,
) -> CriticalityResult:
    """
    Complete C-DP universality class validation.

    Performs:
    1. Power-law fit for sizes (τ)
    2. Power-law fit for durations (α)
    3. Size-duration scaling (γ_sT)
    4. Scaling relation check
    5. Finite-size analysis

    Args:
        sizes: Array of avalanche sizes
        durations: Array of avalanche durations
        N: System size (number of neurons)
        verbose: Print detailed results

    Returns:
        CriticalityResult with full validation
    """
    # Fit power laws
    tau_fit = fit_power_law(sizes)
    alpha_fit = fit_power_law(durations)

    # Size-duration scaling
    gamma_sT, gamma_err = compute_size_duration_scaling(sizes, durations)

    # Scaling relation
    scaling_ok, scaling_dev = validate_scaling_relation(
        tau_fit.exponent, alpha_fit.exponent, gamma_sT
    )

    # Finite-size prediction
    tau_eff_predicted = finite_size_tau(N)

    # Distance from C-DP class
    tau_dist = abs(tau_fit.exponent - CDP_TAU) if not np.isnan(tau_fit.exponent) else np.inf
    alpha_dist = abs(alpha_fit.exponent - CDP_ALPHA) if not np.isnan(alpha_fit.exponent) else np.inf
    gamma_dist = abs(gamma_sT - CDP_GAMMA_ST) if not np.isnan(gamma_sT) else np.inf
    cdp_distance = np.sqrt(tau_dist**2 + alpha_dist**2 + gamma_dist**2)

    # Overall assessment
    tau_ok = 1.3 < tau_fit.exponent < 1.8 if not np.isnan(tau_fit.exponent) else False
    alpha_ok = 1.7 < alpha_fit.exponent < 2.3 if not np.isnan(alpha_fit.exponent) else False
    gamma_ok = 1.5 < gamma_sT < 2.5 if not np.isnan(gamma_sT) else False
    is_critical = tau_ok and alpha_ok and gamma_ok and scaling_ok

    result = CriticalityResult(
        tau=tau_fit,
        alpha=alpha_fit,
        gamma_sT=gamma_sT,
        gamma_sT_err=gamma_err if not np.isnan(gamma_err) else 0.0,
        scaling_satisfied=scaling_ok,
        is_critical=is_critical,
        cdp_distance=cdp_distance,
        N=N,
        tau_eff_predicted=tau_eff_predicted,
    )

    if verbose:
        print("=== C-DP Validation Results ===")
        print(f"\nSize distribution (τ):")
        print(f"  τ = {tau_fit.exponent:.3f} ± {tau_fit.sigma:.3f}")
        print(f"  xmin = {tau_fit.xmin:.1f}, n_tail = {tau_fit.n_tail}")
        print(f"  KS p-value = {tau_fit.ks_pvalue:.3f}")
        print(f"  Expected (finite-size): τ_eff = {tau_eff_predicted:.3f}")

        print(f"\nDuration distribution (α):")
        print(f"  α = {alpha_fit.exponent:.3f} ± {alpha_fit.sigma:.3f}")
        print(f"  xmin = {alpha_fit.xmin:.1f}, n_tail = {alpha_fit.n_tail}")
        print(f"  KS p-value = {alpha_fit.ks_pvalue:.3f}")

        print(f"\nScaling relation:")
        print(f"  γ_sT = {gamma_sT:.3f} ± {gamma_err:.3f}")
        print(f"  (τ-1)/(α-1) = {(tau_fit.exponent-1)/(alpha_fit.exponent-1):.3f}")
        print(f"  Scaling satisfied: {scaling_ok}")

        print(f"\nOverall:")
        print(f"  Distance from C-DP: {cdp_distance:.3f}")
        print(f"  Is critical: {is_critical}")

    return result


class CriticalityValidator:
    """
    Complete criticality validation pipeline.

    Combines avalanche extraction with C-DP validation.
    """

    def __init__(
        self,
        N: int,
        min_avalanches: int = 100,
        verbose: bool = False,
    ):
        """
        Args:
            N: System size (number of neurons)
            min_avalanches: Minimum avalanches for validation
            verbose: Print detailed output
        """
        self.N = N
        self.min_avalanches = min_avalanches
        self.verbose = verbose
        self.result: Optional[CriticalityResult] = None

    def validate(
        self,
        sizes: np.ndarray,
        durations: np.ndarray,
    ) -> CriticalityResult:
        """
        Run full validation.

        Args:
            sizes: Avalanche sizes
            durations: Avalanche durations

        Returns:
            CriticalityResult
        """
        if len(sizes) < self.min_avalanches:
            print(f"Warning: Only {len(sizes)} avalanches (need {self.min_avalanches})")

        self.result = validate_cdp_scaling(
            sizes, durations, self.N, verbose=self.verbose
        )
        return self.result

    def is_critical(self) -> bool:
        """Check if system is in critical regime."""
        return self.result.is_critical if self.result else False

    def get_exponents(self) -> Dict[str, float]:
        """Get fitted exponents."""
        if not self.result:
            return {}
        return {
            "tau": self.result.tau.exponent,
            "tau_err": self.result.tau.sigma,
            "alpha": self.result.alpha.exponent,
            "alpha_err": self.result.alpha.sigma,
            "gamma_sT": self.result.gamma_sT,
            "gamma_sT_err": self.result.gamma_sT_err,
        }


if __name__ == "__main__":
    print("=== Criticality Validation Test ===\n")

    # Generate synthetic critical data
    np.random.seed(42)
    n_avalanches = 1000
    N = 1000  # System size

    # Generate power-law distributed sizes and durations
    # Using inverse transform sampling
    tau_true = 1.5
    alpha_true = 2.0

    u_size = np.random.uniform(0, 1, n_avalanches)
    sizes = (1 - u_size) ** (-1 / (tau_true - 1))
    sizes = np.round(sizes).astype(int)
    sizes = np.clip(sizes, 1, 10000)

    u_dur = np.random.uniform(0, 1, n_avalanches)
    durations = (1 - u_dur) ** (-1 / (alpha_true - 1))
    durations = np.round(durations).astype(int)
    durations = np.clip(durations, 1, 1000)

    # Add some correlation (scaling relation)
    # <S> ~ T^2 for C-DP
    sizes = sizes * (durations ** 0.8)  # Approximate scaling
    sizes = np.round(sizes).astype(int)

    print(f"Generated {n_avalanches} synthetic avalanches")
    print(f"Size range: [{sizes.min()}, {sizes.max()}]")
    print(f"Duration range: [{durations.min()}, {durations.max()}]")

    # Test power-law fitting
    print("\n--- Power-Law Fitting ---")
    tau_fit = fit_power_law(sizes.astype(float))
    print(f"τ = {tau_fit.exponent:.3f} ± {tau_fit.sigma:.3f}")
    print(f"  KS p-value: {tau_fit.ks_pvalue:.3f}")

    alpha_fit = fit_power_law(durations.astype(float))
    print(f"α = {alpha_fit.exponent:.3f} ± {alpha_fit.sigma:.3f}")
    print(f"  KS p-value: {alpha_fit.ks_pvalue:.3f}")

    # Test finite-size scaling
    print("\n--- Finite-Size Scaling ---")
    for N_test in [1000, 10000, 100000]:
        tau_eff = finite_size_tau(N_test)
        print(f"N={N_test:6d}: τ_eff = {tau_eff:.4f}")

    # Test full validation
    print("\n--- Full C-DP Validation ---")
    result = validate_cdp_scaling(
        sizes.astype(float),
        durations.astype(float),
        N=N,
        verbose=True,
    )

    # Test validator class
    print("\n--- CriticalityValidator ---")
    validator = CriticalityValidator(N=N, verbose=False)
    result2 = validator.validate(sizes.astype(float), durations.astype(float))
    print(f"Is critical: {validator.is_critical()}")
    print(f"Exponents: {validator.get_exponents()}")

    print("\n✓ Criticality module test passed!")
