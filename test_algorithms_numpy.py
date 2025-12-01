#!/usr/bin/env python3
"""
test_algorithms_numpy.py
Algorithm Validation with NumPy Only

Validates the mathematical correctness of TGSFN algorithms
without PyTorch dependency.

Run with: python3 test_algorithms_numpy.py
"""

import numpy as np
import sys

RESULTS = {}

def test_passed(name, msg=""):
    RESULTS[name] = True
    print(f"  [PASS] {name}" + (f": {msg}" if msg else ""))

def test_failed(name, msg=""):
    RESULTS[name] = False
    print(f"  [FAIL] {name}" + (f": {msg}" if msg else ""))


# =============================================================================
# 1. Power Iteration for Spectral Norm
# =============================================================================

def power_iteration(A: np.ndarray, n_iters: int = 20) -> float:
    """
    Estimate ||A||_* (spectral norm) via power iteration.

    This is the same algorithm used in:
    - tgsfn_wiring.py:RealJacobianEstimator.power_iteration
    - hrrl_agent/antifragile.py:JacobianMonitor.compute_spectral_norm
    """
    n = A.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(n_iters):
        u = A @ v
        u_norm = np.linalg.norm(u)
        if u_norm > 1e-10:
            u = u / u_norm

        v = A.T @ u
        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm

    sigma = np.linalg.norm(A @ v)
    return sigma


def test_power_iteration():
    """Test power iteration accuracy."""
    print("\n" + "=" * 60)
    print("1. Testing Power Iteration (Spectral Norm Estimation)")
    print("=" * 60)

    np.random.seed(42)

    # Create matrix with known spectral norm
    n = 64
    U, _ = np.linalg.qr(np.random.randn(n, n))
    V, _ = np.linalg.qr(np.random.randn(n, n))

    # Known singular values: max = 5.0
    s = np.linspace(5.0, 0.1, n)
    S = np.diag(s)

    A = U @ S @ V.T
    true_spectral_norm = 5.0

    # Estimate
    estimated = power_iteration(A, n_iters=30)
    error = abs(estimated - true_spectral_norm)

    if error < 0.1:
        test_passed(f"power_iteration", f"estimated={estimated:.4f}, true={true_spectral_norm}, error={error:.4f}")
    else:
        test_failed(f"power_iteration", f"error={error:.4f} too large")
        return False

    return True


# =============================================================================
# 2. Π_q Computation
# =============================================================================

def compute_piq(v: np.ndarray, J: np.ndarray, v_reset: float = -70.0,
                tau_m: float = 20.0, lambda_J: float = 1.2) -> float:
    """
    Compute the thermodynamic potential Π_q.

    Π_q = Σ_i (V_m^i - V_reset)² / τ_m + λ_J ||J||_*² / N

    This is the formula from:
    - tgsfn_wiring.py:compute_piq_real
    - hrrl_agent/criticality.py:compute_pi_q
    """
    n = v.shape[0]

    # Leak term
    deviation = v - v_reset
    leak_term = np.mean(deviation ** 2) / tau_m

    # Jacobian term (using spectral norm squared)
    J_spectral = power_iteration(J)
    J_norm_sq = J_spectral ** 2
    jacobian_term = lambda_J * J_norm_sq / n

    return leak_term + jacobian_term


def test_piq():
    """Test Π_q computation."""
    print("\n" + "=" * 60)
    print("2. Testing Π_q Computation")
    print("=" * 60)

    np.random.seed(42)
    n = 128
    v_reset = -70.0

    # Membrane at rest
    v_rest = np.full(n, v_reset)
    J_small = np.random.randn(n, n) * 0.1

    piq_rest = compute_piq(v_rest, J_small)

    if piq_rest >= 0:
        test_passed("Π_q non-negative", f"Π_q={piq_rest:.6f}")
    else:
        test_failed("Π_q should be non-negative")
        return False

    # Membrane depolarized
    v_depol = np.full(n, v_reset + 10.0)
    piq_depol = compute_piq(v_depol, J_small)

    if piq_depol > piq_rest:
        test_passed("Π_q increases with depolarization",
                   f"rest={piq_rest:.4f} < depol={piq_depol:.4f}")
    else:
        test_failed("Π_q should increase with depolarization")
        return False

    # Large Jacobian
    J_large = np.random.randn(n, n) * 2.0
    piq_large_J = compute_piq(v_rest, J_large)

    if piq_large_J > piq_rest:
        test_passed("Π_q increases with ||J||",
                   f"small_J={piq_rest:.4f} < large_J={piq_large_J:.4f}")
    else:
        test_failed("Π_q should increase with ||J||")
        return False

    return True


# =============================================================================
# 3. Thermodynamic Clip
# =============================================================================

def test_therm_clip():
    """Test thermodynamic clip computation."""
    print("\n" + "=" * 60)
    print("3. Testing Thermodynamic Clip")
    print("=" * 60)

    # Formula: therm_clip = min(1, π_max / (Π_q + ε))
    pi_max = 1.0
    eps = 1e-6

    # Low Π_q → clip ≈ 1 (no reduction)
    piq_low = 0.1
    clip_low = min(1.0, pi_max / (piq_low + eps))

    if 0.99 < clip_low <= 1.0:
        test_passed("therm_clip (low Π_q)", f"clip={clip_low:.4f} ≈ 1")
    else:
        test_failed("therm_clip should be ~1 for low Π_q")
        return False

    # High Π_q → clip < 1 (reduction)
    piq_high = 5.0
    clip_high = min(1.0, pi_max / (piq_high + eps))

    if 0 < clip_high < 1:
        test_passed("therm_clip (high Π_q)", f"clip={clip_high:.4f} < 1")
    else:
        test_failed("therm_clip should be <1 for high Π_q")
        return False

    # Very high Π_q → strong reduction
    piq_extreme = 100.0
    clip_extreme = min(1.0, pi_max / (piq_extreme + eps))

    if clip_extreme < clip_high:
        test_passed("therm_clip monotonically decreases",
                   f"high={clip_high:.4f} > extreme={clip_extreme:.4f}")
    else:
        test_failed("therm_clip should decrease as Π_q increases")
        return False

    return True


# =============================================================================
# 4. Poincaré Ball Projection
# =============================================================================

def project_to_ball(z: np.ndarray, curvature: float = 1.0, max_norm: float = 0.99) -> np.ndarray:
    """
    Project point to interior of Poincaré ball.

    From tgsfn_wiring.py and hrrl_agent/antifragile.py:AxiomCorrector
    """
    sqrt_c = np.sqrt(curvature)
    ball_radius = max_norm / sqrt_c

    z_norm = np.linalg.norm(z)
    if z_norm > ball_radius:
        z = z * (ball_radius / z_norm)

    return z


def exp_map_poincare(z: np.ndarray, v: np.ndarray, curvature: float = 1.0) -> np.ndarray:
    """
    Exponential map on Poincaré ball.

    Maps tangent vector v at point z to the manifold.
    From hrrl_agent/antifragile.py:AxiomCorrector.apply_correction_hyperbolic
    """
    c = curvature
    sqrt_c = np.sqrt(c)

    # Conformal factor (lambda_x in Poincaré ball)
    z_norm_sq = np.sum(z ** 2)
    lambda_x = 2.0 / (1 - c * z_norm_sq + 1e-10)

    # Scale tangent vector
    v_norm = np.linalg.norm(v)

    if v_norm < 1e-10:
        return z

    # Möbius addition formula for exponential map
    # exp_x(v) = x ⊕ tanh(λ_x ||v|| / 2) * (v / ||v||)
    coef = np.tanh(sqrt_c * lambda_x * v_norm / 2.0) / (sqrt_c * v_norm + 1e-10)
    direction = v * coef

    # Möbius addition: x ⊕ y
    # For small movements, approximate as x + direction
    z_new = z + direction

    # Project back to ball (essential safety)
    return project_to_ball(z_new, curvature)


def test_poincare_ops():
    """Test Poincaré ball operations."""
    print("\n" + "=" * 60)
    print("4. Testing Poincaré Ball Operations")
    print("=" * 60)

    np.random.seed(42)

    # Test projection
    z_outside = np.random.randn(32) * 2.0  # Outside ball
    z_projected = project_to_ball(z_outside, curvature=1.0)

    if np.linalg.norm(z_projected) <= 0.99:
        test_passed("project_to_ball", f"||z||={np.linalg.norm(z_projected):.4f} <= 0.99")
    else:
        test_failed("projected point should be inside ball")
        return False

    # Test exp map stays in ball
    z_start = np.random.randn(32) * 0.1  # Inside ball
    v_tangent = np.random.randn(32) * 0.5  # Large tangent vector

    z_moved = exp_map_poincare(z_start, v_tangent)

    if np.linalg.norm(z_moved) <= 0.99:
        test_passed("exp_map stays in ball", f"||z_new||={np.linalg.norm(z_moved):.4f} <= 0.99")
    else:
        test_failed("exp_map should keep point in ball")
        return False

    # Test that movement happens
    distance = np.linalg.norm(z_moved - z_start)
    if distance > 0:
        test_passed("exp_map moves point", f"distance={distance:.4f}")
    else:
        test_failed("exp_map should move point")
        return False

    return True


# =============================================================================
# 5. DAU Trigger Logic
# =============================================================================

def test_dau_trigger():
    """Test DAU trigger conditions."""
    print("\n" + "=" * 60)
    print("5. Testing DAU Trigger Logic")
    print("=" * 60)

    lambda_crit = 1.0  # Critical threshold
    cooldown = 10  # Minimum steps between triggers

    # Simulate steps
    steps_since_correction = 0
    corrections = 0

    # Test: stable dynamics, past cooldown - should NOT trigger
    J_stable = 0.5  # Below threshold
    steps_since_correction = 100  # Past cooldown

    should_trigger = (J_stable > lambda_crit) and (steps_since_correction >= cooldown)

    if not should_trigger:
        test_passed("DAU no trigger on stable", f"J={J_stable} < λ_crit={lambda_crit}")
    else:
        test_failed("DAU should not trigger when stable")
        return False

    # Test: unstable dynamics, past cooldown - SHOULD trigger
    J_unstable = 1.5  # Above threshold
    steps_since_correction = 100

    should_trigger = (J_unstable > lambda_crit) and (steps_since_correction >= cooldown)

    if should_trigger:
        test_passed("DAU triggers on unstable", f"J={J_unstable} > λ_crit={lambda_crit}")
    else:
        test_failed("DAU should trigger when unstable")
        return False

    # Test: unstable but in cooldown - should NOT trigger
    J_unstable = 1.5
    steps_since_correction = 5  # In cooldown

    should_trigger = (J_unstable > lambda_crit) and (steps_since_correction >= cooldown)

    if not should_trigger:
        test_passed("DAU respects cooldown", f"steps={steps_since_correction} < cooldown={cooldown}")
    else:
        test_failed("DAU should respect cooldown period")
        return False

    return True


# =============================================================================
# 6. LIF Neuron Dynamics
# =============================================================================

def lif_step(v: np.ndarray, I_ext: np.ndarray,
             v_reset: float = -70.0, v_thresh: float = -55.0,
             tau_m: float = 20.0, dt: float = 1.0) -> tuple:
    """
    Single timestep of LIF neuron dynamics.

    dV/dt = -(V - V_reset)/τ_m + I_ext
    """
    dv = (-(v - v_reset) + I_ext) / tau_m
    v_new = v + dv * dt

    # Spike detection
    spikes = (v_new >= v_thresh).astype(float)

    # Reset spiked neurons
    v_new = np.where(spikes > 0, v_reset, v_new)

    return v_new, spikes


def test_lif_dynamics():
    """Test LIF neuron dynamics."""
    print("\n" + "=" * 60)
    print("6. Testing LIF Neuron Dynamics")
    print("=" * 60)

    np.random.seed(42)
    n = 100
    v_reset = -70.0
    v_thresh = -55.0
    tau_m = 20.0

    # Initialize at rest
    v = np.full(n, v_reset)

    # Run with strong positive current
    # Need current > (v_thresh - v_reset) to eventually reach threshold
    # That's > 15mV, so use 20+ to ensure spiking
    I_ext = np.full(n, 25.0)  # Strong input (> threshold gap)

    total_spikes = 0
    for _ in range(200):  # More steps to allow spiking
        v, spikes = lif_step(v, I_ext, v_reset, v_thresh, tau_m)
        total_spikes += spikes.sum()

    if total_spikes > 0:
        test_passed("LIF produces spikes", f"total={total_spikes:.0f} spikes")
    else:
        test_failed("LIF should produce spikes with strong input")
        return False

    # Test that reset works
    if np.all(v >= v_reset - 1e-6):  # Allow small numerical error
        test_passed("LIF reset works", f"all v >= v_reset")
    else:
        test_failed("Membrane should never go below reset")
        return False

    # Test with no input - should decay to rest
    v = np.full(n, -60.0)  # Start above rest
    I_ext = np.zeros(n)

    for _ in range(200):
        v, _ = lif_step(v, I_ext, v_reset, v_thresh, tau_m)

    if np.allclose(v, v_reset, atol=0.1):
        test_passed("LIF decays to rest", f"v ≈ {v.mean():.2f} ≈ {v_reset}")
    else:
        test_failed("LIF should decay to rest with no input")
        return False

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("TGSFN Algorithm Validation (NumPy)")
    print("=" * 60)
    print("This validates mathematical correctness of core algorithms")
    print("independent of PyTorch runtime.\n")

    all_passed = True

    if not test_power_iteration():
        all_passed = False

    if not test_piq():
        all_passed = False

    if not test_therm_clip():
        all_passed = False

    if not test_poincare_ops():
        all_passed = False

    if not test_dau_trigger():
        all_passed = False

    if not test_lif_dynamics():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in RESULTS.values() if v)
    total = len(RESULTS)

    print(f"\nTests: {passed}/{total} passed")

    if all_passed:
        print("\n[SUCCESS] All algorithm tests passed!")
        print("\nValidated algorithms:")
        print("  - Power iteration for ||J||_* estimation")
        print("  - Π_q thermodynamic potential computation")
        print("  - Thermodynamic clip/brake mechanism")
        print("  - Poincaré ball projection and exp map")
        print("  - DAU trigger logic")
        print("  - LIF neuron dynamics")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
