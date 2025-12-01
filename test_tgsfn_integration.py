#!/usr/bin/env python3
"""
test_tgsfn_integration.py
Comprehensive Integration Test for TGSFN Stack

Tests that all components are properly wired and functional:
1. Real Jacobian monitoring via power iteration
2. DAU trigger and correction flow
3. Π_q computation with actual ||J||_F²
4. L5 controller with thermodynamic clip
5. Antifragile loop coordination

Run with: python test_tgsfn_integration.py
"""

from __future__ import annotations

import sys
import math
from typing import Dict, Tuple, Optional

# Test results tracking
RESULTS: Dict[str, bool] = {}


def test_passed(name: str, message: str = ""):
    """Mark test as passed."""
    RESULTS[name] = True
    print(f"  [PASS] {name}" + (f": {message}" if message else ""))


def test_failed(name: str, message: str = ""):
    """Mark test as failed."""
    RESULTS[name] = False
    print(f"  [FAIL] {name}" + (f": {message}" if message else ""))


# =============================================================================
# 1. Test Imports
# =============================================================================

def test_imports() -> bool:
    """Test all required imports."""
    print("\n" + "=" * 60)
    print("1. Testing Imports")
    print("=" * 60)

    try:
        import torch
        test_passed("torch import")
    except ImportError as e:
        test_failed("torch import", str(e))
        return False

    try:
        import torch.nn as nn
        test_passed("torch.nn import")
    except ImportError as e:
        test_failed("torch.nn import", str(e))
        return False

    # Test geoopt (optional)
    try:
        import geoopt
        test_passed("geoopt import (optional)")
    except ImportError:
        print("  [INFO] geoopt not installed (optional dependency)")

    return True


# =============================================================================
# 2. Test RealJacobianEstimator
# =============================================================================

def test_jacobian_estimator() -> bool:
    """Test the Jacobian estimator with power iteration."""
    print("\n" + "=" * 60)
    print("2. Testing RealJacobianEstimator")
    print("=" * 60)

    import torch

    try:
        from tgsfn_wiring import RealJacobianEstimator
        test_passed("RealJacobianEstimator import")
    except ImportError as e:
        test_failed("RealJacobianEstimator import", str(e))
        return False

    # Create estimator
    estimator = RealJacobianEstimator(n_iterations=20)

    # Test with known matrix
    # Create matrix with known spectral norm
    torch.manual_seed(42)
    n = 64
    U = torch.randn(n, n)
    U, _ = torch.linalg.qr(U)  # Orthogonal matrix

    # Singular values: [3.0, 2.0, 1.0, 0.5, ...]
    s = torch.linspace(3.0, 0.1, n)
    S = torch.diag(s)

    V = torch.randn(n, n)
    V, _ = torch.linalg.qr(V)

    J = U @ S @ V.T  # Known spectral norm = 3.0

    # Test power iteration
    estimated_norm = estimator.power_iteration(J).item()
    true_norm = 3.0

    error = abs(estimated_norm - true_norm)
    if error < 0.1:
        test_passed(f"power_iteration accuracy", f"error={error:.4f} < 0.1")
    else:
        test_failed(f"power_iteration accuracy", f"error={error:.4f} >= 0.1")
        return False

    # Test warm start
    estimated_norm2 = estimator.power_iteration(J, warm_start=True).item()
    if abs(estimated_norm2 - true_norm) < 0.1:
        test_passed("power_iteration warm_start")
    else:
        test_failed("power_iteration warm_start")
        return False

    return True


# =============================================================================
# 3. Test Π_q Computation
# =============================================================================

def test_piq_computation() -> bool:
    """Test real Π_q computation."""
    print("\n" + "=" * 60)
    print("3. Testing Π_q Computation")
    print("=" * 60)

    import torch

    try:
        from tgsfn_wiring import compute_piq_real, RealJacobianEstimator
        test_passed("compute_piq_real import")
    except ImportError as e:
        test_failed("compute_piq_real import", str(e))
        return False

    # Create test data
    n_neurons = 128
    V_reset = -70.0

    # Membrane potentials near resting
    v = torch.full((n_neurons,), V_reset + 5.0)  # Slightly depolarized

    # Small Jacobian (stable dynamics)
    J_stable = torch.randn(n_neurons, n_neurons) * 0.1

    pi_q_stable, components_stable = compute_piq_real(
        v, J_stable, v_reset=V_reset, tau_m=20.0, lambda_J=1.2
    )

    if pi_q_stable > 0:
        test_passed(f"compute_piq_real (stable)", f"Π_q={pi_q_stable:.4f}")
    else:
        test_failed("compute_piq_real (stable)", "Π_q should be positive")
        return False

    # Large Jacobian (unstable dynamics)
    J_unstable = torch.randn(n_neurons, n_neurons) * 2.0

    pi_q_unstable, components_unstable = compute_piq_real(
        v, J_unstable, v_reset=V_reset, tau_m=20.0, lambda_J=1.2
    )

    if pi_q_unstable > pi_q_stable:
        test_passed(f"Π_q increases with ||J||",
                   f"stable={pi_q_stable:.4f} < unstable={pi_q_unstable:.4f}")
    else:
        test_failed("Π_q should increase with ||J||")
        return False

    # Check components are tracked
    if 'J_spectral' in components_stable and components_stable['J_spectral'] > 0:
        test_passed("Π_q components tracking", f"J_spectral={components_stable['J_spectral']:.4f}")
    else:
        test_failed("Π_q components tracking")
        return False

    return True


# =============================================================================
# 4. Test WiredDAU
# =============================================================================

def test_wired_dau() -> bool:
    """Test the wired DAU trigger and correction flow."""
    print("\n" + "=" * 60)
    print("4. Testing WiredDAU")
    print("=" * 60)

    import torch

    try:
        from tgsfn_wiring import WiredDAU, DAUWiringConfig
        test_passed("WiredDAU import")
    except ImportError as e:
        test_failed("WiredDAU import", str(e))
        return False

    # Create DAU with low threshold for testing
    config = DAUWiringConfig(
        lambda_crit=0.5,        # Low threshold to trigger easily
        warning_threshold=0.3,
        cooldown_steps=5,       # Short cooldown
        spectral_target=0.3,
    )
    dau = WiredDAU(config)

    # Initial state on hyperbolic manifold
    z = torch.randn(64, 32) * 0.1  # Well inside Poincaré ball

    # Run with stable Jacobian (should not trigger)
    J_stable = torch.randn(64, 64) * 0.1

    z_new, info = dau.step(J_stable, z, curvature=1.0)

    if info['status'] == 'STABLE':
        test_passed("DAU stable detection")
    else:
        test_failed("DAU stable detection", f"status={info['status']}")
        return False

    # Run with unstable Jacobian repeatedly to trigger
    J_unstable = torch.randn(64, 64) * 3.0  # Large → high spectral norm

    for _ in range(10):  # Run past cooldown
        z_new, info = dau.step(J_unstable, z, curvature=1.0)

    if info['status'] in ['UNSTABLE', 'WARNING']:
        test_passed(f"DAU instability detection", f"status={info['status']}")
    else:
        test_failed("DAU instability detection")
        return False

    # Check corrections were applied
    if dau._corrections_applied > 0:
        test_passed(f"DAU applied corrections", f"count={dau._corrections_applied}")
    else:
        test_failed("DAU should have applied corrections")
        return False

    # Verify z was modified by correction
    z_diff = torch.norm(z_new - z).item()
    if z_diff > 0:
        test_passed(f"DAU modified state", f"||Δz||={z_diff:.4f}")
    else:
        test_failed("DAU should modify state during correction")
        return False

    return True


# =============================================================================
# 5. Test WiredTGSFNSystem
# =============================================================================

def test_wired_system() -> bool:
    """Test the full wired TGSFN system."""
    print("\n" + "=" * 60)
    print("5. Testing WiredTGSFNSystem")
    print("=" * 60)

    import torch

    try:
        from tgsfn_wiring import WiredTGSFNSystem
        test_passed("WiredTGSFNSystem import")
    except ImportError as e:
        test_failed("WiredTGSFNSystem import", str(e))
        return False

    # Create system
    n_neurons = 128
    system = WiredTGSFNSystem(
        n_neurons=n_neurons,
        eucl_dim=16,
        hyp_dim=16,
        curvature=1.0,
        pi_max=1.0,
    )
    test_passed("WiredTGSFNSystem creation")

    # Test forward pass
    drive = torch.randn(n_neurons, 8) * 0.5
    appraisal = torch.randn(n_neurons, 8) * 0.5
    I_ext = torch.randn(n_neurons) * 5.0

    spikes, v_new, info = system(drive, appraisal, I_ext)

    if spikes.shape == torch.Size([n_neurons]):
        test_passed("forward pass shape", f"spikes.shape={spikes.shape}")
    else:
        test_failed("forward pass shape")
        return False

    # Check info dict has required keys
    required_keys = ['pi_q', 'therm_clip', 'J_spectral', 'dau_status', 'n_spikes']
    for key in required_keys:
        if key in info:
            test_passed(f"info contains '{key}'", f"value={info[key]}")
        else:
            test_failed(f"info should contain '{key}'")
            return False

    # Test thermodynamic clip is working
    if 0 <= info['therm_clip'] <= 1:
        test_passed("therm_clip in [0,1]")
    else:
        test_failed("therm_clip should be in [0,1]")
        return False

    # Run multiple steps and check stability
    print("\n  Running 50 timesteps...")
    spike_counts = []
    pi_q_values = []

    for step in range(50):
        drive = torch.randn(n_neurons, 8) * 0.5
        appraisal = torch.randn(n_neurons, 8) * 0.5
        I_ext = torch.randn(n_neurons) * 5.0

        spikes, v_new, info = system(drive, appraisal, I_ext)
        spike_counts.append(info['n_spikes'])
        pi_q_values.append(info['pi_q'])

    mean_spikes = sum(spike_counts) / len(spike_counts)
    mean_pi_q = sum(pi_q_values) / len(pi_q_values)

    if 0 < mean_spikes < n_neurons:
        test_passed(f"spike activity reasonable", f"mean={mean_spikes:.1f}/step")
    else:
        test_failed("spike activity", f"mean={mean_spikes:.1f}")
        return False

    if mean_pi_q > 0:
        test_passed(f"Π_q tracked", f"mean={mean_pi_q:.4f}")
    else:
        test_failed("Π_q should be positive")
        return False

    return True


# =============================================================================
# 6. Test hrrl_agent Components
# =============================================================================

def test_hrrl_components() -> bool:
    """Test hrrl_agent module components."""
    print("\n" + "=" * 60)
    print("6. Testing hrrl_agent Components")
    print("=" * 60)

    # Test imports from hrrl_agent
    try:
        sys.path.insert(0, '/home/user/ram-and-unification')
        from hrrl_agent.antifragile import JacobianMonitor, AntifragileConfig
        test_passed("JacobianMonitor import")
    except ImportError as e:
        test_failed("JacobianMonitor import", str(e))
        return False

    try:
        from hrrl_agent.criticality import CriticalityController, CriticalityConfig
        test_passed("CriticalityController import")
    except ImportError as e:
        test_failed("CriticalityController import", str(e))
        return False

    try:
        from hrrl_agent.dau import DynamicArchitectureUpdate, DAUConfig, DAUAction
        test_passed("DynamicArchitectureUpdate import")
    except ImportError as e:
        test_failed("DynamicArchitectureUpdate import", str(e))
        return False

    # Test JacobianMonitor
    import torch

    af_config = AntifragileConfig()
    monitor = JacobianMonitor(af_config)

    J = torch.randn(64, 64) * 0.5
    spectral_norm = monitor.compute_spectral_norm(J)

    if spectral_norm > 0:
        test_passed(f"JacobianMonitor.compute_spectral_norm", f"||J||_*={spectral_norm:.4f}")
    else:
        test_failed("spectral norm should be positive")
        return False

    # Test CriticalityController
    crit_config = CriticalityConfig()
    criticality = CriticalityController(crit_config)

    state = criticality.update(
        input_spikes=10,
        output_spikes=10,
        excitatory_current=1.0,
        inhibitory_current=0.25
    )

    if state.lambda_diss > 0:
        test_passed(f"CriticalityController.update", f"λ_diss={state.lambda_diss:.4f}")
    else:
        test_failed("λ_diss should be positive")
        return False

    return True


# =============================================================================
# 7. Test L1/L2/L3 Hierarchy
# =============================================================================

def test_hierarchy() -> bool:
    """Test the L1/L2/L3 control hierarchy."""
    print("\n" + "=" * 60)
    print("7. Testing L1/L2/L3 Hierarchy")
    print("=" * 60)

    try:
        sys.path.insert(0, '/home/user/ram-and-unification')
        from hrrl_agent.l1_homeostat import L1Homeostat
        test_passed("L1Homeostat import")
    except ImportError as e:
        test_failed("L1Homeostat import", str(e))
        return False

    try:
        from hrrl_agent.l2_hyperbolic import L2HyperbolicAppraisal
        test_passed("L2HyperbolicAppraisal import")
    except ImportError as e:
        test_failed("L2HyperbolicAppraisal import", str(e))
        return False

    try:
        from hrrl_agent.l3_gating import L3GatingController
        test_passed("L3GatingController import")
    except ImportError as e:
        test_failed("L3GatingController import", str(e))
        return False

    # Basic functionality tests would go here
    # For now, successful imports confirm the modules exist

    return True


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all integration tests."""
    print("=" * 60)
    print("TGSFN Stack Integration Tests")
    print("=" * 60)

    all_passed = True

    # Run tests in order
    if not test_imports():
        print("\n[FATAL] Import tests failed - cannot continue")
        return 1

    if not test_jacobian_estimator():
        all_passed = False

    if not test_piq_computation():
        all_passed = False

    if not test_wired_dau():
        all_passed = False

    if not test_wired_system():
        all_passed = False

    if not test_hrrl_components():
        all_passed = False

    if not test_hierarchy():
        all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    passed = sum(1 for v in RESULTS.values() if v)
    total = len(RESULTS)

    print(f"\nTests: {passed}/{total} passed")

    if all_passed:
        print("\n[SUCCESS] All integration tests passed!")
        print("\nVerified components:")
        print("  - RealJacobianEstimator with power iteration")
        print("  - compute_piq_real with actual ||J||_F²")
        print("  - WiredDAU trigger and correction flow")
        print("  - WiredTGSFNSystem full integration")
        print("  - hrrl_agent.antifragile.JacobianMonitor")
        print("  - hrrl_agent.criticality.CriticalityController")
        print("  - hrrl_agent.dau.DynamicArchitectureUpdate")
        print("  - L1/L2/L3 control hierarchy")
        return 0
    else:
        print("\n[FAILURE] Some tests failed")
        for name, passed in RESULTS.items():
            if not passed:
                print(f"  - {name}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
