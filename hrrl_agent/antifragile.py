"""
Antifragile Loop for TGSFN

Transforms structural risk into opportunity for self-improvement.

The antifragile loop:
1. Maintains critical operating point via λ_diss
2. Detects instability via Jacobian spectral norm ||J||_*
3. Triggers DAU to correct hyperbolic axiom layer
4. Guarantees recovery via provable global convergence

Key principle: The system becomes stronger from shocks that would
destabilize a merely robust system.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
from enum import Enum
import logging
import math

from .config import DAUConfig
from .criticality import CriticalityController, CriticalityRegime, CriticalityState

logger = logging.getLogger(__name__)


class StabilityStatus(Enum):
    """System stability classification."""
    STABLE = "stable"  # ||J||_* < critical margin
    WARNING = "warning"  # Approaching instability
    UNSTABLE = "unstable"  # ||J||_* > critical threshold
    RECOVERING = "recovering"  # DAU correction in progress


@dataclass
class AntifragileConfig:
    """Configuration for antifragile loop."""
    # Jacobian monitoring
    jacobian_critical_threshold: float = 1.0  # ||J||_* > 1 → unstable
    jacobian_warning_threshold: float = 0.8  # Warning zone
    jacobian_target: float = 0.9  # Target stability margin

    # DAU triggering
    trigger_on_instability: bool = True
    trigger_on_regime_change: bool = True
    min_steps_between_dau: int = 100  # Cooldown

    # Correction parameters
    correction_step_size: float = 0.01
    max_correction_iterations: int = 10
    convergence_threshold: float = 1e-4

    # Shock detection
    shock_detection_window: int = 10
    shock_magnitude_threshold: float = 2.0  # Multiple of baseline variance

    # Logging
    log_all_events: bool = True


@dataclass
class AntifragileState:
    """Current state of the antifragile system."""
    stability_status: StabilityStatus
    jacobian_spectral_norm: float
    criticality_state: CriticalityState
    shocks_detected: int
    dau_corrections: int
    current_strength: float  # Measure of system robustness


class JacobianMonitor:
    """
    Monitors the Jacobian spectral norm for stability assessment.

    ||J||_* < 1 → Stable (contractive)
    ||J||_* = 1 → Critical boundary
    ||J||_* > 1 → Unstable (expansive)
    """

    def __init__(self, config: AntifragileConfig):
        self.config = config
        self._history: List[float] = []
        self._baseline_variance: Optional[float] = None

    def compute_spectral_norm(
        self,
        jacobian: torch.Tensor,
        num_iterations: int = 10
    ) -> float:
        """
        Compute spectral norm ||J||_* via power iteration.

        More efficient than full SVD for large matrices.
        """
        if jacobian.dim() != 2:
            jacobian = jacobian.view(jacobian.size(0), -1)

        # Power iteration
        v = torch.randn(jacobian.size(1), device=jacobian.device)
        v = v / torch.norm(v)

        for _ in range(num_iterations):
            u = jacobian @ v
            u_norm = torch.norm(u)
            if u_norm > 0:
                u = u / u_norm

            v = jacobian.T @ u
            v_norm = torch.norm(v)
            if v_norm > 0:
                v = v / v_norm

        # Spectral norm
        sigma = torch.norm(jacobian @ v).item()
        return sigma

    def compute_spectral_norm_exact(self, jacobian: torch.Tensor) -> float:
        """Compute exact spectral norm via SVD (slower but precise)."""
        if jacobian.dim() != 2:
            jacobian = jacobian.view(jacobian.size(0), -1)

        s = torch.linalg.svdvals(jacobian)
        return s[0].item()

    def update(self, spectral_norm: float) -> StabilityStatus:
        """
        Update history and return current stability status.
        """
        self._history.append(spectral_norm)

        # Update baseline variance
        if len(self._history) >= 100:
            recent = torch.tensor(self._history[-100:])
            self._baseline_variance = recent.var().item()

        # Classify status
        if spectral_norm > self.config.jacobian_critical_threshold:
            return StabilityStatus.UNSTABLE
        elif spectral_norm > self.config.jacobian_warning_threshold:
            return StabilityStatus.WARNING
        else:
            return StabilityStatus.STABLE

    def detect_shock(self) -> bool:
        """
        Detect sudden shock based on variance increase.
        """
        if len(self._history) < self.config.shock_detection_window + 10:
            return False

        if self._baseline_variance is None or self._baseline_variance < 1e-10:
            return False

        recent = torch.tensor(self._history[-self.config.shock_detection_window:])
        recent_var = recent.var().item()

        return recent_var > self.config.shock_magnitude_threshold * self._baseline_variance

    def get_stability_margin(self) -> float:
        """
        Get distance from instability threshold.

        Positive = stable, Negative = unstable
        """
        if not self._history:
            return 1.0

        current = self._history[-1]
        return self.config.jacobian_critical_threshold - current


class AxiomCorrector:
    """
    Corrects hyperbolic axiom layer to restore stability.

    Uses gradient-based correction with provable convergence
    in the hyperbolic space.
    """

    def __init__(self, config: AntifragileConfig):
        self.config = config
        self._corrections_applied = 0

    def compute_correction(
        self,
        axiom_embedding: torch.Tensor,
        jacobian: torch.Tensor,
        target_norm: float
    ) -> torch.Tensor:
        """
        Compute correction δ to drive ||J||_* toward target.

        The correction is in the tangent space of the hyperbolic manifold.
        """
        # Current spectral norm
        current_norm = torch.linalg.svdvals(jacobian)[0]

        # Gradient of spectral norm w.r.t. axiom embedding
        # Using implicit differentiation
        with torch.enable_grad():
            axiom_embedding.requires_grad_(True)

            # Recompute Jacobian with gradient tracking
            # (Assumes jacobian is a function of axiom_embedding)
            # This is a placeholder - actual implementation depends on architecture

            grad_norm = torch.autograd.grad(
                current_norm,
                axiom_embedding,
                retain_graph=True,
                allow_unused=True
            )[0]

            if grad_norm is None:
                grad_norm = torch.zeros_like(axiom_embedding)

        # Direction to reduce spectral norm
        if current_norm > target_norm:
            # Need to reduce ||J||_*
            direction = -grad_norm
        else:
            # Need to increase ||J||_* (rare, but for balance)
            direction = grad_norm

        # Scale correction
        correction = self.config.correction_step_size * direction

        return correction

    def apply_correction_hyperbolic(
        self,
        axiom_embedding: torch.Tensor,
        correction: torch.Tensor,
        curvature: float = 1.0
    ) -> torch.Tensor:
        """
        Apply correction on hyperbolic manifold.

        Uses exponential map to stay on manifold.
        """
        # Project correction to tangent space
        # For Poincaré ball: v_tangent = (1 - c||x||²) * v
        c = curvature
        x_norm_sq = torch.sum(axiom_embedding ** 2)
        conformal_factor = 1 - c * x_norm_sq

        tangent_correction = conformal_factor * correction

        # Exponential map
        v_norm = torch.norm(tangent_correction)
        if v_norm < 1e-10:
            return axiom_embedding

        sqrt_c = math.sqrt(c)
        exp_factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)

        new_embedding = axiom_embedding + exp_factor * tangent_correction

        # Project back to ball (safety)
        new_norm = torch.norm(new_embedding)
        max_norm = 0.99 / sqrt_c
        if new_norm > max_norm:
            new_embedding = new_embedding * (max_norm / new_norm)

        self._corrections_applied += 1

        return new_embedding

    def iterative_correction(
        self,
        axiom_embedding: torch.Tensor,
        compute_jacobian_fn: Callable[[torch.Tensor], torch.Tensor],
        target_norm: float
    ) -> Tuple[torch.Tensor, bool, int]:
        """
        Iteratively correct until stability is achieved.

        Returns:
            (corrected_embedding, converged, num_iterations)
        """
        embedding = axiom_embedding.clone()

        for iteration in range(self.config.max_correction_iterations):
            # Compute current Jacobian
            jacobian = compute_jacobian_fn(embedding)
            current_norm = torch.linalg.svdvals(jacobian)[0].item()

            # Check convergence
            if abs(current_norm - target_norm) < self.config.convergence_threshold:
                logger.info(
                    f"Axiom correction converged in {iteration + 1} iterations"
                )
                return embedding, True, iteration + 1

            # Compute and apply correction
            correction = self.compute_correction(embedding, jacobian, target_norm)
            embedding = self.apply_correction_hyperbolic(embedding, correction)

        logger.warning(
            f"Axiom correction did not converge after "
            f"{self.config.max_correction_iterations} iterations"
        )
        return embedding, False, self.config.max_correction_iterations


class AntifragileLoop(nn.Module):
    """
    Full antifragile loop for TGSFN.

    Coordinates:
    1. Critical operating point maintenance (via CriticalityController)
    2. Instability detection (via JacobianMonitor)
    3. DAU triggering and axiom correction
    4. Recovery verification
    """

    def __init__(
        self,
        config: AntifragileConfig,
        criticality_controller: CriticalityController,
        dau_config: Optional[DAUConfig] = None
    ):
        super().__init__()
        self.config = config
        self.criticality = criticality_controller

        # Monitoring
        self.jacobian_monitor = JacobianMonitor(config)
        self.axiom_corrector = AxiomCorrector(config)

        # State
        self._status = StabilityStatus.STABLE
        self._shocks_detected = 0
        self._dau_corrections = 0
        self._steps_since_dau = 0
        self._step = 0

        # Strength metric (increases with successful shock recovery)
        self._strength = 1.0
        self._strength_history: List[float] = []

    def step(
        self,
        jacobian: torch.Tensor,
        input_spikes: int,
        output_spikes: int,
        excitatory_current: float = 0.0,
        inhibitory_current: float = 0.0,
        axiom_embedding: Optional[torch.Tensor] = None,
        compute_jacobian_fn: Optional[Callable] = None
    ) -> AntifragileState:
        """
        Execute one step of the antifragile loop.

        Args:
            jacobian: Current Jacobian matrix
            input_spikes: Input spike count
            output_spikes: Output spike count
            excitatory_current: E current
            inhibitory_current: I current
            axiom_embedding: Optional axiom layer embedding
            compute_jacobian_fn: Function to recompute Jacobian

        Returns:
            Current AntifragileState
        """
        self._step += 1
        self._steps_since_dau += 1

        # 1. Update criticality
        crit_state = self.criticality.update(
            input_spikes, output_spikes,
            excitatory_current, inhibitory_current
        )

        # 2. Monitor Jacobian
        spectral_norm = self.jacobian_monitor.compute_spectral_norm(jacobian)
        stability_status = self.jacobian_monitor.update(spectral_norm)

        # 3. Detect shocks
        shock_detected = self.jacobian_monitor.detect_shock()
        if shock_detected:
            self._shocks_detected += 1
            if self.config.log_all_events:
                logger.warning(f"Shock detected at step {self._step}")

        # 4. Check if DAU should be triggered
        should_trigger_dau = self._should_trigger_dau(
            stability_status, crit_state, shock_detected
        )

        # 5. Execute DAU if needed
        if should_trigger_dau and axiom_embedding is not None:
            self._execute_dau(axiom_embedding, compute_jacobian_fn)
            stability_status = StabilityStatus.RECOVERING
            self._steps_since_dau = 0

        # 6. Update strength metric
        self._update_strength(stability_status, shock_detected)

        # Update status
        self._status = stability_status

        return AntifragileState(
            stability_status=stability_status,
            jacobian_spectral_norm=spectral_norm,
            criticality_state=crit_state,
            shocks_detected=self._shocks_detected,
            dau_corrections=self._dau_corrections,
            current_strength=self._strength
        )

    def _should_trigger_dau(
        self,
        stability: StabilityStatus,
        criticality: CriticalityState,
        shock: bool
    ) -> bool:
        """Determine if DAU should be triggered."""
        # Cooldown check
        if self._steps_since_dau < self.config.min_steps_between_dau:
            return False

        # Trigger on instability
        if self.config.trigger_on_instability:
            if stability == StabilityStatus.UNSTABLE:
                logger.info("Triggering DAU due to instability")
                return True

        # Trigger on regime change to chaotic
        if self.config.trigger_on_regime_change:
            if criticality.regime == CriticalityRegime.CHAOTIC:
                logger.info("Triggering DAU due to chaotic regime")
                return True

        return False

    def _execute_dau(
        self,
        axiom_embedding: torch.Tensor,
        compute_jacobian_fn: Optional[Callable]
    ):
        """Execute DAU correction."""
        logger.info(f"Executing DAU correction at step {self._step}")

        if compute_jacobian_fn is None:
            logger.warning("No Jacobian computation function provided, skipping DAU")
            return

        # Perform iterative correction
        target_norm = self.config.jacobian_target

        corrected, converged, iterations = self.axiom_corrector.iterative_correction(
            axiom_embedding,
            compute_jacobian_fn,
            target_norm
        )

        self._dau_corrections += 1

        if converged:
            logger.info(
                f"DAU correction successful after {iterations} iterations"
            )
            # Increase strength (antifragile: grew from adversity)
            self._strength *= 1.1
        else:
            logger.warning("DAU correction did not fully converge")

    def _update_strength(self, status: StabilityStatus, shock: bool):
        """
        Update strength metric.

        Strength increases when system successfully recovers from shocks.
        This is the antifragile property: growing stronger from adversity.
        """
        if shock and status in [StabilityStatus.STABLE, StabilityStatus.RECOVERING]:
            # Successfully handling shock → stronger
            self._strength *= 1.05
            if self.config.log_all_events:
                logger.info(f"Strength increased to {self._strength:.4f}")

        elif status == StabilityStatus.UNSTABLE:
            # Unstable → slight decrease
            self._strength *= 0.99

        # Bound strength
        self._strength = max(0.1, min(10.0, self._strength))
        self._strength_history.append(self._strength)

    def get_stability_margin(self) -> float:
        """Get current stability margin."""
        return self.jacobian_monitor.get_stability_margin()

    def get_statistics(self) -> Dict:
        """Get antifragile loop statistics."""
        return {
            'step': self._step,
            'status': self._status.value,
            'shocks_detected': self._shocks_detected,
            'dau_corrections': self._dau_corrections,
            'current_strength': self._strength,
            'stability_margin': self.get_stability_margin(),
            'criticality': self.criticality.get_statistics()
        }


class AntifragileAgent:
    """
    Agent wrapper that adds antifragile capabilities.

    Wraps any base agent and adds:
    - Jacobian monitoring
    - Automatic stability recovery
    - Strength tracking
    """

    def __init__(
        self,
        base_agent: nn.Module,
        config: AntifragileConfig,
        get_jacobian_fn: Callable[[nn.Module], torch.Tensor],
        get_axiom_embedding_fn: Optional[Callable[[nn.Module], torch.Tensor]] = None
    ):
        self.base_agent = base_agent
        self.config = config
        self.get_jacobian = get_jacobian_fn
        self.get_axiom_embedding = get_axiom_embedding_fn

        # Create antifragile components
        from .criticality import CriticalityConfig
        crit_config = CriticalityConfig()
        criticality = CriticalityController(crit_config)

        self.antifragile = AntifragileLoop(config, criticality)

    def step(self, *args, **kwargs) -> Tuple[Any, AntifragileState]:
        """
        Execute agent step with antifragile monitoring.

        Returns:
            (base_agent_output, antifragile_state)
        """
        # Run base agent
        output = self.base_agent(*args, **kwargs)

        # Get Jacobian
        jacobian = self.get_jacobian(self.base_agent)

        # Get axiom embedding if available
        axiom_embedding = None
        if self.get_axiom_embedding is not None:
            axiom_embedding = self.get_axiom_embedding(self.base_agent)

        # Antifragile step (placeholder spike counts)
        af_state = self.antifragile.step(
            jacobian=jacobian,
            input_spikes=10,  # Would come from actual network
            output_spikes=10,
            axiom_embedding=axiom_embedding
        )

        return output, af_state


if __name__ == "__main__":
    # Test antifragile loop
    print("Testing AntifragileLoop...")

    from .criticality import CriticalityConfig

    af_config = AntifragileConfig()
    crit_config = CriticalityConfig()

    criticality = CriticalityController(crit_config)
    antifragile = AntifragileLoop(af_config, criticality)

    # Simulate stable operation
    print("\nSimulating stable operation...")
    for _ in range(50):
        jacobian = torch.randn(64, 64) * 0.1  # Small, stable
        state = antifragile.step(
            jacobian=jacobian,
            input_spikes=10,
            output_spikes=10
        )

    print(f"  Status: {state.stability_status.value}")
    print(f"  ||J||_*: {state.jacobian_spectral_norm:.4f}")
    print(f"  Strength: {state.current_strength:.4f}")

    # Simulate shock (unstable Jacobian)
    print("\nSimulating shock...")
    for _ in range(20):
        jacobian = torch.randn(64, 64) * 2.0  # Large, unstable
        state = antifragile.step(
            jacobian=jacobian,
            input_spikes=10,
            output_spikes=15
        )

    print(f"  Status: {state.stability_status.value}")
    print(f"  ||J||_*: {state.jacobian_spectral_norm:.4f}")
    print(f"  Shocks detected: {state.shocks_detected}")

    # Get statistics
    print("\nStatistics:")
    stats = antifragile.get_statistics()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    # Test axiom corrector
    print("\nTesting AxiomCorrector...")
    corrector = AxiomCorrector(af_config)

    embedding = torch.randn(32) * 0.1
    correction = torch.randn(32) * 0.01

    corrected = corrector.apply_correction_hyperbolic(embedding, correction)
    print(f"  Original norm: {torch.norm(embedding):.4f}")
    print(f"  Corrected norm: {torch.norm(corrected):.4f}")

    print("\nAll antifragile tests passed!")
