# tfan/antifragility_monitor.py
# TGSFN Antifragility Monitor
#
# Implements the Thermodynamic Geometry of Stochastic Field Networks (TGSFN)
# concepts for monitoring system antifragility.
#
# Key concepts:
# - Π_q (entropy production): measures distance from equilibrium
# - Jacobian spectral analysis: tracks dynamical stability
# - DAU (Dynamic Antifragility Unit): triggers antifragile responses
# - Convexity detection: identifies opportunities to gain from volatility

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math
from collections import deque


# ============================================================================
#  Antifragility Metrics
# ============================================================================

@dataclass
class AntifragilityMetrics:
    """Current antifragility state."""
    # Entropy production (Π_q)
    pi_q: float = 0.0                    # Current entropy production rate
    pi_q_mean: float = 0.0               # Rolling mean
    pi_q_trend: float = 0.0              # Trend direction (-1 to 1)

    # Jacobian spectral analysis
    jacobian_spectral_norm: float = 0.0  # Largest singular value
    jacobian_trace: float = 0.0          # Trace (sum of eigenvalues)
    jacobian_condition: float = 1.0      # Condition number

    # Stability indicators
    lyapunov_proxy: float = 0.0          # Proxy for max Lyapunov exponent
    stability_margin: float = 1.0        # Distance from instability (>0 stable)

    # Antifragility scores
    convexity: float = 0.0               # Second derivative of performance
    antifragility_index: float = 0.0     # Overall antifragility score
    fragility_exposure: float = 0.0      # Downside risk exposure

    # DAU state
    dau_active: bool = False
    dau_trigger_reason: str = ""
    dau_action_suggested: str = ""


@dataclass
class DAUTrigger:
    """Dynamic Antifragility Unit trigger event."""
    timestamp: float
    step: int
    trigger_type: str       # "instability", "opportunity", "stress_test", "recovery"
    severity: float         # 0-1
    metrics_snapshot: Dict[str, float]
    action_suggested: str
    was_acted_upon: bool = False


# ============================================================================
#  Antifragility Monitor
# ============================================================================

class AntifragilityMonitor:
    """
    TGSFN Antifragility Monitor.

    Tracks entropy production, dynamical stability, and convexity
    to detect antifragile vs fragile system states.

    Antifragile: gains from volatility (convex payoff)
    Robust: unaffected by volatility (linear payoff)
    Fragile: harmed by volatility (concave payoff)
    """

    def __init__(
        self,
        history_size: int = 100,
        pi_q_threshold: float = 0.5,
        instability_threshold: float = 0.8,
        convexity_window: int = 20,
        dau_cooldown_steps: int = 50,
    ):
        """
        Args:
            history_size: Rolling window for statistics
            pi_q_threshold: Entropy production level triggering DAU
            instability_threshold: Stability margin triggering DAU
            convexity_window: Window for convexity estimation
            dau_cooldown_steps: Minimum steps between DAU triggers
        """
        self.history_size = history_size
        self.pi_q_threshold = pi_q_threshold
        self.instability_threshold = instability_threshold
        self.convexity_window = convexity_window
        self.dau_cooldown_steps = dau_cooldown_steps

        # History buffers
        self.pi_q_history: deque = deque(maxlen=history_size)
        self.loss_history: deque = deque(maxlen=history_size)
        self.grad_norm_history: deque = deque(maxlen=history_size)
        self.sigma_history: deque = deque(maxlen=history_size)

        # DAU state
        self.dau_triggers: List[DAUTrigger] = []
        self.last_dau_step: int = -1000
        self.current_step: int = 0

        # Current metrics
        self._current_metrics = AntifragilityMetrics()

    def update(
        self,
        step: int,
        loss: float,
        grad_norm: float,
        sigma_proxy: float,
        epsilon_proxy: float,
        weights_delta_norm: Optional[float] = None,
        second_order_info: Optional[Dict[str, float]] = None,
    ) -> AntifragilityMetrics:
        """
        Update antifragility metrics with new observations.

        Args:
            step: Current training step
            loss: Current loss value
            grad_norm: Current gradient norm
            sigma_proxy: UDK sigma proxy (information compression)
            epsilon_proxy: UDK epsilon proxy (dissipation rate)
            weights_delta_norm: Optional L2 norm of weight changes
            second_order_info: Optional Hessian/curvature information

        Returns:
            Updated AntifragilityMetrics
        """
        self.current_step = step

        # === Compute Π_q (entropy production proxy) ===
        # Π_q ≈ ε * σ + grad_norm variance (energy dissipation)
        pi_q = self._compute_entropy_production(
            epsilon_proxy, sigma_proxy, grad_norm
        )
        self.pi_q_history.append(pi_q)

        # === Update other histories ===
        self.loss_history.append(loss)
        self.grad_norm_history.append(grad_norm)
        self.sigma_history.append(sigma_proxy)

        # === Compute Jacobian spectral properties (proxy) ===
        jacobian_metrics = self._estimate_jacobian_metrics(
            grad_norm, weights_delta_norm, second_order_info
        )

        # === Compute stability indicators ===
        lyapunov_proxy = self._estimate_lyapunov(grad_norm, loss)
        stability_margin = self._compute_stability_margin(
            jacobian_metrics["spectral_norm"], lyapunov_proxy
        )

        # === Compute convexity (antifragility indicator) ===
        convexity = self._estimate_convexity()
        antifragility_index = self._compute_antifragility_index(
            convexity, stability_margin, pi_q
        )
        fragility_exposure = max(0.0, -convexity) * (1.0 - stability_margin)

        # === Rolling statistics for Π_q ===
        pi_q_mean = sum(self.pi_q_history) / len(self.pi_q_history) if self.pi_q_history else 0.0
        pi_q_trend = self._compute_trend(list(self.pi_q_history))

        # === Check DAU triggers ===
        dau_active, trigger_reason, action = self._check_dau_triggers(
            step, pi_q, stability_margin, convexity, antifragility_index
        )

        # === Build metrics ===
        self._current_metrics = AntifragilityMetrics(
            pi_q=pi_q,
            pi_q_mean=pi_q_mean,
            pi_q_trend=pi_q_trend,
            jacobian_spectral_norm=jacobian_metrics["spectral_norm"],
            jacobian_trace=jacobian_metrics["trace"],
            jacobian_condition=jacobian_metrics["condition"],
            lyapunov_proxy=lyapunov_proxy,
            stability_margin=stability_margin,
            convexity=convexity,
            antifragility_index=antifragility_index,
            fragility_exposure=fragility_exposure,
            dau_active=dau_active,
            dau_trigger_reason=trigger_reason,
            dau_action_suggested=action,
        )

        return self._current_metrics

    def _compute_entropy_production(
        self,
        epsilon: float,
        sigma: float,
        grad_norm: float,
    ) -> float:
        """
        Compute entropy production rate Π_q.

        Π_q = ε * σ + variance(grad_norm) proxy
        Higher values indicate system is far from equilibrium.
        """
        # Base: product of dissipation and compression
        base_pi = epsilon * sigma

        # Add gradient variance contribution
        if len(self.grad_norm_history) >= 5:
            recent = list(self.grad_norm_history)[-5:]
            mean_g = sum(recent) / len(recent)
            var_g = sum((g - mean_g) ** 2 for g in recent) / len(recent)
            grad_contribution = math.sqrt(var_g) * 0.1
        else:
            grad_contribution = 0.0

        return base_pi + grad_contribution

    def _estimate_jacobian_metrics(
        self,
        grad_norm: float,
        weights_delta_norm: Optional[float],
        second_order: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Estimate Jacobian spectral properties.

        Without explicit Jacobian computation, we use proxies:
        - Spectral norm ≈ grad_norm / delta_w (sensitivity)
        - Trace ≈ sum of curvature info
        - Condition ≈ ratio of max/min eigenvalues
        """
        if second_order and "max_eigenvalue" in second_order:
            spectral_norm = second_order["max_eigenvalue"]
            trace = second_order.get("trace", spectral_norm)
            min_eig = second_order.get("min_eigenvalue", spectral_norm * 0.01)
            condition = spectral_norm / max(1e-8, abs(min_eig))
        else:
            # Proxy estimation
            if weights_delta_norm and weights_delta_norm > 1e-8:
                spectral_norm = grad_norm / weights_delta_norm
            else:
                spectral_norm = grad_norm

            # Use grad history variance as trace proxy
            if len(self.grad_norm_history) >= 3:
                recent = list(self.grad_norm_history)[-3:]
                trace = sum(recent) / len(recent)
            else:
                trace = grad_norm

            condition = 1.0 + abs(spectral_norm - trace)

        return {
            "spectral_norm": min(100.0, spectral_norm),
            "trace": trace,
            "condition": min(1000.0, condition),
        }

    def _estimate_lyapunov(self, grad_norm: float, loss: float) -> float:
        """
        Estimate proxy for maximum Lyapunov exponent.

        Positive → chaotic/unstable
        Negative → stable/convergent
        """
        if len(self.loss_history) < 3:
            return 0.0

        recent_losses = list(self.loss_history)[-10:]

        # Look at rate of divergence in loss trajectory
        if len(recent_losses) >= 2:
            diffs = [recent_losses[i+1] - recent_losses[i]
                    for i in range(len(recent_losses) - 1)]
            avg_diff = sum(diffs) / len(diffs)

            # Normalize by current loss magnitude
            if loss > 1e-8:
                lyapunov = avg_diff / loss
            else:
                lyapunov = avg_diff

            return max(-1.0, min(1.0, lyapunov))

        return 0.0

    def _compute_stability_margin(
        self,
        spectral_norm: float,
        lyapunov: float,
    ) -> float:
        """
        Compute stability margin.

        >0 indicates stable system
        <0 indicates unstable
        """
        # Spectral norm > 1 suggests potential instability
        spectral_factor = max(0.0, 1.0 - spectral_norm * 0.1)

        # Positive Lyapunov suggests instability
        lyapunov_factor = max(0.0, 1.0 - lyapunov * 2.0)

        return spectral_factor * lyapunov_factor

    def _estimate_convexity(self) -> float:
        """
        Estimate payoff convexity (second derivative of performance).

        Positive convexity → antifragile (gains from volatility)
        Zero convexity → robust
        Negative convexity → fragile (harmed by volatility)
        """
        if len(self.loss_history) < self.convexity_window:
            return 0.0

        losses = list(self.loss_history)[-self.convexity_window:]

        # Compute second differences (discrete second derivative)
        if len(losses) < 3:
            return 0.0

        second_diffs = []
        for i in range(1, len(losses) - 1):
            d2 = losses[i+1] - 2*losses[i] + losses[i-1]
            second_diffs.append(d2)

        if not second_diffs:
            return 0.0

        # Negative second derivative of loss = positive convexity of performance
        avg_second_diff = sum(second_diffs) / len(second_diffs)

        # Normalize and negate (we want convexity of performance, not loss)
        mean_loss = sum(losses) / len(losses)
        if mean_loss > 1e-8:
            convexity = -avg_second_diff / mean_loss
        else:
            convexity = -avg_second_diff

        return max(-1.0, min(1.0, convexity))

    def _compute_antifragility_index(
        self,
        convexity: float,
        stability_margin: float,
        pi_q: float,
    ) -> float:
        """
        Compute overall antifragility index.

        High antifragility = high convexity + good stability + moderate Π_q
        """
        # Convexity is the main driver
        index = convexity * 0.5

        # Stability contributes
        index += stability_margin * 0.3

        # Moderate entropy production is good (too low = stagnant, too high = chaos)
        pi_q_sweet_spot = 1.0 - abs(pi_q - 0.3) * 2
        index += max(0, pi_q_sweet_spot) * 0.2

        return max(-1.0, min(1.0, index))

    def _compute_trend(self, values: List[float]) -> float:
        """Compute linear trend of values (-1 to 1)."""
        if len(values) < 3:
            return 0.0

        n = len(values)
        # Simple linear regression slope
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n

        numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
        denominator = sum((i - x_mean) ** 2 for i in range(n))

        if denominator < 1e-8:
            return 0.0

        slope = numerator / denominator

        # Normalize slope to [-1, 1] range
        if y_mean > 1e-8:
            normalized = slope / y_mean
        else:
            normalized = slope

        return max(-1.0, min(1.0, normalized))

    def _check_dau_triggers(
        self,
        step: int,
        pi_q: float,
        stability_margin: float,
        convexity: float,
        antifragility_index: float,
    ) -> Tuple[bool, str, str]:
        """
        Check if DAU should trigger.

        Returns:
            (dau_active, trigger_reason, suggested_action)
        """
        # Cooldown check
        if step - self.last_dau_step < self.dau_cooldown_steps:
            return False, "", ""

        trigger_reason = ""
        action = ""

        # Check instability
        if stability_margin < self.instability_threshold:
            trigger_reason = f"Stability margin low ({stability_margin:.2f})"
            action = "reduce_learning_rate"
            trigger_type = "instability"

        # Check high entropy production
        elif pi_q > self.pi_q_threshold:
            trigger_reason = f"High entropy production Π_q={pi_q:.3f}"
            action = "increase_regularization"
            trigger_type = "stress_test"

        # Check fragility exposure
        elif convexity < -0.3:
            trigger_reason = f"Fragile state (convexity={convexity:.2f})"
            action = "diversify_exploration"
            trigger_type = "fragility"

        # Check opportunity (high antifragility)
        elif antifragility_index > 0.5:
            trigger_reason = f"Antifragile opportunity (index={antifragility_index:.2f})"
            action = "increase_exploration"
            trigger_type = "opportunity"

        else:
            return False, "", ""

        # Record trigger
        self.last_dau_step = step
        trigger = DAUTrigger(
            timestamp=0.0,  # Will be set externally if needed
            step=step,
            trigger_type=trigger_type,
            severity=abs(antifragility_index),
            metrics_snapshot={
                "pi_q": pi_q,
                "stability_margin": stability_margin,
                "convexity": convexity,
                "antifragility_index": antifragility_index,
            },
            action_suggested=action,
        )
        self.dau_triggers.append(trigger)

        return True, trigger_reason, action

    def get_state(self) -> Dict[str, Any]:
        """Get current state for telemetry."""
        m = self._current_metrics
        return {
            "pi_q": m.pi_q,
            "pi_q_mean": m.pi_q_mean,
            "pi_q_trend": m.pi_q_trend,
            "jacobian_spectral_norm": m.jacobian_spectral_norm,
            "jacobian_trace": m.jacobian_trace,
            "jacobian_condition": m.jacobian_condition,
            "lyapunov_proxy": m.lyapunov_proxy,
            "stability_margin": m.stability_margin,
            "convexity": m.convexity,
            "antifragility_index": m.antifragility_index,
            "fragility_exposure": m.fragility_exposure,
            "dau_active": m.dau_active,
            "dau_trigger_reason": m.dau_trigger_reason,
            "dau_action_suggested": m.dau_action_suggested,
        }

    def get_recent_triggers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get N most recent DAU triggers."""
        return [
            {
                "step": t.step,
                "type": t.trigger_type,
                "severity": t.severity,
                "action": t.action_suggested,
                "acted": t.was_acted_upon,
            }
            for t in self.dau_triggers[-n:]
        ]

    @property
    def current_metrics(self) -> AntifragilityMetrics:
        """Get current metrics snapshot."""
        return self._current_metrics


# ============================================================================
#  Factory Function
# ============================================================================

def create_antifragility_monitor(
    history_size: int = 100,
    pi_q_threshold: float = 0.5,
    instability_threshold: float = 0.8,
) -> AntifragilityMonitor:
    """Create an antifragility monitor with given thresholds."""
    return AntifragilityMonitor(
        history_size=history_size,
        pi_q_threshold=pi_q_threshold,
        instability_threshold=instability_threshold,
    )
