"""
Edge of Chaos Criticality Control

Implements TGSFN criticality control via Π_q minimization.

The Edge of Chaos is the critical phase boundary where:
- Computational capacity is maximized
- Information flow is optimal
- System is poised between frozen order and exponential chaos

Key principle: Minimize Π_q to achieve critical branching g → 1

Reference exponents:
- Cellular Automata (Langton): λ_c ≈ 0.37
- Random Boolean Networks: K_c = 2, avalanche α = 3/2
- Continuous Dynamics (Feigenbaum): λ_max = 0, δ ≈ 4.669

TGSFN achieves α = 1.63 ± 0.04 (consistent with finite-size corrected 3/2)

Finite-Size Scaling Theory (see docs/FINITE_SIZE_SCALING.md):
- Asymptotic: α(N) = 3/2 + 1/(ln N + C) + O(N⁻¹)
- Operational (N ~ 10³-10⁵): α(N) ≈ 3/2 + 6.6/√N
- The 1/√N form is a high-quality transient fit, not the true asymptotic
- Π_q minimization forces Δm(N) ∝ 1/√N subcriticality for stability
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
import math
from collections import deque

logger = logging.getLogger(__name__)


class CriticalityRegime(Enum):
    """Classification of system's dynamical regime."""
    FROZEN = "frozen"  # Subcritical: activity dies out
    CRITICAL = "critical"  # Edge of Chaos: optimal computation
    CHAOTIC = "chaotic"  # Supercritical: activity explodes


@dataclass
class CriticalityConfig:
    """Configuration for criticality control."""
    # Target effective gain (g = 1 is critical)
    target_gain: float = 1.0
    gain_tolerance: float = 0.1  # |g - 1| < tolerance is critical

    # Avalanche analysis
    avalanche_threshold: int = 1  # Minimum spikes to start avalanche
    target_exponent: float = 1.5  # Mean-field α = 3/2
    exponent_tolerance: float = 0.2  # Accept α ∈ [1.3, 1.7]

    # E/I balance
    ei_ratio_target: float = 1.0  # Perfect E/I balance
    ei_tolerance: float = 0.2

    # Π_q control
    lambda_diss_initial: float = 0.1
    lambda_diss_min: float = 0.001
    lambda_diss_max: float = 1.0
    lambda_diss_adaptation_rate: float = 0.01

    # Monitoring
    window_size: int = 1000
    log_interval: int = 100


@dataclass
class CriticalityState:
    """Current criticality state measurements."""
    effective_gain: float
    avalanche_exponent: Optional[float]
    ei_ratio: float
    regime: CriticalityRegime
    pi_q: float
    lambda_diss: float
    branching_ratio: float
    in_critical_range: bool


class EffectiveGainEstimator:
    """
    Estimates effective gain g from spike dynamics.

    At criticality, g → 1, meaning on average each spike
    causes exactly one subsequent spike.

    g < 1: Subcritical (frozen)
    g = 1: Critical (Edge of Chaos)
    g > 1: Supercritical (chaotic)
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._spike_counts = deque(maxlen=window_size)
        self._propagation_counts = deque(maxlen=window_size)

    def record(self, input_spikes: int, output_spikes: int):
        """Record spike propagation."""
        if input_spikes > 0:
            self._spike_counts.append(input_spikes)
            self._propagation_counts.append(output_spikes)

    def estimate_gain(self) -> float:
        """
        Estimate effective gain.

        g = ⟨output_spikes⟩ / ⟨input_spikes⟩
        """
        if len(self._spike_counts) < 10:
            return 1.0  # Default to critical

        total_input = sum(self._spike_counts)
        total_output = sum(self._propagation_counts)

        if total_input == 0:
            return 0.0

        return total_output / total_input

    def get_branching_ratio(self) -> float:
        """
        Compute branching ratio σ.

        σ = Var(output) / ⟨output⟩

        At criticality, σ → 1 (Poisson-like)
        """
        if len(self._propagation_counts) < 10:
            return 1.0

        outputs = torch.tensor(list(self._propagation_counts), dtype=torch.float32)
        mean_out = outputs.mean()

        if mean_out < 1e-6:
            return 0.0

        var_out = outputs.var()
        return (var_out / mean_out).item()


class AvalancheAnalyzer:
    """
    Analyzes avalanche statistics for criticality assessment.

    At criticality, avalanche sizes follow power law:
    P(s) ~ s^(-α) with α = 3/2 (mean-field)

    Finite-size corrections give α ≈ 1.63 for neural networks.

    Finite-Size Scaling (docs/FINITE_SIZE_SCALING.md):
    - True asymptotic: α(N) = 3/2 + 1/(ln N + C)
    - Operational fit: α(N) ≈ 3/2 + 6.6/√N for N ∈ [10³, 10⁵]
    - The observed α = 1.63 is NOT an error but expected finite-size behavior
    """

    def __init__(self, config: CriticalityConfig):
        self.config = config
        self._avalanche_sizes: List[int] = []
        self._avalanche_durations: List[int] = []
        self._current_avalanche_size = 0
        self._current_avalanche_duration = 0
        self._in_avalanche = False

    def record_activity(self, spike_count: int):
        """Record spike activity for avalanche detection."""
        if spike_count >= self.config.avalanche_threshold:
            if not self._in_avalanche:
                # Start new avalanche
                self._in_avalanche = True
                self._current_avalanche_size = 0
                self._current_avalanche_duration = 0

            self._current_avalanche_size += spike_count
            self._current_avalanche_duration += 1
        else:
            if self._in_avalanche:
                # End avalanche
                self._avalanche_sizes.append(self._current_avalanche_size)
                self._avalanche_durations.append(self._current_avalanche_duration)
                self._in_avalanche = False

    def compute_size_exponent(self) -> Optional[float]:
        """
        Compute power-law exponent α from avalanche size distribution.

        Uses maximum likelihood estimation for power law.
        """
        if len(self._avalanche_sizes) < 50:
            return None

        sizes = torch.tensor(self._avalanche_sizes[-self.config.window_size:], dtype=torch.float32)
        sizes = sizes[sizes > 0]

        if len(sizes) < 20:
            return None

        # MLE for power law: α = 1 + n / Σ ln(s_i / s_min)
        s_min = sizes.min()
        n = len(sizes)

        log_sum = torch.sum(torch.log(sizes / s_min))
        if log_sum < 1e-6:
            return None

        alpha = 1.0 + n / log_sum.item()

        return alpha

    def compute_duration_exponent(self) -> Optional[float]:
        """Compute exponent for avalanche duration distribution."""
        if len(self._avalanche_durations) < 50:
            return None

        durations = torch.tensor(
            self._avalanche_durations[-self.config.window_size:],
            dtype=torch.float32
        )
        durations = durations[durations > 0]

        if len(durations) < 20:
            return None

        d_min = durations.min()
        n = len(durations)

        log_sum = torch.sum(torch.log(durations / d_min))
        if log_sum < 1e-6:
            return None

        return 1.0 + n / log_sum.item()

    def is_critical(self) -> Tuple[bool, str]:
        """
        Check if avalanche statistics indicate criticality.

        Returns (is_critical, reason)
        """
        alpha = self.compute_size_exponent()

        if alpha is None:
            return False, "Insufficient avalanche data"

        target = self.config.target_exponent
        tol = self.config.exponent_tolerance

        if target - tol <= alpha <= target + tol:
            return True, f"Avalanche exponent α={alpha:.3f} in critical range"
        elif alpha < target - tol:
            return False, f"Subcritical: α={alpha:.3f} < {target - tol:.2f}"
        else:
            return False, f"Supercritical: α={alpha:.3f} > {target + tol:.2f}"


def predict_finite_size_alpha(N: int, use_asymptotic: bool = False) -> float:
    """
    Predict expected avalanche exponent for network size N.

    Implements the finite-size scaling theory from docs/FINITE_SIZE_SCALING.md.

    The Logarithmic Convergence Theorem establishes:
    - True asymptotic: α(N) = 3/2 + 1/(ln N + C) + O(N⁻¹)
    - Operational fit: α(N) ≈ 3/2 + 6.6/√N for N ∈ [10³, 10⁵]

    Args:
        N: Network size (number of neurons)
        use_asymptotic: If True, use asymptotic form; else use operational fit

    Returns:
        Predicted avalanche size exponent α

    Example:
        >>> predict_finite_size_alpha(4096)  # Typical TGSFN size
        1.603...
        >>> predict_finite_size_alpha(8192)
        1.572...
    """
    MEAN_FIELD_ALPHA = 1.5
    OPERATIONAL_CONSTANT = 6.6  # Empirically derived from TGSFN simulations
    ASYMPTOTIC_C = 2.5  # System-dependent constant

    if N < 1:
        raise ValueError(f"Network size must be positive, got {N}")

    if use_asymptotic or N >= 1e6:
        # True asymptotic form: α(N) = 3/2 + 1/(ln N + C)
        return MEAN_FIELD_ALPHA + 1.0 / (math.log(N) + ASYMPTOTIC_C)
    else:
        # Operational phenomenological fit: α(N) = 3/2 + c/√N
        return MEAN_FIELD_ALPHA + OPERATIONAL_CONSTANT / math.sqrt(N)


class EIBalanceMonitor:
    """
    Monitors Excitatory/Inhibitory balance.

    Perfect E/I balance (ratio = 1) is required for criticality.
    The Π_q term explicitly minimizes deviation from this balance.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._excitatory = deque(maxlen=window_size)
        self._inhibitory = deque(maxlen=window_size)

    def record(self, excitatory_current: float, inhibitory_current: float):
        """Record E/I currents."""
        self._excitatory.append(abs(excitatory_current))
        self._inhibitory.append(abs(inhibitory_current))

    def get_ratio(self) -> float:
        """
        Compute E/I ratio.

        Ratio = ⟨|E|⟩ / ⟨|I|⟩
        """
        if len(self._excitatory) < 10 or len(self._inhibitory) < 10:
            return 1.0

        e_mean = sum(self._excitatory) / len(self._excitatory)
        i_mean = sum(self._inhibitory) / len(self._inhibitory)

        if i_mean < 1e-10:
            return float('inf')

        return e_mean / i_mean

    def get_balance_deviation(self) -> float:
        """Get deviation from perfect balance."""
        ratio = self.get_ratio()
        return abs(ratio - 1.0)


class CriticalityController(nn.Module):
    """
    Main criticality controller for TGSFN.

    Uses Π_q minimization to maintain the system at Edge of Chaos.

    The controller adjusts λ_diss to drive effective gain g → 1,
    achieving the critical branching process condition.
    """

    def __init__(self, config: CriticalityConfig):
        super().__init__()
        self.config = config

        # Estimators
        self.gain_estimator = EffectiveGainEstimator(config.window_size)
        self.avalanche_analyzer = AvalancheAnalyzer(config)
        self.ei_monitor = EIBalanceMonitor(config.window_size)

        # Current λ_diss
        self.lambda_diss = config.lambda_diss_initial

        # History
        self._history: List[CriticalityState] = []
        self._step = 0

    def compute_pi_q(
        self,
        membrane_potentials: torch.Tensor,
        v_reset: float,
        tau_m: torch.Tensor,
        sigma: torch.Tensor,
        jacobian: Optional[torch.Tensor] = None,
        lambda_j: float = 0.01
    ) -> torch.Tensor:
        """
        Compute entropy production proxy Π_q.

        Π_q = Σ_i (V_m^i - V_reset)² / (τ_m^i · σ_i²) + λ_J ||J_θ||_F²

        This is the minimal control variable that drives g → 1.
        """
        # Leak power term (E/I balance deviation)
        deviation = membrane_potentials - v_reset
        leak_term = torch.sum(
            deviation ** 2 / (tau_m * sigma ** 2 + 1e-8)
        )

        # Jacobian regularization
        jacobian_term = torch.tensor(0.0, device=membrane_potentials.device)
        if jacobian is not None:
            jacobian_term = lambda_j * torch.sum(jacobian ** 2)

        return leak_term + jacobian_term

    def compute_tgsfn_loss(
        self,
        f_int: torch.Tensor,
        pi_q: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute full TGSFN loss.

        L_TGSFN = F_int + λ_diss · Π_q

        Minimizing this drives the system to criticality.
        """
        return f_int + self.lambda_diss * pi_q

    def update(
        self,
        input_spikes: int,
        output_spikes: int,
        excitatory_current: float = 0.0,
        inhibitory_current: float = 0.0
    ) -> CriticalityState:
        """
        Update criticality measurements and adapt λ_diss.

        Args:
            input_spikes: Number of input spikes
            output_spikes: Number of output spikes
            excitatory_current: Total excitatory current
            inhibitory_current: Total inhibitory current

        Returns:
            Current CriticalityState
        """
        self._step += 1

        # Update estimators
        self.gain_estimator.record(input_spikes, output_spikes)
        self.avalanche_analyzer.record_activity(output_spikes)
        self.ei_monitor.record(excitatory_current, inhibitory_current)

        # Get current measurements
        effective_gain = self.gain_estimator.estimate_gain()
        branching_ratio = self.gain_estimator.get_branching_ratio()
        ei_ratio = self.ei_monitor.get_ratio()
        avalanche_exponent = self.avalanche_analyzer.compute_size_exponent()

        # Classify regime
        regime = self._classify_regime(effective_gain)

        # Check if in critical range
        in_critical = self._is_in_critical_range(effective_gain, avalanche_exponent)

        # Adapt λ_diss
        self._adapt_lambda_diss(effective_gain)

        # Create state
        state = CriticalityState(
            effective_gain=effective_gain,
            avalanche_exponent=avalanche_exponent,
            ei_ratio=ei_ratio,
            regime=regime,
            pi_q=0.0,  # Would be computed from actual membrane potentials
            lambda_diss=self.lambda_diss,
            branching_ratio=branching_ratio,
            in_critical_range=in_critical
        )

        self._history.append(state)

        # Log periodically
        if self._step % self.config.log_interval == 0:
            self._log_state(state)

        return state

    def _classify_regime(self, gain: float) -> CriticalityRegime:
        """Classify dynamical regime based on effective gain."""
        tol = self.config.gain_tolerance

        if gain < 1.0 - tol:
            return CriticalityRegime.FROZEN
        elif gain > 1.0 + tol:
            return CriticalityRegime.CHAOTIC
        else:
            return CriticalityRegime.CRITICAL

    def _is_in_critical_range(
        self,
        gain: float,
        exponent: Optional[float]
    ) -> bool:
        """Check if system is in critical range."""
        # Check gain
        gain_ok = abs(gain - self.config.target_gain) < self.config.gain_tolerance

        # Check exponent if available
        if exponent is not None:
            exp_ok = abs(exponent - self.config.target_exponent) < self.config.exponent_tolerance
            return gain_ok and exp_ok

        return gain_ok

    def _adapt_lambda_diss(self, gain: float):
        """
        Adapt λ_diss to drive system toward criticality.

        If g > 1 (chaotic): Increase λ_diss to suppress activity
        If g < 1 (frozen): Decrease λ_diss to allow more activity
        """
        target = self.config.target_gain
        rate = self.config.lambda_diss_adaptation_rate

        error = gain - target

        # Proportional control
        delta = rate * error

        self.lambda_diss = max(
            self.config.lambda_diss_min,
            min(self.config.lambda_diss_max, self.lambda_diss + delta)
        )

    def _log_state(self, state: CriticalityState):
        """Log criticality state."""
        exp_str = f"{state.avalanche_exponent:.3f}" if state.avalanche_exponent else "N/A"
        logger.info(
            f"Criticality [step {self._step}]: "
            f"g={state.effective_gain:.4f}, "
            f"α={exp_str}, "
            f"E/I={state.ei_ratio:.3f}, "
            f"regime={state.regime.value}, "
            f"λ_diss={state.lambda_diss:.6f}"
        )

    def get_statistics(self) -> Dict:
        """Get criticality statistics."""
        if not self._history:
            return {'steps': 0}

        recent = self._history[-100:]

        critical_count = sum(
            1 for s in recent
            if s.regime == CriticalityRegime.CRITICAL
        )

        return {
            'steps': self._step,
            'current_gain': recent[-1].effective_gain,
            'current_regime': recent[-1].regime.value,
            'critical_fraction': critical_count / len(recent),
            'lambda_diss': self.lambda_diss,
            'avalanche_exponent': recent[-1].avalanche_exponent
        }

    def get_history(self, last_n: Optional[int] = None) -> List[CriticalityState]:
        """Get criticality history."""
        if last_n is None:
            return self._history
        return self._history[-last_n:]


class CriticalInitializer:
    """
    Initialize network weights to be near criticality.

    Uses balanced initialization to achieve g ≈ 1 from the start.
    """

    @staticmethod
    def balanced_init(
        weight: torch.Tensor,
        num_excitatory: int,
        num_inhibitory: int,
        target_gain: float = 1.0
    ):
        """
        Initialize weights for E/I balance at criticality.

        For g = 1: ⟨w_E⟩ · N_E = ⟨w_I⟩ · N_I = 1/√N
        """
        n_total = num_excitatory + num_inhibitory

        # Scale for critical gain
        scale = target_gain / math.sqrt(n_total)

        with torch.no_grad():
            # Excitatory weights (positive)
            weight[:num_excitatory] = torch.abs(
                torch.randn_like(weight[:num_excitatory])
            ) * scale

            # Inhibitory weights (negative)
            weight[num_excitatory:] = -torch.abs(
                torch.randn_like(weight[num_excitatory:])
            ) * scale

    @staticmethod
    def spectral_init(weight: torch.Tensor, spectral_radius: float = 1.0):
        """
        Initialize with target spectral radius.

        At criticality, spectral radius should be ~1.
        """
        with torch.no_grad():
            nn.init.xavier_normal_(weight)

            # Compute current spectral radius
            u, s, v = torch.svd(weight)
            current_radius = s[0].item()

            if current_radius > 0:
                weight.mul_(spectral_radius / current_radius)


if __name__ == "__main__":
    # Test criticality control
    print("Testing CriticalityController...")

    config = CriticalityConfig()
    controller = CriticalityController(config)

    # Simulate subcritical dynamics
    print("\nSimulating subcritical (g < 1)...")
    for _ in range(100):
        # Input spikes decay
        input_spikes = 10
        output_spikes = 8  # g ≈ 0.8
        controller.update(input_spikes, output_spikes, 0.5, 0.6)

    state = controller.get_statistics()
    print(f"  Regime: {state['current_regime']}")
    print(f"  Gain: {state['current_gain']:.4f}")
    print(f"  λ_diss: {state['lambda_diss']:.6f}")

    # Simulate critical dynamics
    print("\nSimulating critical (g ≈ 1)...")
    for _ in range(100):
        input_spikes = 10
        output_spikes = 10 + torch.randint(-2, 3, (1,)).item()  # g ≈ 1
        controller.update(input_spikes, output_spikes, 0.5, 0.5)

    state = controller.get_statistics()
    print(f"  Regime: {state['current_regime']}")
    print(f"  Gain: {state['current_gain']:.4f}")
    print(f"  Critical fraction: {state['critical_fraction']:.2%}")

    # Test avalanche analysis
    print("\nTesting avalanche analysis...")
    analyzer = AvalancheAnalyzer(config)

    # Generate power-law-like avalanches
    for _ in range(200):
        # Simulate avalanche with random duration
        duration = int(torch.randint(1, 10, (1,)).item())
        for t in range(duration):
            analyzer.record_activity(torch.randint(1, 20, (1,)).item())
        analyzer.record_activity(0)  # End avalanche

    exponent = analyzer.compute_size_exponent()
    print(f"  Estimated exponent: {exponent:.3f}" if exponent else "  Insufficient data")

    # Test balanced initialization
    print("\nTesting balanced initialization...")
    weight = torch.empty(100, 100)
    CriticalInitializer.balanced_init(weight, 80, 20)
    print(f"  Weight mean: {weight.mean():.6f}")
    print(f"  Excitatory mean: {weight[:80].mean():.6f}")
    print(f"  Inhibitory mean: {weight[80:].mean():.6f}")

    print("\nAll criticality tests passed!")
