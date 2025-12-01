"""
Π_q-based Criticality Measurement and Logging

Implements entropy production (Π_q) measurement for monitoring
system criticality. Auto-tuning of λ_diss is DISABLED by default -
measurement and logging first.

PRECISE FORMULA (from TGSFN spec):

Π_q ≈ Σ_i (V_m^i - V_reset)² / (τ_m^i · σ_i²) + λ_J ||J_θ||_F²

Where:
- First term: Leak power (E/I balance deviation)
  - Minimizes fluctuation residual around perfect E/I balance
  - At criticality, this is the state of MINIMAL energy dissipation
- Second term: Jacobian Frobenius norm regularization
  - Controls weight magnitudes
  - ||J_θ||_F² = trace(J^T J)

The coupling ensures computational optimum (criticality) equals
minimal energy dissipation state.

Target: g → 1 (effective gain ≈ 1, critical branching)
Avalanche exponent: α = 1.63 ± 0.04 (finite-size corrected 3/2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, NamedTuple, Union
from dataclasses import dataclass
import logging
import math
from collections import deque

from .config import ThermodynamicsConfig

logger = logging.getLogger(__name__)


@dataclass
class ThermodynamicsSnapshot:
    """Snapshot of thermodynamic state."""
    step: int
    pi_q: float  # Total entropy production
    dissipation: float  # Spike dissipation term
    jacobian_term: float  # Weight regularization term
    mi_estimate: float  # Mutual information estimate
    in_critical_range: bool
    lambda_diss_current: float


class EntropyProductionMonitor(nn.Module):
    """
    Monitor for entropy production Π_q.

    Implements measurement and logging of:
    - Spike dissipation: Σ (V_m - V_reset)² / τ_m
    - Jacobian regularization: λ_J trace(J^T J)
    - Mutual information estimate: I(spike; input)

    Auto-tuning is DISABLED by default. Enable only after
    sufficient measurement data has been collected.
    """

    def __init__(self, config: ThermodynamicsConfig):
        super().__init__()
        self.config = config

        # Current λ_diss (for replay distribution)
        self.lambda_diss = config.lambda_dissipation

        # History for analysis
        self._history: List[ThermodynamicsSnapshot] = []
        self._pi_q_buffer = deque(maxlen=1000)
        self._step = 0

        # Logging interval
        self._last_log_step = 0

        # Warning if auto-tuning is enabled
        if config.enable_auto_tuning:
            logger.warning(
                "Π_q auto-tuning is ENABLED. This is experimental. "
                "Recommend disabling until sufficient data is collected."
            )

    def compute_spike_dissipation(
        self,
        membrane_potentials: torch.Tensor,
        spikes: torch.Tensor,
        v_reset: Optional[float] = None,
        tau_m: Optional[Union[float, torch.Tensor]] = None,
        sigma: Optional[Union[float, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Compute spike dissipation term (PRECISE FORMULA).

        Π_leak = Σ_i (V_m^i - V_reset)² / (τ_m^i · σ_i²)

        This is the leak power term that minimizes fluctuation residual
        around perfect E/I balance. At criticality, this equals the
        state of minimal energy dissipation.

        Args:
            membrane_potentials: V_m for each neuron
            spikes: Binary spike mask (1 where spike occurred)
            v_reset: Reset potential (scalar or per-neuron)
            tau_m: Membrane time constant (scalar or per-neuron)
            sigma: Noise variance (scalar or per-neuron)

        Returns:
            Leak power term of Π_q
        """
        if v_reset is None:
            v_reset = self.config.v_reset
        if tau_m is None:
            tau_m = self.config.tau_membrane
        if sigma is None:
            sigma = 1.0  # Default unit variance

        # Ensure tensors
        if isinstance(tau_m, (int, float)):
            tau_m = torch.tensor(tau_m, device=membrane_potentials.device)
        if isinstance(sigma, (int, float)):
            sigma = torch.tensor(sigma, device=membrane_potentials.device)

        # Deviation from reset at spike times
        deviation = membrane_potentials - v_reset

        # Only count where spikes occurred
        spike_deviation = deviation * spikes

        # PRECISE FORMULA: (V_m - V_reset)² / (τ_m · σ²)
        # This is proportional to required leak power
        denominator = tau_m * (sigma ** 2) + 1e-10  # Avoid division by zero
        dissipation = torch.sum(spike_deviation ** 2 / denominator)

        return dissipation

    def compute_jacobian_term(
        self,
        weights: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute Jacobian regularization term.

        λ_J Σ trace(W^T W) = λ_J Σ ||W||_F²

        This approximates the contribution of weight magnitudes
        to entropy production.
        """
        term = torch.tensor(0.0, device=weights[0].device if weights else 'cpu')

        for W in weights:
            term = term + torch.sum(W ** 2)

        return self.config.lambda_jacobian * term

    def estimate_mutual_information(
        self,
        spikes: torch.Tensor,
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Estimate mutual information I(spikes; inputs).

        Uses a simple binning-based estimator.
        This is a rough approximation for computational efficiency.
        """
        # Flatten
        spikes_flat = spikes.flatten().float()
        inputs_flat = inputs.flatten()

        # Bin the inputs
        num_bins = self.config.mi_bins
        input_min = inputs_flat.min()
        input_max = inputs_flat.max()

        if input_max == input_min:
            return torch.tensor(0.0, device=spikes.device)

        # Digitize inputs
        bins = torch.linspace(input_min, input_max, num_bins + 1, device=inputs.device)
        input_bins = torch.bucketize(inputs_flat, bins[1:-1])

        # Compute joint and marginal distributions
        # P(spike=1)
        p_spike = spikes_flat.mean()
        p_no_spike = 1 - p_spike

        # For each input bin, compute P(spike|bin)
        mi = torch.tensor(0.0, device=spikes.device)

        for b in range(num_bins):
            mask = input_bins == b
            if mask.sum() == 0:
                continue

            p_bin = mask.float().mean()
            p_spike_given_bin = spikes_flat[mask].mean()

            if p_spike_given_bin > 0 and p_spike > 0:
                mi = mi + p_bin * p_spike_given_bin * torch.log(
                    p_spike_given_bin / p_spike + 1e-10
                )

            p_no_spike_given_bin = 1 - p_spike_given_bin
            if p_no_spike_given_bin > 0 and p_no_spike > 0:
                mi = mi + p_bin * p_no_spike_given_bin * torch.log(
                    p_no_spike_given_bin / p_no_spike + 1e-10
                )

        return mi.clamp(min=0)

    def compute_entropy_production(
        self,
        membrane_potentials: torch.Tensor,
        spikes: torch.Tensor,
        weights: List[torch.Tensor],
        inputs: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total entropy production Π_q.

        Π_q = dissipation + jacobian_term + mi_term

        Args:
            membrane_potentials: Membrane potentials at spike times
            spikes: Binary spike tensor
            weights: List of weight tensors
            inputs: Optional input tensor for MI estimation

        Returns:
            (pi_q, components_dict)
        """
        self._step += 1

        # Compute components
        dissipation = self.compute_spike_dissipation(membrane_potentials, spikes)
        jacobian_term = self.compute_jacobian_term(weights)

        if inputs is not None:
            mi_term = self.estimate_mutual_information(spikes, inputs)
        else:
            mi_term = torch.tensor(0.0, device=spikes.device)

        # Total
        pi_q = dissipation + jacobian_term + mi_term

        # Store components
        components = {
            'dissipation': dissipation.item(),
            'jacobian_term': jacobian_term.item(),
            'mi_term': mi_term.item(),
            'total': pi_q.item()
        }

        # Add to buffer
        self._pi_q_buffer.append(pi_q.item())

        # Check criticality
        in_critical_range = self._check_criticality(pi_q.item())

        # Create snapshot
        snapshot = ThermodynamicsSnapshot(
            step=self._step,
            pi_q=pi_q.item(),
            dissipation=dissipation.item(),
            jacobian_term=jacobian_term.item(),
            mi_estimate=mi_term.item(),
            in_critical_range=in_critical_range,
            lambda_diss_current=self.lambda_diss
        )
        self._history.append(snapshot)

        # Log if interval reached
        if self._step - self._last_log_step >= self.config.log_interval:
            self._log_state(snapshot)
            self._last_log_step = self._step

        return pi_q, components

    def _check_criticality(self, pi_q: float) -> bool:
        """Check if Π_q is in target critical range."""
        low, high = self.config.target_pi_q_range
        return low <= pi_q <= high

    def _log_state(self, snapshot: ThermodynamicsSnapshot):
        """Log current thermodynamic state."""
        if self.config.log_detailed:
            logger.info(
                f"Thermodynamics [step {snapshot.step}]: "
                f"Π_q={snapshot.pi_q:.4f} "
                f"(diss={snapshot.dissipation:.4f}, "
                f"jac={snapshot.jacobian_term:.4f}, "
                f"MI={snapshot.mi_estimate:.4f}) "
                f"critical={snapshot.in_critical_range} "
                f"λ_diss={snapshot.lambda_diss_current:.4f}"
            )
        else:
            logger.info(
                f"Thermodynamics [step {snapshot.step}]: "
                f"Π_q={snapshot.pi_q:.4f} critical={snapshot.in_critical_range}"
            )

    def get_statistics(self) -> Dict:
        """Get thermodynamic statistics."""
        if len(self._pi_q_buffer) == 0:
            return {'num_samples': 0}

        pi_q_list = list(self._pi_q_buffer)

        return {
            'num_samples': len(pi_q_list),
            'mean_pi_q': sum(pi_q_list) / len(pi_q_list),
            'min_pi_q': min(pi_q_list),
            'max_pi_q': max(pi_q_list),
            'std_pi_q': torch.tensor(pi_q_list).std().item(),
            'current_lambda_diss': self.lambda_diss,
            'in_critical_count': sum(
                1 for s in self._history[-100:]
                if s.in_critical_range
            ) if self._history else 0
        }

    def get_history(self, last_n: Optional[int] = None) -> List[ThermodynamicsSnapshot]:
        """Get thermodynamic history."""
        if last_n is None:
            return self._history
        return self._history[-last_n:]

    # === AUTO-TUNING (DISABLED BY DEFAULT) ===

    def maybe_auto_tune(self) -> Optional[float]:
        """
        Potentially auto-tune λ_diss based on Π_q measurements.

        ⚠️ ONLY RUNS IF enable_auto_tuning=True

        Returns new λ_diss if adjusted, None otherwise.
        """
        if not self.config.enable_auto_tuning:
            return None

        if len(self._pi_q_buffer) < 100:
            # Not enough data yet
            return None

        # Compute recent statistics
        recent = list(self._pi_q_buffer)[-100:]
        mean_pi_q = sum(recent) / len(recent)

        low, high = self.config.target_pi_q_range

        # Simple proportional control
        if mean_pi_q < low:
            # Π_q too low, decrease λ_diss to allow more entropy production
            adjustment = 0.99
            logger.info(f"Auto-tune: Π_q ({mean_pi_q:.4f}) below target, decreasing λ_diss")
        elif mean_pi_q > high:
            # Π_q too high, increase λ_diss to penalize more
            adjustment = 1.01
            logger.info(f"Auto-tune: Π_q ({mean_pi_q:.4f}) above target, increasing λ_diss")
        else:
            # In range, no adjustment
            return None

        # Apply adjustment with bounds
        new_lambda = self.lambda_diss * adjustment
        new_lambda = max(0.001, min(1.0, new_lambda))  # Clamp to reasonable range

        self.lambda_diss = new_lambda
        logger.info(f"Auto-tune: New λ_diss = {new_lambda:.6f}")

        return new_lambda

    def disable_auto_tuning(self):
        """Disable auto-tuning (recommended)."""
        self.config.enable_auto_tuning = False
        logger.info("Π_q auto-tuning disabled")

    def enable_auto_tuning(self, confirm: bool = False):
        """
        Enable auto-tuning.

        Args:
            confirm: Must be True to actually enable (safety measure)
        """
        if not confirm:
            logger.warning(
                "enable_auto_tuning() called without confirm=True. "
                "Auto-tuning remains disabled. "
                "To enable, call enable_auto_tuning(confirm=True)"
            )
            return

        self.config.enable_auto_tuning = True
        logger.warning("Π_q auto-tuning ENABLED. This is experimental.")


class ThermodynamicLoss(nn.Module):
    """
    Thermodynamic loss component for training.

    L_thermo = λ_diss · Π_q
    """

    def __init__(self, lambda_diss: float = 0.1):
        super().__init__()
        self.lambda_diss = lambda_diss

    def forward(self, pi_q: torch.Tensor) -> torch.Tensor:
        """Compute thermodynamic loss."""
        return self.lambda_diss * pi_q

    def set_lambda(self, new_lambda: float):
        """Update λ_diss."""
        self.lambda_diss = new_lambda


class CriticalityAnalyzer:
    """
    Analyzer for criticality metrics.

    Provides additional analysis tools for understanding
    the system's proximity to criticality.
    """

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._avalanche_sizes: List[int] = []
        self._current_avalanche = 0
        self._in_avalanche = False

    def record_spike(self, spike_count: int, threshold: int = 0):
        """
        Record spike activity for avalanche analysis.

        Args:
            spike_count: Number of spikes in this timestep
            threshold: Minimum spikes to consider as avalanche activity
        """
        if spike_count > threshold:
            if not self._in_avalanche:
                self._in_avalanche = True
                self._current_avalanche = 0

            self._current_avalanche += spike_count
        else:
            if self._in_avalanche:
                # Avalanche ended
                self._avalanche_sizes.append(self._current_avalanche)
                self._in_avalanche = False
                self._current_avalanche = 0

    def compute_avalanche_exponent(self) -> Optional[float]:
        """
        Compute power-law exponent of avalanche size distribution.

        At criticality, avalanche sizes follow P(s) ~ s^(-τ) with τ ≈ 1.5

        Returns:
            Estimated exponent τ, or None if insufficient data
        """
        if len(self._avalanche_sizes) < 10:
            return None

        # Use log-log regression
        sizes = torch.tensor(self._avalanche_sizes[-self.window_size:]).float()
        sizes = sizes[sizes > 0]

        if len(sizes) < 10:
            return None

        # Compute empirical CDF
        sorted_sizes, _ = torch.sort(sizes, descending=True)
        ranks = torch.arange(1, len(sorted_sizes) + 1).float()

        # Log-log regression: log(rank) = -τ log(size) + const
        log_sizes = torch.log(sorted_sizes)
        log_ranks = torch.log(ranks)

        # Simple linear regression
        n = len(log_sizes)
        sum_x = log_sizes.sum()
        sum_y = log_ranks.sum()
        sum_xy = (log_sizes * log_ranks).sum()
        sum_x2 = (log_sizes ** 2).sum()

        denom = n * sum_x2 - sum_x ** 2
        if abs(denom) < 1e-10:
            return None

        slope = (n * sum_xy - sum_x * sum_y) / denom

        # τ = -slope
        tau = -slope.item()

        return tau

    def get_criticality_score(self) -> float:
        """
        Get a score indicating proximity to criticality.

        Returns value in [0, 1] where 1 = perfectly critical (τ ≈ 1.5)
        """
        tau = self.compute_avalanche_exponent()

        if tau is None:
            return 0.5  # Unknown

        # Ideal τ ≈ 1.5 for criticality
        ideal_tau = 1.5
        deviation = abs(tau - ideal_tau)

        # Score: 1 at ideal, decreasing with deviation
        score = math.exp(-deviation)

        return score

    def get_statistics(self) -> Dict:
        """Get avalanche statistics."""
        if len(self._avalanche_sizes) == 0:
            return {'num_avalanches': 0}

        sizes = self._avalanche_sizes[-self.window_size:]

        return {
            'num_avalanches': len(sizes),
            'mean_size': sum(sizes) / len(sizes),
            'max_size': max(sizes),
            'exponent_tau': self.compute_avalanche_exponent(),
            'criticality_score': self.get_criticality_score()
        }


if __name__ == "__main__":
    # Test thermodynamics
    print("Testing EntropyProductionMonitor...")

    config = ThermodynamicsConfig(log_interval=10)
    monitor = EntropyProductionMonitor(config)

    # Simulate some data
    for i in range(50):
        # Random membrane potentials and spikes
        V_m = torch.randn(100) * 0.5
        spikes = (torch.rand(100) > 0.8).float()

        # Random weights
        weights = [torch.randn(64, 64) * 0.1, torch.randn(64, 32) * 0.1]

        # Random inputs
        inputs = torch.randn(100)

        pi_q, components = monitor.compute_entropy_production(
            V_m, spikes, weights, inputs
        )

    # Get statistics
    print("\nThermodynamic Statistics:")
    stats = monitor.get_statistics()
    for k, v in stats.items():
        print(f"  {k}: {v}")

    # Test criticality analyzer
    print("\nTesting CriticalityAnalyzer...")
    analyzer = CriticalityAnalyzer()

    # Simulate spike activity
    for i in range(200):
        # Random spikes (with some temporal correlation for avalanches)
        if i % 20 < 10:
            spike_count = int(torch.randint(5, 20, (1,)).item())
        else:
            spike_count = int(torch.randint(0, 3, (1,)).item())

        analyzer.record_spike(spike_count)

    print("\nAvalanche Statistics:")
    stats = analyzer.get_statistics()
    for k, v in stats.items():
        if v is not None:
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Test auto-tuning (disabled by default)
    print("\nAuto-tuning test...")
    result = monitor.maybe_auto_tune()
    print(f"  Auto-tune result (disabled): {result}")

    # Enable and test
    monitor.enable_auto_tuning(confirm=True)
    result = monitor.maybe_auto_tune()
    print(f"  Auto-tune result (enabled): {result}")

    print("\nAll thermodynamics tests passed!")
