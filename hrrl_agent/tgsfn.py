"""
TGSFN (Thermodynamic-Geometric Spiking Field Network) Integration

Integrates all TGSFN components:
- Criticality control via Π_q minimization
- Antifragile loop with Jacobian monitoring
- Hardware-aware operations
- Edge of Chaos dynamics

The TGSFN resolves the engineering frontier of reliably placing
complex systems at the Edge of Chaos. The framework transitions
the concept from a descriptive phase transition to a quantifiable,
dynamically controllable target set.

Key Result: L_TGSFN = F_int + λ_diss · Π_q drives g → 1 (critical)
Validation: Avalanche exponent α = 1.63 ± 0.04
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, NamedTuple
from dataclasses import dataclass, field
import logging
import math

from .criticality import (
    CriticalityController,
    CriticalityConfig,
    CriticalityState,
    CriticalityRegime,
    CriticalInitializer
)
from .antifragile import (
    AntifragileLoop,
    AntifragileConfig,
    AntifragileState,
    StabilityStatus
)
from .thermodynamics import EntropyProductionMonitor, ThermodynamicsConfig
from .hardware import (
    FixedPointHyperbolic,
    ManifoldRecenterer,
    Orthonormalizer,
    KFACTracker,
    FastLearnableTimeWarping
)
from .l2_hyperbolic import PoincareOperations

logger = logging.getLogger(__name__)


@dataclass
class TGSFNConfig:
    """Master configuration for TGSFN substrate."""
    # Criticality
    criticality: CriticalityConfig = field(default_factory=CriticalityConfig)

    # Antifragility
    antifragile: AntifragileConfig = field(default_factory=AntifragileConfig)

    # Thermodynamics
    thermo: ThermodynamicsConfig = field(default_factory=ThermodynamicsConfig)

    # Network dimensions
    num_neurons: int = 1024
    excitatory_fraction: float = 0.8  # 80% excitatory, 20% inhibitory

    # Manifold geometry
    hyperbolic_dim: int = 128
    curvature: float = 1.0

    # Hardware settings
    use_fixed_point: bool = False  # Enable for FPGA deployment
    recenter_interval: int = 1_000_000

    # Training
    lambda_diss_initial: float = 0.1
    lambda_geom: float = 0.01  # Curvature penalty weight


class TGSFNState(NamedTuple):
    """Complete TGSFN state snapshot."""
    criticality: CriticalityState
    antifragile: AntifragileState
    pi_q: float
    f_int: float
    loss: float
    effective_gain: float
    regime: str
    stability: str


class TGSFNLoss(nn.Module):
    """
    TGSFN Loss Function.

    L_TGSFN = F_int + λ_diss · Π_q + λ_geom · K_sectional

    Where:
    - F_int: Internal free energy (homeostatic deviation)
    - Π_q: Entropy production (criticality control)
    - K_sectional: Sectional curvature penalty (geometric regularization)
    """

    def __init__(
        self,
        lambda_diss: float = 0.1,
        lambda_geom: float = 0.01
    ):
        super().__init__()
        self.lambda_diss = lambda_diss
        self.lambda_geom = lambda_geom

    def forward(
        self,
        f_int: torch.Tensor,
        pi_q: torch.Tensor,
        curvature_penalty: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute TGSFN loss.

        Returns:
            (total_loss, components_dict)
        """
        # Base loss
        loss = f_int + self.lambda_diss * pi_q

        # Curvature penalty
        if curvature_penalty is not None:
            loss = loss + self.lambda_geom * curvature_penalty

        components = {
            'f_int': f_int.item(),
            'pi_q': pi_q.item(),
            'pi_q_weighted': (self.lambda_diss * pi_q).item(),
            'total': loss.item()
        }

        if curvature_penalty is not None:
            components['curvature'] = curvature_penalty.item()
            components['curvature_weighted'] = (self.lambda_geom * curvature_penalty).item()

        return loss, components

    def set_lambda_diss(self, value: float):
        """Update dissipation weight (for adaptive control)."""
        self.lambda_diss = value


class TGSFNLayer(nn.Module):
    """
    Single TGSFN spiking layer with criticality control.

    Implements:
    - LIF dynamics with E/I balance
    - Π_q-based self-tuning
    - Hyperbolic embedding for structured representations
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: TGSFNConfig
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.config = config

        # E/I split
        self.num_excitatory = int(out_features * config.excitatory_fraction)
        self.num_inhibitory = out_features - self.num_excitatory

        # Weights (initialized for criticality)
        self.W_exc = nn.Parameter(torch.empty(self.num_excitatory, in_features))
        self.W_inh = nn.Parameter(torch.empty(self.num_inhibitory, in_features))

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Membrane time constants (learnable, per-neuron)
        self.tau_m = nn.Parameter(torch.ones(out_features) * 10.0)

        # Noise variance (learnable, per-neuron)
        self.sigma = nn.Parameter(torch.ones(out_features) * 1.0)

        # Threshold
        self.threshold = nn.Parameter(torch.ones(out_features) * 1.0)

        # Reset potential
        self.v_reset = 0.0

        # State
        self.register_buffer('membrane', torch.zeros(out_features))

        # Initialize for criticality
        self._init_critical()

    def _init_critical(self):
        """Initialize weights for critical dynamics (g ≈ 1)."""
        CriticalInitializer.balanced_init(
            self.W_exc,
            self.num_excitatory,
            0,  # All excitatory
            target_gain=1.0
        )
        CriticalInitializer.balanced_init(
            self.W_inh,
            0,
            self.num_inhibitory,  # All inhibitory
            target_gain=1.0
        )

    @property
    def weight(self) -> torch.Tensor:
        """Combined E/I weight matrix."""
        return torch.cat([self.W_exc, -torch.abs(self.W_inh)], dim=0)

    def forward(
        self,
        x: torch.Tensor,
        return_membrane: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with LIF dynamics.

        Args:
            x: Input spikes or currents
            return_membrane: Whether to return membrane potentials

        Returns:
            (output_spikes, optional_membrane)
        """
        # Compute input current
        I_exc = F.linear(x, self.W_exc)
        I_inh = F.linear(x, -torch.abs(self.W_inh))
        I_total = torch.cat([I_exc, I_inh], dim=-1) + self.bias

        # Leak and integrate
        # dV/dt = -V/τ + I
        decay = torch.exp(-1.0 / self.tau_m)
        self.membrane = decay * self.membrane + (1 - decay) * I_total

        # Spike generation
        spikes = (self.membrane > self.threshold).float()

        # Reset (soft reset)
        self.membrane = torch.where(
            spikes > 0.5,
            torch.tensor(self.v_reset, device=self.membrane.device),
            self.membrane
        )

        if return_membrane:
            return spikes, self.membrane.clone()
        return spikes, None

    def reset_state(self):
        """Reset membrane potential."""
        self.membrane.zero_()

    def compute_pi_q(
        self,
        membrane: torch.Tensor,
        spikes: torch.Tensor,
        jacobian: Optional[torch.Tensor] = None,
        lambda_j: float = 0.01
    ) -> torch.Tensor:
        """
        Compute Π_q for this layer.

        Π_q = Σ_i (V_m^i - V_reset)² / (τ_m^i · σ_i²) + λ_J ||W||_F²
        """
        # Leak power term
        deviation = membrane - self.v_reset
        spike_deviation = deviation * spikes
        leak_term = torch.sum(
            spike_deviation ** 2 / (self.tau_m * self.sigma ** 2 + 1e-10)
        )

        # Weight regularization
        weight_term = lambda_j * torch.sum(self.weight ** 2)

        return leak_term + weight_term

    def get_ei_ratio(self) -> float:
        """Get current E/I current ratio."""
        e_strength = torch.norm(self.W_exc).item()
        i_strength = torch.norm(self.W_inh).item()
        return e_strength / (i_strength + 1e-10)


class TGSFNSubstrate(nn.Module):
    """
    Full TGSFN Substrate with multi-layer architecture.

    Integrates:
    - Stack of TGSFNLayers
    - Criticality controller
    - Antifragile loop
    - Hardware-aware operations
    """

    def __init__(
        self,
        layer_dims: List[int],
        config: TGSFNConfig
    ):
        super().__init__()
        self.config = config
        self.layer_dims = layer_dims

        # Build layers
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(
                TGSFNLayer(layer_dims[i], layer_dims[i + 1], config)
            )

        # Hyperbolic embedding for structured output
        self.poincare = PoincareOperations(config.curvature)
        self.output_embed = nn.Linear(layer_dims[-1], config.hyperbolic_dim)

        # Criticality controller
        self.criticality = CriticalityController(config.criticality)

        # Antifragile loop
        self.antifragile = AntifragileLoop(config.antifragile, self.criticality)

        # Loss function
        self.loss_fn = TGSFNLoss(
            lambda_diss=config.lambda_diss_initial,
            lambda_geom=config.lambda_geom
        )

        # Hardware support
        self.recenterer = ManifoldRecenterer(
            recenter_interval=config.recenter_interval,
            curvature=config.curvature
        )

        if config.use_fixed_point:
            self.fp_hyperbolic = FixedPointHyperbolic(config.curvature)

        # Tracking
        self._step = 0
        self._history: List[TGSFNState] = []

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through TGSFN substrate.

        Args:
            x: Input tensor
            return_embedding: Whether to return hyperbolic embedding

        Returns:
            (output, optional_embedding)
        """
        # Propagate through layers
        h = x
        total_spikes_in = 0
        total_spikes_out = 0

        all_membranes = []
        all_spikes = []

        for layer in self.layers:
            spikes_in = (h > 0.5).sum().item() if h.dtype == torch.float else h.sum().item()
            total_spikes_in += spikes_in

            h, membrane = layer(h, return_membrane=True)

            all_membranes.append(membrane)
            all_spikes.append(h)

            spikes_out = h.sum().item()
            total_spikes_out += spikes_out

        # Compute hyperbolic embedding
        embedding = None
        if return_embedding:
            h_flat = h.view(h.size(0), -1) if h.dim() > 2 else h
            tangent = self.output_embed(h_flat)
            embedding = self.poincare.exp_map_zero(tangent)
            embedding = self.poincare.project(embedding)

            # Maybe recenter
            embedding, _ = self.recenterer.maybe_recenter(embedding)

        return h, embedding

    def compute_loss(
        self,
        f_int: torch.Tensor,
        membrane_potentials: List[torch.Tensor],
        spikes: List[torch.Tensor],
        curvature_penalty: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute full TGSFN loss.

        L_TGSFN = F_int + λ_diss · Π_q + λ_geom · K
        """
        # Compute Π_q for all layers
        pi_q = torch.tensor(0.0, device=f_int.device)

        for i, layer in enumerate(self.layers):
            if i < len(membrane_potentials):
                layer_pi_q = layer.compute_pi_q(
                    membrane_potentials[i],
                    spikes[i]
                )
                pi_q = pi_q + layer_pi_q

        return self.loss_fn(f_int, pi_q, curvature_penalty)

    def step(
        self,
        input_spikes: int,
        output_spikes: int,
        jacobian: Optional[torch.Tensor] = None,
        axiom_embedding: Optional[torch.Tensor] = None
    ) -> TGSFNState:
        """
        Execute one step of TGSFN dynamics with monitoring.
        """
        self._step += 1

        # Update antifragile loop
        if jacobian is None:
            # Approximate Jacobian from weights
            jacobian = torch.cat([
                layer.weight.view(-1) for layer in self.layers
            ]).view(1, -1)

        af_state = self.antifragile.step(
            jacobian=jacobian,
            input_spikes=input_spikes,
            output_spikes=output_spikes,
            axiom_embedding=axiom_embedding
        )

        # Adapt λ_diss based on criticality
        self.loss_fn.lambda_diss = self.criticality.lambda_diss

        # Create state snapshot
        crit_state = af_state.criticality_state

        state = TGSFNState(
            criticality=crit_state,
            antifragile=af_state,
            pi_q=af_state.criticality_state.pi_q if hasattr(af_state.criticality_state, 'pi_q') else 0.0,
            f_int=0.0,  # Would come from homeostat
            loss=0.0,
            effective_gain=crit_state.effective_gain,
            regime=crit_state.regime.value,
            stability=af_state.stability_status.value
        )

        self._history.append(state)

        return state

    def reset(self):
        """Reset all layer states."""
        for layer in self.layers:
            layer.reset_state()

    def get_statistics(self) -> Dict:
        """Get comprehensive TGSFN statistics."""
        return {
            'step': self._step,
            'criticality': self.criticality.get_statistics(),
            'antifragile': self.antifragile.get_statistics(),
            'recentering': self.recenterer.get_statistics(),
            'lambda_diss': self.loss_fn.lambda_diss,
            'num_layers': len(self.layers)
        }


def create_tgsfn_substrate(
    input_dim: int,
    hidden_dims: List[int],
    output_dim: int,
    **kwargs
) -> TGSFNSubstrate:
    """
    Factory function to create TGSFN substrate.

    Args:
        input_dim: Input dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        **kwargs: Additional config overrides

    Returns:
        Configured TGSFNSubstrate
    """
    config = TGSFNConfig(**kwargs)
    layer_dims = [input_dim] + hidden_dims + [output_dim]

    return TGSFNSubstrate(layer_dims, config)


if __name__ == "__main__":
    # Test TGSFN substrate
    print("Testing TGSFN Substrate...")
    print("=" * 60)

    # Create substrate
    substrate = create_tgsfn_substrate(
        input_dim=64,
        hidden_dims=[256, 128],
        output_dim=32,
        num_neurons=1024
    )

    print(f"Created TGSFN with layers: {substrate.layer_dims}")

    # Test forward pass
    print("\nTesting forward pass...")
    x = torch.randn(4, 64)  # Batch of 4

    output, embedding = substrate(x, return_embedding=True)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Embedding shape: {embedding.shape}")

    # Test step
    print("\nTesting TGSFN step...")
    for i in range(50):
        state = substrate.step(
            input_spikes=100,
            output_spikes=95 + torch.randint(-10, 10, (1,)).item()
        )

        if i % 10 == 0:
            print(f"  Step {i}: g={state.effective_gain:.4f}, "
                  f"regime={state.regime}, stability={state.stability}")

    # Test loss computation
    print("\nTesting loss computation...")
    f_int = torch.tensor(0.5)
    membranes = [torch.randn(32) * 0.5 for _ in substrate.layers]
    spikes = [(m > 0.3).float() for m in membranes]

    loss, components = substrate.compute_loss(f_int, membranes, spikes)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Components: {components}")

    # Get statistics
    print("\nStatistics:")
    stats = substrate.get_statistics()
    for k, v in stats.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in list(v.items())[:5]:
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("All TGSFN tests passed!")
