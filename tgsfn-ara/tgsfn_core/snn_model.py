# tgsfn_core/snn_model.py
# TGSFN Spiking Neural Network Model
#
# Implements:
#   - LIFLayer: Leaky Integrate-and-Fire neurons with E/I connectivity
#   - TGSFNNetwork: Complete recurrent SNN with thermodynamic regularization hooks
#
# Key features:
#   - E:I ratio of ~4:1 for biological realism
#   - Sparse random connectivity
#   - Branching ratio control for criticality
#   - Jacobian proxy computation for stability monitoring

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class LIFLayer(nn.Module):
    """
    Leaky Integrate-and-Fire layer with E/I connectivity.

    Implements the LIF dynamics:
        τ_m dV/dt = -(V - V_rest) + R_m * I_syn + I_ext

    When V >= V_th: emit spike, reset to V_reset

    Attributes:
        N: Total number of neurons
        N_exc: Number of excitatory neurons
        N_inh: Number of inhibitory neurons
        tau_m: Membrane time constant
        v_reset: Reset potential
        v_th: Threshold potential
        W: Weight matrix (learnable)
    """

    def __init__(
        self,
        N: int,
        p_connect: float = 0.05,
        frac_inh: float = 0.2,
        tau_m: float = 20e-3,
        v_reset: float = 0.0,
        v_th: float = 1.0,
        dt: float = 1e-3,
        w_scale: float = 0.1,
        initial_branching: float = 1.1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize LIF layer.

        Args:
            N: Number of neurons
            p_connect: Connection probability
            frac_inh: Fraction of inhibitory neurons (default 0.2 for 4:1 E:I)
            tau_m: Membrane time constant in seconds
            v_reset: Reset potential after spike
            v_th: Spike threshold
            dt: Simulation timestep in seconds
            w_scale: Initial weight scale
            initial_branching: Target initial branching ratio m₀
            device: Torch device
        """
        super().__init__()

        self.N = N
        self.frac_inh = frac_inh
        self.N_inh = int(frac_inh * N)
        self.N_exc = N - self.N_inh

        self.tau_m = tau_m
        self.v_reset = v_reset
        self.v_th = v_th
        self.dt = dt
        self.initial_branching = initial_branching

        # E/I masks
        self.register_buffer("E_mask", torch.zeros(N, dtype=torch.bool))
        self.register_buffer("I_mask", torch.zeros(N, dtype=torch.bool))
        self.E_mask[:self.N_exc] = True
        self.I_mask[self.N_exc:] = True

        # Initialize weights with sparse random connectivity
        W = self._init_weights(p_connect, w_scale)
        self.W = nn.Parameter(W)

        # Learnable threshold (optional)
        self.register_buffer("v_th_base", torch.tensor(v_th))

        # State buffers (set during reset_state)
        self._V: Optional[torch.Tensor] = None
        self._spikes: Optional[torch.Tensor] = None
        self._refractory: Optional[torch.Tensor] = None

        self.device = device or torch.device("cpu")
        self.to(self.device)

    def _init_weights(self, p_connect: float, w_scale: float) -> torch.Tensor:
        """Initialize sparse E/I weight matrix."""
        W = torch.zeros(self.N, self.N)

        # Random sparse connectivity
        mask = torch.rand(self.N, self.N) < p_connect
        # No self-connections
        mask.fill_diagonal_(False)

        n_connections = mask.sum().item()
        W[mask] = torch.randn(n_connections) * w_scale

        # Make inhibitory projections negative
        # Rows correspond to presynaptic neurons
        W[self.N_exc:, :] = -torch.abs(W[self.N_exc:, :])
        # Excitatory projections positive
        W[:self.N_exc, :] = torch.abs(W[:self.N_exc, :])

        # Scale to target branching ratio
        # m ≈ mean(|W|) * p_connect * N * dt / tau_m
        current_scale = W.abs().mean() * p_connect * self.N * self.dt / self.tau_m
        if current_scale > 0:
            W = W * (self.initial_branching / current_scale)

        return W

    def reset_state(self, batch_size: int) -> None:
        """Reset membrane potentials and spike history."""
        self._V = torch.zeros(batch_size, self.N, device=self.device)
        self._spikes = torch.zeros(batch_size, self.N, device=self.device)
        self._refractory = torch.zeros(batch_size, self.N, device=self.device)

    def forward(
        self,
        I_ext: torch.Tensor,
        return_voltage: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: one timestep of LIF dynamics.

        Args:
            I_ext: External input current [batch, N]
            return_voltage: Whether to return membrane potentials

        Returns:
            spikes: Binary spike tensor [batch, N]
            V: Membrane potentials [batch, N] (if return_voltage=True)
        """
        if self._V is None:
            batch_size = I_ext.shape[0]
            self.reset_state(batch_size)

        # Recurrent input from previous spikes
        # W is [N_pre, N_post], spikes is [batch, N_pre]
        # I_rec = spikes @ W gives [batch, N_post]
        I_rec = torch.matmul(self._spikes, self.W)

        # Total synaptic current
        I_total = I_rec + I_ext

        # LIF membrane update
        # dV/dt = (-V + I_total) / tau_m
        alpha = self.dt / self.tau_m
        V_new = self._V * (1 - alpha) + I_total * alpha

        # Spike generation
        spikes = (V_new >= self.v_th).float()

        # Reset spiking neurons
        V_new = torch.where(
            spikes > 0,
            torch.full_like(V_new, self.v_reset),
            V_new
        )

        # Update state
        self._V = V_new
        self._spikes = spikes

        if return_voltage:
            return spikes, V_new
        return spikes, None

    def get_state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get current membrane potential and spikes."""
        return self._V, self._spikes

    def compute_jacobian_proxy(self) -> torch.Tensor:
        """
        Compute a proxy for ||J||_F² (Jacobian Frobenius norm squared).

        For LIF networks, the Jacobian of the dynamics depends on:
        - Weight matrix W
        - Current firing rates (which neurons are near threshold)

        Simple proxy: ||W||_F² scaled by activity
        """
        W_norm_sq = (self.W ** 2).sum()

        # Scale by fraction of neurons near threshold
        if self._V is not None:
            # Neurons within 20% of threshold are "active"
            near_threshold = (self._V > 0.8 * self.v_th).float().mean()
            activity_scale = 0.5 + near_threshold  # Range [0.5, 1.5]
        else:
            activity_scale = 1.0

        return W_norm_sq * activity_scale

    def get_effective_branching(self) -> torch.Tensor:
        """Estimate current effective branching ratio."""
        # m ≈ mean absolute weight * connectivity * N * dt / tau
        W_abs_mean = self.W.abs().mean()
        connectivity = (self.W.abs() > 1e-6).float().mean()
        m = W_abs_mean * connectivity * self.N * self.dt / self.tau_m
        return m


class TGSFNNetwork(nn.Module):
    """
    Complete TGSFN Network with input encoding and readout.

    Architecture:
        Input → Encoder → LIF Recurrent Layer → Readout → Output

    The network maintains:
        - Internal spike dynamics
        - Running rate estimates
        - Thermodynamic state (via Π_q)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 4096,
        output_dim: int = 10,
        p_connect: float = 0.05,
        frac_inh: float = 0.2,
        tau_m: float = 20e-3,
        dt: float = 1e-3,
        target_rate: float = 0.05,
        rate_ema: float = 0.99,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize TGSFN Network.

        Args:
            input_dim: Dimension of input
            hidden_dim: Number of neurons in recurrent layer
            output_dim: Dimension of output
            p_connect: Recurrent connection probability
            frac_inh: Fraction of inhibitory neurons
            tau_m: Membrane time constant
            dt: Simulation timestep
            target_rate: Target firing rate for homeostasis
            rate_ema: EMA decay for rate estimation
            device: Torch device
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.target_rate = target_rate
        self.rate_ema = rate_ema
        self.dt = dt

        self.device = device or torch.device("cpu")

        # Input encoder: projects input to all neurons
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=False)

        # Recurrent LIF layer
        self.lif = LIFLayer(
            N=hidden_dim,
            p_connect=p_connect,
            frac_inh=frac_inh,
            tau_m=tau_m,
            dt=dt,
            device=device,
        )

        # Readout: linear decoder from spike rates
        self.readout = nn.Linear(hidden_dim, output_dim)

        # Running rate estimate
        self.register_buffer(
            "rate_estimate",
            torch.ones(hidden_dim) * target_rate
        )

        # Target rates (can be made learnable or neuron-specific)
        self.register_buffer(
            "target_rates",
            torch.ones(hidden_dim) * target_rate
        )

        self.to(self.device)

    def reset(self, batch_size: int) -> None:
        """Reset network state."""
        self.lif.reset_state(batch_size)

    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: run network for num_steps timesteps.

        Args:
            x: Input tensor [batch, input_dim] or [batch, num_steps, input_dim]
            num_steps: Number of simulation steps (if x is 2D)

        Returns:
            Dict with:
                - output: Readout [batch, output_dim]
                - spikes: All spikes [batch, num_steps, hidden_dim]
                - voltages: All voltages [batch, num_steps, hidden_dim]
                - rate_series: Population rate per step [num_steps]
        """
        batch_size = x.shape[0]

        # Handle input shape
        if x.dim() == 2:
            # Static input: repeat for all steps
            x = x.unsqueeze(1).expand(-1, num_steps, -1)
        else:
            num_steps = x.shape[1]

        # Initialize state
        self.reset(batch_size)

        # Storage for outputs
        all_spikes = []
        all_voltages = []
        rate_series = []

        # Run simulation
        for t in range(num_steps):
            # Encode input
            I_ext = self.encoder(x[:, t, :])

            # LIF step
            spikes, V = self.lif(I_ext)

            all_spikes.append(spikes)
            all_voltages.append(V)

            # Population rate this step
            R_t = spikes.sum(dim=1).mean()  # Mean over batch
            rate_series.append(R_t)

            # Update running rate estimate
            with torch.no_grad():
                batch_rate = spikes.mean(dim=0)
                self.rate_estimate = (
                    self.rate_ema * self.rate_estimate +
                    (1 - self.rate_ema) * batch_rate
                )

        # Stack outputs
        spikes_tensor = torch.stack(all_spikes, dim=1)  # [batch, T, N]
        voltages_tensor = torch.stack(all_voltages, dim=1)
        rate_tensor = torch.stack(rate_series)  # [T]

        # Readout from average spike rate over time
        avg_rate = spikes_tensor.mean(dim=1)  # [batch, N]
        output = self.readout(avg_rate)

        return {
            "output": output,
            "spikes": spikes_tensor,
            "voltages": voltages_tensor,
            "rate_series": rate_tensor,
            "final_voltage": voltages_tensor[:, -1, :],
        }

    def get_thermodynamic_state(self) -> Dict[str, torch.Tensor]:
        """Get current thermodynamic quantities."""
        V, spikes = self.lif.get_state()
        J_proxy = self.lif.compute_jacobian_proxy()
        m_eff = self.lif.get_effective_branching()

        return {
            "V": V,
            "spikes": spikes,
            "J_proxy": J_proxy,
            "m_eff": m_eff,
            "rate_estimate": self.rate_estimate,
            "rate_deviation": (self.rate_estimate - self.target_rates).abs().mean(),
        }

    def get_parameters_for_dau(self) -> Dict[str, nn.Parameter]:
        """Get parameters that DAU can adjust."""
        return {
            "W_rec": self.lif.W,
            "W_enc": self.encoder.weight,
            "W_read": self.readout.weight,
        }


if __name__ == "__main__":
    print("=== TGSFN SNN Model Test ===\n")

    device = torch.device("cpu")

    # Test LIF layer
    print("--- LIFLayer Test ---")
    lif = LIFLayer(N=256, p_connect=0.1, device=device)
    lif.reset_state(batch_size=8)

    I_ext = torch.randn(8, 256) * 0.5
    spikes, V = lif(I_ext)

    print(f"  Spikes shape: {spikes.shape}")
    print(f"  Voltage shape: {V.shape}")
    print(f"  Spike rate: {spikes.mean().item():.4f}")
    print(f"  J proxy: {lif.compute_jacobian_proxy().item():.4f}")
    print(f"  Effective m: {lif.get_effective_branching().item():.4f}")

    # Test full network
    print("\n--- TGSFNNetwork Test ---")
    net = TGSFNNetwork(
        input_dim=64,
        hidden_dim=512,
        output_dim=10,
        device=device,
    )

    x = torch.randn(8, 64)
    result = net(x, num_steps=100)

    print(f"  Output shape: {result['output'].shape}")
    print(f"  Spikes shape: {result['spikes'].shape}")
    print(f"  Mean firing rate: {result['spikes'].mean().item():.4f}")
    print(f"  Rate series range: [{result['rate_series'].min():.1f}, {result['rate_series'].max():.1f}]")

    thermo = net.get_thermodynamic_state()
    print(f"  J proxy: {thermo['J_proxy'].item():.4f}")
    print(f"  Rate deviation: {thermo['rate_deviation'].item():.6f}")

    print("\n✓ SNN Model test passed!")
