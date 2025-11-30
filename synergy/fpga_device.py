"""FPGA Device Interface for Ara-SYNERGY.

Provides the hardware twin of SNNFabric with the same step() API
but backed by FPGA execution.
"""

from __future__ import annotations

import time
import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

# Import SNN types (these are shared between software and hardware)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models" / "tfan" / "snn"))
from types import PopulationState, SpikeBatch


@dataclass
class FpgaConfig:
    """FPGA configuration.

    Args:
        device_id: PCIe device identifier
        bram_size: Available BRAM in bytes
        dsp_count: Number of DSP slices
        clock_mhz: FPGA clock frequency
        timeout_ms: Operation timeout in milliseconds
    """
    device_id: int = 0
    bram_size: int = 4 * 1024 * 1024  # 4 MB
    dsp_count: int = 2520
    clock_mhz: float = 200.0
    timeout_ms: int = 1000


class FpgaHandle:
    """Low-level FPGA device handle.

    Provides PCIe communication with the FPGA board.
    This is a stub implementation - actual hardware integration
    would replace these methods with real PCIe driver calls.

    Args:
        config: FPGA configuration
    """

    def __init__(self, config: Optional[FpgaConfig] = None):
        self.config = config or FpgaConfig()
        self._connected = False
        self._bram_allocations: Dict[str, Tuple[int, int]] = {}  # name -> (offset, size)
        self._next_bram_offset = 0

    def connect(self) -> bool:
        """Connect to FPGA device.

        Returns:
            True if connection successful
        """
        # Stub: In real implementation, this would:
        # 1. Open PCIe device file
        # 2. Map BRAM regions
        # 3. Verify bitstream is loaded
        print(f"[FpgaHandle] Connecting to device {self.config.device_id}...")
        self._connected = True
        print(f"[FpgaHandle] Connected (BRAM: {self.config.bram_size / 1024 / 1024:.1f} MB)")
        return True

    def disconnect(self):
        """Disconnect from FPGA device."""
        if self._connected:
            print("[FpgaHandle] Disconnecting...")
            self._connected = False
            self._bram_allocations.clear()
            self._next_bram_offset = 0

    def allocate_bram(self, name: str, size: int) -> int:
        """Allocate BRAM region.

        Args:
            name: Region identifier
            size: Size in bytes

        Returns:
            BRAM offset
        """
        if self._next_bram_offset + size > self.config.bram_size:
            raise RuntimeError(f"BRAM allocation failed: need {size} bytes, have {self.config.bram_size - self._next_bram_offset}")

        offset = self._next_bram_offset
        self._bram_allocations[name] = (offset, size)
        self._next_bram_offset += size

        return offset

    def write_bram(self, offset: int, data: np.ndarray):
        """Write data to BRAM.

        Args:
            offset: BRAM offset
            data: Data to write
        """
        # Stub: In real implementation, this would DMA data to FPGA
        pass

    def read_bram(self, offset: int, size: int, dtype: np.dtype) -> np.ndarray:
        """Read data from BRAM.

        Args:
            offset: BRAM offset
            size: Number of elements
            dtype: Data type

        Returns:
            Data array
        """
        # Stub: In real implementation, this would DMA data from FPGA
        return np.zeros(size, dtype=dtype)

    def send_events(self, events: np.ndarray):
        """Send spike events to FPGA.

        Args:
            events: Event packet array
        """
        # Stub: In real implementation, this would:
        # 1. Format events into hardware packet format
        # 2. Write to event input FIFO
        pass

    def step(self, wait: bool = True) -> bool:
        """Trigger one simulation timestep.

        Args:
            wait: Whether to wait for completion

        Returns:
            True if step completed successfully
        """
        # Stub: In real implementation, this would:
        # 1. Assert step trigger signal
        # 2. Wait for completion flag (if wait=True)
        # 3. Check for errors
        if wait:
            time.sleep(0.001)  # Simulated execution time
        return True

    def read_spikes(self, pop_name: str, N: int, batch: int) -> np.ndarray:
        """Read spike output from population.

        Args:
            pop_name: Population name
            N: Population size
            batch: Batch size

        Returns:
            Spike array [batch, N]
        """
        # Stub: In real implementation, this would:
        # 1. Read from population's output BRAM region
        # 2. Decode spike format
        return np.zeros((batch, N), dtype=np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Get hardware statistics.

        Returns:
            Dictionary with timing and utilization stats
        """
        return {
            "bram_used": self._next_bram_offset,
            "bram_total": self.config.bram_size,
            "bram_utilization": self._next_bram_offset / self.config.bram_size,
            "connected": self._connected,
        }


class FpgaFabric:
    """Hardware twin of SNNFabric.

    Same step() signature as SNNFabric, but internally:
    - Packs spikes into event frames
    - Sends over PCIe to FPGA
    - Waits for BRAM-updated outputs
    - Returns results as PyTorch tensors

    Args:
        hw_handle: FPGA hardware handle
        hw_config: Exported fabric configuration
    """

    def __init__(self, hw_handle: FpgaHandle, hw_config: Dict[str, Any]):
        self.hw = hw_handle
        self.hw_config = hw_config

        # Map population names to metadata
        self.pop_meta = {p["name"]: p for p in hw_config.get("populations", [])}

        # Map projection names to metadata
        self.proj_meta = {p["name"]: p for p in hw_config.get("projections", [])}

        # Track state locations in BRAM
        self._state_offsets: Dict[str, int] = {}

        # Initialize hardware
        self._setup_hardware()

    def _setup_hardware(self):
        """Set up hardware resources."""
        if not self.hw._connected:
            self.hw.connect()

        # Allocate BRAM for each population's state
        for name, meta in self.pop_meta.items():
            N = meta["N"]
            # State: v, i_syn, refractory (3 * N * 4 bytes for float32)
            state_size = 3 * N * 4
            offset = self.hw.allocate_bram(f"state_{name}", state_size)
            self._state_offsets[name] = offset

        # Allocate BRAM for projection weights (already loaded during export)
        for name, meta in self.proj_meta.items():
            # CSR: indptr + indices + values
            nnz = meta.get("nnz", 0)
            N_post = meta.get("N_post", 0)
            weight_size = (N_post + 1) * 4 + nnz * 4 + nnz * 2  # indptr + indices + int16 values
            self.hw.allocate_bram(f"weights_{name}", weight_size)

    def init_state(
        self,
        batch: int = 1,
        device: str = "cpu",
    ) -> Dict[str, PopulationState]:
        """Initialize population states.

        On hardware, state lives in BRAM. Host just tracks batch dimension.

        Args:
            batch: Batch size
            device: Device for placeholder tensors

        Returns:
            Dictionary of placeholder states
        """
        states = {}
        for name, meta in self.pop_meta.items():
            N = meta["N"]
            # Create placeholder tensors (actual state is in FPGA BRAM)
            states[name] = PopulationState(data={
                "v": torch.zeros(batch, N, device=device),
                "i_syn": torch.zeros(batch, N, device=device),
                "refractory": torch.zeros(batch, N, dtype=torch.int32, device=device),
                "_fpga_managed": torch.tensor([1]),  # Flag indicating hardware state
            })

        # Initialize hardware state (reset to resting potential)
        for name, meta in self.pop_meta.items():
            N = meta["N"]
            offset = self._state_offsets[name]
            # Write initial state to BRAM
            init_state = np.zeros(3 * N, dtype=np.float32)
            self.hw.write_bram(offset, init_state)

        return states

    def step(
        self,
        states: Dict[str, PopulationState],
        external_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[Dict[str, PopulationState], Dict[str, SpikeBatch]]:
        """Advance fabric by one timestep on FPGA.

        Args:
            states: Current population states (placeholders)
            external_inputs: External currents by population name

        Returns:
            new_states: Updated states
            spikes: Spike batches for each population
        """
        external_inputs = external_inputs or {}

        # Determine batch size
        batch = 1
        for inp in external_inputs.values():
            batch = inp.shape[0]
            break

        # 1) Pack external inputs into event frames
        if external_inputs:
            events = self._encode_events(external_inputs)
            self.hw.send_events(events)

        # 2) Trigger one global timestep on FPGA
        success = self.hw.step(wait=True)
        if not success:
            raise RuntimeError("FPGA step failed")

        # 3) Read back output spikes for all populations
        spikes_out: Dict[str, SpikeBatch] = {}
        for name, meta in self.pop_meta.items():
            N = meta["N"]
            spikes_np = self.hw.read_spikes(name, N, batch)
            device = states[name].v.device if name in states else "cpu"
            spikes_t = torch.from_numpy(spikes_np).to(device)
            spikes_out[name] = SpikeBatch(spikes=spikes_t)

        # Host-side state is placeholder; real state is on FPGA
        return states, spikes_out

    def _encode_events(
        self,
        external_inputs: Dict[str, torch.Tensor],
    ) -> np.ndarray:
        """Convert external inputs to event packets.

        Args:
            external_inputs: Dictionary of input currents

        Returns:
            Event packet array for FPGA
        """
        # Simple encoding: concatenate all inputs
        # Real implementation would format into hardware-specific packets
        all_inputs = []
        for name, current in external_inputs.items():
            if current is not None:
                all_inputs.append(current.detach().cpu().numpy().flatten())

        if not all_inputs:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(all_inputs).astype(np.float32)

    def run(
        self,
        external_inputs: Optional[Dict[str, torch.Tensor]] = None,
        time_steps: int = 256,
        batch: int = 1,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Run full simulation on FPGA.

        Args:
            external_inputs: External input currents
            time_steps: Number of timesteps
            batch: Batch size

        Returns:
            spike_accumulators: Accumulated spikes per population
            aux: Auxiliary metrics
        """
        device = "cpu"
        if external_inputs:
            for v in external_inputs.values():
                device = str(v.device)
                break

        # Initialize state
        states = self.init_state(batch, device)

        # Prepare spike accumulators
        spike_accumulators = {
            name: torch.zeros(batch, meta["N"], device=device)
            for name, meta in self.pop_meta.items()
        }

        total_spikes = 0.0
        total_events = 0

        # Prepare time-varying inputs
        if external_inputs is not None:
            for key, val in external_inputs.items():
                if val.dim() == 2:
                    external_inputs[key] = val.unsqueeze(1).repeat(1, time_steps, 1)

        # Run simulation
        start_time = time.time()

        for t in range(time_steps):
            # Get inputs for this timestep
            ext_t = None
            if external_inputs is not None:
                ext_t = {
                    name: inp[:, t, :] if inp.dim() == 3 else inp
                    for name, inp in external_inputs.items()
                }

            # Step on FPGA
            states, spikes = self.step(states, ext_t)

            # Accumulate spikes
            for name, spike_batch in spikes.items():
                spike_accumulators[name] += spike_batch.spikes
                total_spikes += spike_batch.spikes.sum().item()
                total_events += spike_batch.count

        elapsed = time.time() - start_time

        # Compute metrics
        total_neurons = sum(m["N"] for m in self.pop_meta.values())
        spike_rate = total_spikes / (batch * total_neurons * time_steps)

        aux = {
            "spike_rate": spike_rate,
            "spike_sparsity": 1.0 - spike_rate,
            "active_events": total_events,
            "time_steps": time_steps,
            "elapsed_time": elapsed,
            "steps_per_second": time_steps / elapsed,
            "hardware": "fpga",
        }

        return spike_accumulators, aux

    def get_stats(self) -> Dict[str, Any]:
        """Get hardware utilization statistics."""
        hw_stats = self.hw.get_stats()

        return {
            **hw_stats,
            "populations": len(self.pop_meta),
            "projections": len(self.proj_meta),
            "total_neurons": sum(m["N"] for m in self.pop_meta.values()),
        }


class FpgaFabricModel(torch.nn.Module):
    """PyTorch module wrapping FpgaFabric.

    Provides the same (output, aux) interface as SNNFabricModel
    but executes on FPGA.

    Args:
        fpga_fabric: FpgaFabric instance
        time_steps: Simulation timesteps
        input_pop: Input population name
        output_pop: Output population name
    """

    def __init__(
        self,
        fpga_fabric: FpgaFabric,
        time_steps: int = 256,
        input_pop: str = "input",
        output_pop: str = "output",
    ):
        super().__init__()
        self.fpga_fabric = fpga_fabric
        self.time_steps = time_steps
        self.input_pop = input_pop
        self.output_pop = output_pop

    def forward(
        self,
        x: torch.Tensor,
        time_steps: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward pass on FPGA.

        Args:
            x: Input tensor [batch, N] or [batch, time, N]
            time_steps: Override default timesteps

        Returns:
            output: Output spikes [batch, N_out]
            aux: Metrics dictionary
        """
        time_steps = time_steps or self.time_steps

        # Prepare external inputs
        external_inputs = {self.input_pop: x}

        # Run on FPGA
        spike_accumulators, aux = self.fpga_fabric.run(
            external_inputs=external_inputs,
            time_steps=time_steps,
            batch=x.shape[0],
        )

        # Get output
        if self.output_pop in spike_accumulators:
            out = spike_accumulators[self.output_pop]
        else:
            out = torch.cat(
                [spike_accumulators[n] for n in sorted(spike_accumulators.keys())],
                dim=-1,
            )

        return out, aux


def create_fpga_fabric(
    export_dir: str | Path,
    device_id: int = 0,
) -> FpgaFabric:
    """Create FpgaFabric from exported data.

    Args:
        export_dir: Directory containing exported fabric
        device_id: FPGA device ID

    Returns:
        Configured FpgaFabric
    """
    import json

    export_dir = Path(export_dir)
    config_path = export_dir / "config.json"

    with open(config_path, "r") as f:
        hw_config = json.load(f)

    # Create hardware handle
    fpga_config = FpgaConfig(device_id=device_id)
    hw_handle = FpgaHandle(fpga_config)

    return FpgaFabric(hw_handle, hw_config)


__all__ = [
    "FpgaConfig",
    "FpgaHandle",
    "FpgaFabric",
    "FpgaFabricModel",
    "create_fpga_fabric",
]
