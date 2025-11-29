# tests/snn/test_fabric.py
"""
Unit tests for SNN fabric layer.

Tests cover:
    - Population initialization and stepping
    - Projection forward passes
    - Fabric graph construction and validation
    - Multi-timestep simulation
    - Config loading and building
    - Parameter counting and CI gates
"""

import pytest
import torch
import tempfile
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tfan.snn.fabric import (
    SpikeBatch,
    PopulationState,
    ProjectionParams,
    LIFPopulation,
    InputPopulation,
    ReadoutPopulation,
    LowRankProjection,
    SNNFabric,
    PopulationConfig,
    ProjectionConfig,
    FabricConfig,
    build_fabric_from_config,
    build_feedforward_config,
)


class TestTypes:
    """Test core type classes."""

    def test_spike_batch_properties(self):
        """Test SpikeBatch statistics."""
        batch, N = 8, 1024
        spikes = torch.randint(0, 2, (batch, N), dtype=torch.float32)
        sb = SpikeBatch(spikes=spikes)

        assert sb.batch_size == batch
        assert sb.num_neurons == N
        assert sb.event_count == spikes.sum().item()
        assert 0.0 <= sb.sparsity <= 1.0

    def test_projection_params_gates(self):
        """Test ProjectionParams CI gate validations."""
        # Valid params (matches Ara-SYNERGY)
        params = ProjectionParams(N_pre=4096, N_post=4096, k=64, r=32)

        assert params.sparsity >= 0.98
        assert params.param_reduction_pct >= 97.0
        assert params.k <= 0.02 * params.N_pre
        assert params.r <= 0.02 * params.N_pre

    def test_projection_params_invalid(self):
        """Test ProjectionParams validation."""
        with pytest.raises(AssertionError):
            # k too large
            ProjectionParams(N_pre=100, N_post=100, k=200, r=10)

        with pytest.raises(AssertionError):
            # r negative
            ProjectionParams(N_pre=100, N_post=100, k=10, r=-1)


class TestPopulations:
    """Test neuron population classes."""

    def test_lif_population_init(self):
        """Test LIF population initialization."""
        pop = LIFPopulation(name="test", N=256, v_th=1.0, alpha=0.95)
        assert pop.name == "test"
        assert pop.N == 256

        batch = 4
        state = pop.init_state(batch=batch, device="cpu")

        assert "v" in state.data and "s" in state.data
        assert state.data["v"].shape == (batch, 256)
        assert state.data["s"].shape == (batch, 256)

    def test_lif_population_step(self):
        """Test LIF neuron dynamics."""
        pop = LIFPopulation(name="test", N=256, v_th=1.0, alpha=0.95)
        batch = 4
        state = pop.init_state(batch=batch, device="cpu")

        # Inject strong input to cause spiking
        input_current = torch.ones(batch, 256) * 2.0  # Above threshold
        new_state, spikes = pop.step(state, input_current)

        # Check that spikes occurred
        assert spikes.event_count > 0

        # Check that voltage was reset for spiking neurons
        assert new_state.data["v"].max() <= pop.v_th

    def test_input_population(self):
        """Test input population pass-through."""
        pop = InputPopulation(name="input", N=128)
        batch = 2
        state = pop.init_state(batch=batch, device="cpu")

        # Input should pass through directly
        input_spikes = torch.randint(0, 2, (batch, 128), dtype=torch.float32)
        new_state, out_spikes = pop.step(state, input_spikes)

        assert torch.allclose(out_spikes.spikes, input_spikes)

    def test_readout_population(self):
        """Test readout population smoothing."""
        pop = ReadoutPopulation(name="readout", N=64, beta=0.9)
        batch = 2
        state = pop.init_state(batch=batch, device="cpu")

        # Feed constant input and check temporal smoothing
        input_current = torch.ones(batch, 64)

        state1, spikes1 = pop.step(state, input_current)
        state2, spikes2 = pop.step(state1, input_current)

        # Output should increase over time due to integration
        assert state2.data["y"].mean() > state1.data["y"].mean()


class TestProjections:
    """Test synaptic projection classes."""

    def test_lowrank_projection_shapes(self):
        """Test low-rank projection forward pass shapes."""
        params = ProjectionParams(N_pre=512, N_post=256, k=32, r=16)
        proj = LowRankProjection(
            name="test_proj",
            pre="pop_a",
            post="pop_b",
            params=params,
        )

        batch = 4
        spikes = SpikeBatch(spikes=torch.randint(0, 2, (batch, 512), dtype=torch.float32))

        output = proj(spikes)

        assert output.shape == (batch, 256)

    def test_lowrank_projection_parameter_count(self):
        """Test that parameter count matches expected formula."""
        N_pre, N_post, k, r = 4096, 4096, 64, 32
        params = ProjectionParams(N_pre=N_pre, N_post=N_post, k=k, r=r)
        proj = LowRankProjection(
            name="test",
            pre="a",
            post="b",
            params=params,
        )

        # Count actual parameters in the synapse
        total_params = sum(p.numel() for p in proj.parameters())

        # Expected: U (N_post × r) + V (N_pre × r)
        expected_params = N_post * r + N_pre * r

        # Actual should be close (CSR mask is stored as buffers, not params)
        assert abs(total_params - expected_params) < 1000  # Allow small difference


class TestFabric:
    """Test SNN fabric graph and simulation."""

    def test_fabric_construction(self):
        """Test building a simple fabric."""
        # Build populations
        pops = {
            "input": InputPopulation("input", N=256),
            "hidden": LIFPopulation("hidden", N=128),
            "output": ReadoutPopulation("output", N=64),
        }

        # Build projections
        projs = [
            LowRankProjection(
                name="input_to_hidden",
                pre="input",
                post="hidden",
                params=ProjectionParams(N_pre=256, N_post=128, k=16, r=8),
            ),
            LowRankProjection(
                name="hidden_to_output",
                pre="hidden",
                post="output",
                params=ProjectionParams(N_pre=128, N_post=64, k=16, r=8),
            ),
        ]

        # Build fabric
        fabric = SNNFabric(populations=pops, projections=projs)

        assert len(fabric.populations) == 3
        assert len(fabric.projections) == 2
        assert fabric.total_neurons == 256 + 128 + 64

    def test_fabric_single_step(self):
        """Test a single timestep update."""
        pops = {
            "input": InputPopulation("input", N=128),
            "hidden": LIFPopulation("hidden", N=64),
        }

        projs = [
            LowRankProjection(
                name="input_to_hidden",
                pre="input",
                post="hidden",
                params=ProjectionParams(N_pre=128, N_post=64, k=8, r=4),
            ),
        ]

        fabric = SNNFabric(populations=pops, projections=projs)

        batch = 2
        states = fabric.init_state(batch=batch, device="cpu")

        # Inject spikes into input
        input_spikes = torch.randint(0, 2, (batch, 128), dtype=torch.float32)
        external = {"input": input_spikes}

        # Step once
        new_states, spikes = fabric.step(states, external_inputs=external)

        assert "input" in spikes and "hidden" in spikes
        assert spikes["input"].spikes.shape == (batch, 128)
        assert spikes["hidden"].spikes.shape == (batch, 64)

    def test_fabric_run(self):
        """Test multi-timestep simulation."""
        pops = {
            "input": InputPopulation("input", N=256),
            "output": ReadoutPopulation("output", N=128),
        }

        projs = [
            LowRankProjection(
                name="input_to_output",
                pre="input",
                post="output",
                params=ProjectionParams(N_pre=256, N_post=128, k=16, r=8),
            ),
        ]

        fabric = SNNFabric(populations=pops, projections=projs)

        # Run for 100 timesteps
        results = fabric.run(timesteps=100, batch=4, device="cpu")

        assert "spike_counts" in results
        assert "spike_rate" in results
        assert "total_events" in results
        assert "overall_spike_rate" in results

    def test_fabric_parameter_reduction(self):
        """Test that fabric achieves expected parameter reduction."""
        # Build Ara-SYNERGY-sized fabric
        config = build_feedforward_config(
            N_input=4096,
            N_hidden=4096,
            N_output=4096,
            k=64,
            r=32,
        )

        fabric = config.to_fabric()

        # Check parameter reduction meets CI gate
        assert fabric.param_reduction_pct >= 97.0


class TestConfig:
    """Test configuration loading and building."""

    def test_config_from_yaml(self, tmp_path):
        """Test loading fabric from YAML config."""
        yaml_content = """
populations:
  - name: input
    N: 256
    kind: input

  - name: hidden
    N: 128
    kind: lif
    v_th: 1.0
    alpha: 0.95

projections:
  - name: input_to_hidden
    pre: input
    post: hidden
    k: 16
    r: 8
    kind: lowrank

dtype: float32
validate: true
"""

        # Write temporary YAML file
        yaml_file = tmp_path / "test_config.yaml"
        yaml_file.write_text(yaml_content)

        # Load and build fabric
        fabric = build_fabric_from_config(str(yaml_file))

        assert len(fabric.populations) == 2
        assert len(fabric.projections) == 1
        assert fabric.total_neurons == 256 + 128

    def test_feedforward_config_builder(self):
        """Test programmatic feedforward config builder."""
        config = build_feedforward_config(
            N_input=512,
            N_hidden=256,
            N_output=128,
            k=32,
            r=16,
        )

        fabric = config.to_fabric()

        assert len(fabric.populations) == 3
        assert len(fabric.projections) == 2
        assert fabric.populations["input"].N == 512
        assert fabric.populations["hidden"].N == 256
        assert fabric.populations["output"].N == 128

    def test_config_roundtrip(self, tmp_path):
        """Test saving and loading config preserves structure."""
        config = build_feedforward_config(256, 128, 64)

        # Save to YAML
        yaml_path = tmp_path / "config.yaml"
        config.to_yaml(str(yaml_path))

        # Load back
        config_loaded = FabricConfig.from_yaml(str(yaml_path))

        assert len(config_loaded.populations) == len(config.populations)
        assert len(config_loaded.projections) == len(config.projections)


class TestCIGates:
    """Test CI gate compliance (matching bench_snn.py)."""

    def test_ara_synergy_gates(self):
        """Test that Ara-SYNERGY config meets all CI gates."""
        # Build config matching Ara-SYNERGY FPGA specs
        N, r, k = 4096, 32, 64
        config = build_feedforward_config(N, N, N, k=k, r=r)
        fabric = config.to_fabric()

        # Gate 1: param_reduction_pct ≥ 97.0
        assert fabric.param_reduction_pct >= 97.0, \
            f"Parameter reduction {fabric.param_reduction_pct:.2f}% < 97.0%"

        # Gate 2: avg_degree ≤ 0.02 * N
        for proj in fabric.projections:
            avg_degree = proj.params.k / proj.params.N_pre
            assert avg_degree <= 0.02, \
                f"Avg degree {avg_degree:.4f} > 0.02"

        # Gate 3: rank ≤ 0.02 * N
        for proj in fabric.projections:
            rank_ratio = proj.params.r / proj.params.N_pre
            assert rank_ratio <= 0.02, \
                f"Rank ratio {rank_ratio:.4f} > 0.02"

        # Gate 4: sparsity ≥ 0.98
        for proj in fabric.projections:
            assert proj.params.sparsity >= 0.98, \
                f"Sparsity {proj.params.sparsity:.4f} < 0.98"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
