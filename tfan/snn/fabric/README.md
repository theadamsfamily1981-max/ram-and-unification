# SNN Fabric: Hardware-Shaped Spiking Neural Networks

**Hardware-friendly multi-population SNN framework for software emulation and FPGA deployment.**

---

## Overview

The SNN Fabric layer provides abstractions for building spiking neural network systems that:
- **Map cleanly to FPGA/neuromorphic hardware** (Ara-SYNERGY, Loihi, SpiNNaker)
- **Remain trainable in software** (PyTorch-based, GPU-accelerated)
- **Enforce structural constraints** (low-rank, sparse, event-driven)
- **Pass CI gates** (parameter reduction, sparsity, rank limits)

## Architecture

The fabric consists of three main components:

### 1. Populations (`populations.py`)

Neuron groups with homogeneous dynamics:

| Population Type | Description | Hardware Mapping |
|----------------|-------------|------------------|
| `LIFPopulation` | Leaky Integrate-and-Fire neurons | State buffer + LIF kernel |
| `InputPopulation` | Pass-through inputs from encoders | Input FIFO + demux |
| `ReadoutPopulation` | Continuous readout with smoothing | Accumulator + output buffer |

**Example:**
```python
from tfan.snn.fabric import LIFPopulation

pop = LIFPopulation(
    name="hidden",
    N=4096,            # Number of neurons
    v_th=1.0,          # Spike threshold
    alpha=0.95,        # Leak factor
    surrogate_scale=10.0,  # Gradient scale for training
)
```

### 2. Projections (`projections.py`)

Synaptic connectivity between populations:

| Projection Type | Description | Hardware Mapping |
|----------------|-------------|------------------|
| `LowRankProjection` | W ≈ M ⊙ (U V^T) | CSR sparse MatVec + low-rank factors |
| `DenseProjection` | Dense weights (baseline) | Full MatVec unit |

**Low-Rank Masked Synapse:**
```
W ≈ M ⊙ (U V^T)

Where:
  M: sparse binary mask (k connections per neuron, CSR format)
  U: N_post × r  (learnable)
  V: N_pre × r   (learnable)

Parameters:
  Dense:     N_pre × N_post
  This:      N_post×r + N_pre×r + k×N_post
  Reduction: 97-99% for typical values
```

**Example:**
```python
from tfan.snn.fabric import LowRankProjection, ProjectionParams

params = ProjectionParams(
    N_pre=4096,
    N_post=4096,
    k=64,     # Sparse connections per neuron
    r=32,     # Low-rank dimension
)

proj = LowRankProjection(
    name="input_to_hidden",
    pre="input",
    post="hidden",
    params=params,
)
```

### 3. Fabric (`fabric.py`)

Graph container orchestrating populations and projections:

```python
from tfan.snn.fabric import SNNFabric

fabric = SNNFabric(
    populations={"input": ..., "hidden": ..., "output": ...},
    projections=[proj1, proj2, ...],
)

# Run simulation
results = fabric.run(
    timesteps=256,
    batch=8,
    device="cuda",
)

print(f"Spike rate: {results['overall_spike_rate']:.3f}")
print(f"Parameter reduction: {fabric.param_reduction_pct:.2f}%")
```

---

## Configuration-Driven Building

Define fabrics via YAML config files:

**configs/snn/fabric_toy.yaml:**
```yaml
populations:
  - name: input
    N: 4096
    kind: input

  - name: hidden
    N: 4096
    kind: lif
    v_th: 1.0
    alpha: 0.95

  - name: output
    N: 4096
    kind: readout
    beta: 0.9

projections:
  - name: input_to_hidden
    pre: input
    post: hidden
    k: 64
    r: 32
    kind: lowrank

  - name: hidden_to_output
    pre: hidden
    post: output
    k: 64
    r: 32
    kind: lowrank

dtype: float32
validate: true
```

**Load and run:**
```python
from tfan.snn.fabric import build_fabric_from_config

fabric = build_fabric_from_config("configs/snn/fabric_toy.yaml")
results = fabric.run(timesteps=256, batch=8, device="cuda")
```

---

## CI Gates (Ara-SYNERGY Compliance)

All fabrics must pass these structural gates (enforced by `bench_snn.py`):

| Gate | Threshold | Purpose |
|------|-----------|---------|
| **param_reduction_pct** | ≥ 97.0% | Ensure low-rank sparse reduces parameters |
| **avg_degree** | ≤ 0.02 × N | Sparse connectivity constraint |
| **rank** | ≤ 0.02 × N | Low-rank dimension constraint |
| **sparsity** | ≥ 0.98 | At least 98% of weights are zero |

**Example gate validation:**
```python
config = build_feedforward_config(N_input=4096, N_hidden=4096, N_output=4096, k=64, r=32)
fabric = config.to_fabric()

assert fabric.param_reduction_pct >= 97.0
assert all(proj.params.k <= 0.02 * proj.params.N_pre for proj in fabric.projections)
assert all(proj.params.r <= 0.02 * proj.params.N_pre for proj in fabric.projections)
assert all(proj.params.sparsity >= 0.98 for proj in fabric.projections)
```

---

## Hardware Mapping

### FPGA (Ara-SYNERGY on Forest Kitten FK33)

| Software Component | FPGA Hardware | Implementation |
|--------------------|---------------|----------------|
| `Population.state` | BRAM/URAM blocks | Voltage/spike state (v, s) |
| `Population.step()` | LIF update kernel | Fixed-point arithmetic, II=1 pipeline |
| `Projection.forward()` | Sparse MatVec | CSR indexing + low-rank GEMM (W4A8) |
| `Fabric.step()` | Control FSM | Microcoded sequence of pop updates |
| **Total (N=4096, T=256)** | **< 100ms latency** | **< 30 GB/s HBM bandwidth** |

**HBM Memory Layout:**
```
PC[0-1]:   Model weights (U, V, M factors)  [streaming read]
PC[16-19]: Neuron states (v, s)            [random R/W]
PC[8-11]:  Intermediate buffers             [streaming R/W]
PC[24]:    Output features                  [burst write]
```

**Synthesis-Ready HLS:**
- Already implemented in `project-ara-synergy/fpga/ara_snn_encoder.h`
- This fabric layer is the **software emulation** that mirrors the FPGA architecture
- Training happens in software, then weights are exported to FPGA

---

## Usage Examples

### 1. Programmatic Construction

```python
from tfan.snn.fabric import (
    LIFPopulation,
    InputPopulation,
    ReadoutPopulation,
    LowRankProjection,
    ProjectionParams,
    SNNFabric,
)

# Build populations
pops = {
    "input": InputPopulation("input", N=512),
    "hidden": LIFPopulation("hidden", N=256, v_th=1.0, alpha=0.95),
    "output": ReadoutPopulation("output", N=128, beta=0.9),
}

# Build projections
projs = [
    LowRankProjection(
        name="input_to_hidden",
        pre="input",
        post="hidden",
        params=ProjectionParams(N_pre=512, N_post=256, k=32, r=16),
    ),
    LowRankProjection(
        name="hidden_to_output",
        pre="hidden",
        post="output",
        params=ProjectionParams(N_pre=256, N_post=128, k=32, r=16),
    ),
]

# Build fabric
fabric = SNNFabric(populations=pops, projections=projs)

# Run simulation
results = fabric.run(timesteps=100, batch=4, device="cuda")
print(fabric.summary())
```

### 2. Config-Driven Construction

```python
from tfan.snn.fabric import build_fabric_from_config

fabric = build_fabric_from_config("configs/snn/fabric_toy.yaml")
results = fabric.run(timesteps=256, batch=8)
```

### 3. Feedforward Builder

```python
from tfan.snn.fabric import build_feedforward_config

config = build_feedforward_config(
    N_input=4096,
    N_hidden=4096,
    N_output=4096,
    k=64,
    r=32,
)

fabric = config.to_fabric()
results = fabric.run(timesteps=256, batch=8)
```

---

## Integration with TF-A-N Training

The fabric integrates with TF-A-N's training infrastructure:

1. **SNNBackend** (planned):
   - Wraps `SNNFabric` in a `Backend` interface
   - Integrates with `FDTController` for homeostatic training
   - Logs EPR-CV, spike rate, sparsity metrics

2. **Spike Encoders** (planned):
   - `RateEncoder`, `LatencyEncoder`, `DeltaEncoder` feed `InputPopulation`
   - Convert continuous inputs to spike trains

3. **Topology Integration** (future):
   - Use TLS masks from TF-A-N attention graphs as structural priors for `M`
   - Align topological features across transformer and SNN

---

## Testing

**Run unit tests:**
```bash
# Requires PyTorch and pytest
pip install torch pytest pyyaml
python -m pytest tests/snn/test_fabric.py -v
```

**Test coverage:**
- ✓ Population initialization and dynamics
- ✓ Projection forward passes and parameter counting
- ✓ Fabric graph construction and validation
- ✓ Multi-timestep simulation
- ✓ Config loading (YAML/JSON)
- ✓ CI gate compliance (Ara-SYNERGY specs)

---

## Files

```
tfan/snn/fabric/
├── __init__.py          # Package exports
├── types.py             # SpikeBatch, PopulationState, ProjectionParams
├── populations.py       # LIF, Input, Readout populations
├── projections.py       # Low-rank masked projections
├── fabric.py            # SNNFabric graph container
├── config.py            # YAML/JSON config loaders
└── README.md            # This file

configs/snn/
└── fabric_toy.yaml      # Example 3-layer feedforward config

tests/snn/
└── test_fabric.py       # Unit tests + CI gate validation
```

---

## Next Steps

### Phase 1: Backend Integration (Immediate)
- Create `SNNBackend` in `tfan/backends/snn_emu.py` that wraps `SNNFabric`
- Integrate with `FDTController` for EPR-CV regulation
- Add spike encoder integration (`RateEncoder`, `LatencyEncoder`)
- Wire into `training/train.py` loop

### Phase 2: Hardware Binding (Near-term)
- Define CSR export format for `M` masks
- Create FPGA memory layout translator (fabric → HBM offsets)
- Implement event FIFO format for spike streams
- Build RPC layer for host ↔ FPGA communication

### Phase 3: Topology Integration (Future)
- Extract TLS masks from TF-A-N attention graphs
- Use as structural priors for SNN `M` masks
- Train jointly: transformer + SNN with shared topology

---

## Design Rationale

### Why This Architecture?

1. **Hardware-Shaped**: Clear boundaries (populations, projections) map 1:1 to FPGA blocks
2. **CI-Enforced**: Gates prevent violations of hardware constraints (sparsity, rank, params)
3. **Config-Driven**: YAML configs enable hyperparameter sweeps and reproducibility
4. **Modular**: Swap populations/projections without touching fabric logic
5. **Testable**: Software emulation allows rapid iteration before hardware deployment

### Why Low-Rank Masked Synapses?

- **Parameter Efficiency**: 97-99% reduction vs dense (262k vs 16M params for N=4096)
- **Hardware Friendly**: CSR sparse format + small dense factors (U, V) fit in BRAM
- **Event-Driven**: Only process non-zero entries in `M` (matches spike sparsity)
- **Learnable**: U, V are differentiable, allowing backprop through time (BPTT)

### Relation to Four Pillars of Impossibility

From `limits_four_pillars.md`:

- **P vs NP**: Low-rank + sparse is a heuristic approximation (not exact global optimum)
- **No Free Lunch**: Strong inductive bias (sparsity, temporal locality) matches speech/temporal tasks
- **Halting**: FDT controller + PGU gates provide runtime monitoring, not static proofs
- **Gödel**: Multi-layer (SNN + control plane + external audit), not monolithic formal core

---

## References

- **Ara-SYNERGY FPGA**: `project-ara-synergy/fpga/ara_snn_encoder.h`
- **Four Pillars**: `limits_four_pillars.md`
- **TF-A-N Design Brief**: (provided separately, describes full TF-A-N 7B architecture)
- **Surrogate Gradients**: Zenke & Ganguli (2018), Neftci et al. (2019)
- **Event-Driven SNNs**: Davies et al. (2018, Loihi), Furber et al. (2014, SpiNNaker)

---

## License

This is part of the Quanta / TF-A-N / Ara-SYNERGY project.
See top-level LICENSE for details.
