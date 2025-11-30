# SNN FPGA Solidified Program

**End-to-End Build Protocol: PyTorch SNN → FPGA Silicon**

This document is an actionable SOP for deploying the SNN fabric onto FPGA hardware. Hand this to "future you", Claude, Gemini, or Pulse and say: *"Do this to get an SNN running on FPGA."*

---

## Phase 0: Choose Your Lane (Vitis vs oneAPI)

### Step 0.1: Select Primary Stack

| Board Family | Toolchain | Language |
|--------------|-----------|----------|
| Xilinx/AMD (Alveo, Artix, Kintex, Zynq) | Vitis + Vitis HLS | C++/HLS |
| Intel (Stratix 10, Agilex, PAC) | oneAPI + DPC++ | SYCL |

### Step 0.2: Commit to SNN Flow

- **Training frontend**: PyTorch + snnTorch (or custom `src/models/tfan/snn/`)
- **Deployment mode**: Hardware-Aware Training (HAT)
  - Quantization baked into training
  - TTFS (Time-To-First-Spike) constraints enforced

### Step 0.3: Set Target Spec

```yaml
# Example target for v1 pipeline
task: "Audio KWS (Keyword Spotting)"
latency_target_ms: 1.0
resource_budget:
  luts: 50000
  bram_kb: 512
  dsp_slices: 100
board: "Alveo U50"
```

---

## Phase I: Model & Training (Hardware-Aware SNN)

### Step 1.1: Define SNN Architecture

```python
# Use LIF neurons (no Izhikevich for v1)
from models.tfan.snn import LIFPopulation, LowRankProjection

fabric = SNNFabric(name="kitten_v1", time_steps=32)
fabric.add_population("input", LIFPopulation(N=784))
fabric.add_population("hidden", LIFPopulation(N=256))
fabric.add_population("output", LIFPopulation(N=10))

# Sparse projections with CI constraints
fabric.add_projection(LowRankProjection(
    "input", "hidden",
    k=32, r=16,  # Top-k, rank
    sparsity=0.98
))
```

**Encoding**: Prefer TTFS (time-to-first-spike) from the start.

### Step 1.2: Add Hardware-Aware Training Hooks

Inject into forward pass:

```python
class HATConfig:
    w_bits: int = 8      # Weight quantization bits
    v_bits: int = 16     # Membrane potential bits
    sparsity_target: float = 0.98
    ttfs_penalty: float = 0.1  # Penalize late spikes
```

Hooks to add:
- [ ] Fake-quantization on weights during forward pass
- [ ] Surrogate gradient for spikes (e.g., fast sigmoid)
- [ ] Sparsity regularization in loss
- [ ] TTFS timing penalty (encourage early spikes)

### Step 1.3: Run HAT Training Loop

Train until:
- [ ] Accuracy ≥ minimum acceptable threshold
- [ ] TTFS timing stable under quantization
- [ ] Average spikes per timestep ≤ budget (power proxy)

### Step 1.4: Export Checkpoint

**Deliverable**: Trained SNN model + `hardware_export()` that emits:
- `fabric_topology.json` - Populations + projections structure
- `fabric_weights.bin` - Quantized weights
- `fabric_neurons.bin` - Initial neuron parameters

```bash
python scripts/export_kitten_fpga.py \
    --config configs/snn/kitten_fabric_4pop.yaml \
    --checkpoint checkpoints/kitten_v1.pt \
    --output-dir exports/kitten_v1_fpga/
```

---

## Phase II: Hardware Architecture Template

### Step 2.1: Lock the Architectural Pattern

**Neuron Side**:
- LIF cores in Processing Elements (PEs)
- Clock-driven but designed for event-sparse handling
- Fixed-point state: Q6.10 or Q4.12 for membrane potential

**Synapse Side**:
- CSR (Compressed Sparse Row) storage
- indptr[N_post+1] + indices[nnz] + values[nnz]

**Memory**:
- **SNRAM**: Synaptic weights + neuron states
- **Interconnection Memory**: spike address → neuron index mapping

### Step 2.2: Define Fixed-Point Formats

| Signal | Bitwidth | Format | Notes |
|--------|----------|--------|-------|
| Weights | 8-16 | Q1.7 / Q1.15 | Binary/ternary where possible |
| Membrane V | 16 | Q6.10 | Signed fixed-point |
| Threshold | 16 | Q6.10 | Usually constant |
| Spike | 1 | Boolean | Packed into words |
| Current I | 32 | Q16.16 | Accumulated, then truncated |

### Step 2.3: BRAM Layout Specification

```
┌─────────────────────────────────────────────────────────┐
│ BRAM Block 0: CSR indptr (N_post+1 × 32-bit)           │
├─────────────────────────────────────────────────────────┤
│ BRAM Block 1: CSR indices (nnz × 32-bit)               │
├─────────────────────────────────────────────────────────┤
│ BRAM Block 2: CSR values (nnz × 16-bit quantized)      │
├─────────────────────────────────────────────────────────┤
│ BRAM Block 3: I_post (N_post × 32-bit accumulator)     │
├─────────────────────────────────────────────────────────┤
│ BRAM Block 4: v_mem (N_post × 16-bit membrane)         │
└─────────────────────────────────────────────────────────┘
```

**Deliverable**: Architecture spec markdown with PE design, BRAM layout, tile grid.

---

## Phase III: HLS / SYCL Kernel Implementation

### Step 3.1: Core Kernel Responsibilities

```
Input: Batched spike trains or TTFS-encoded spikes

For each timestep:
  1. Read input spike/event queue
  2. For each spike, walk CSR row → update target neuron I_post
  3. Apply LIF update: leak, integrate, threshold, reset
  4. Emit new spikes to output buffer
```

### Step 3.2: HLS Implementation (Vitis)

```cpp
// Key pragmas for performance
void snn_kernel(
    hls::stream<spike_t>& in_spikes,
    hls::stream<spike_t>& out_spikes,
    int32_t* indptr,      // BRAM
    int32_t* indices,     // BRAM
    int16_t* values,      // BRAM
    int32_t* I_post,      // BRAM
    int16_t* v_mem        // BRAM
) {
    #pragma HLS INTERFACE axis port=in_spikes
    #pragma HLS INTERFACE axis port=out_spikes
    #pragma HLS INTERFACE bram port=indptr
    #pragma HLS INTERFACE bram port=indices
    #pragma HLS INTERFACE bram port=values
    #pragma HLS INTERFACE bram port=I_post
    #pragma HLS INTERFACE bram port=v_mem

    #pragma HLS DATAFLOW

    // Stage 1: CSR projection
    csr_projection_stage(in_spikes, indptr, indices, values, I_post);

    // Stage 2: LIF update
    lif_update_stage(v_mem, I_post, out_spikes);
}
```

### Step 3.3: SYCL Implementation (Intel oneAPI)

```cpp
[[intel::kernel]]
void snn_kernel(
    sycl::pipe<spike_t>& in_pipe,
    sycl::pipe<spike_t>& out_pipe,
    /* device pointers... */
) {
    // Single-task kernel with pipelined stages
    [[intel::initiation_interval(1)]]
    for (int t = 0; t < TIME_STEPS; t++) {
        // CSR traversal + LIF update
    }
}
```

### Step 3.4: Iterate Until Reports Pass

**Checklist**:
- [ ] LUT/FF/DSP/BRAM fit on target FPGA
- [ ] FMax ≥ 200 MHz (or board spec)
- [ ] Loop II (initiation interval) = 1 on critical paths
- [ ] No huge inferred memories where you meant BRAM

**Deliverable**: Kernel project that builds to `.xo` (Vitis) or `.aocx` (Intel).

---

## Phase IV: Bitstream, BSP, and Host Runtime

### Step 4.1: Hardware Install

- [ ] Seat card, check PCIe lane width in BIOS
- [ ] Confirm power and cooling margins
- [ ] Run `lspci | grep -i xilinx` or `lspci | grep -i intel`

### Step 4.2: Software Stack

**Xilinx**:
```bash
# Install XRT
sudo apt install xrt
# Install platform
sudo apt install xilinx-u50-gen3x16-xdma-5-202210-1-dev
# Validate
xbutil validate
```

**Intel**:
```bash
# Install oneAPI + FCD
source /opt/intel/oneapi/setvars.sh
# Check board
aocl diagnose
fpgainfo bmc
```

### Step 4.3: Build Final Bitstream

**Xilinx**:
```bash
v++ -l -t hw \
    --platform xilinx_u50_gen3x16_xdma_5_202210_1 \
    -o snn_kernel.xclbin \
    snn_kernel.xo
```

**Intel**:
```bash
icpx -fintelfpga -Xshardware \
    -Xsboard=pac_s10 \
    -o snn_kernel.aocx \
    snn_kernel.cpp
```

### Step 4.4: Host Application

```python
# XRT Python API example
import pyxrt

device = pyxrt.device(0)
xclbin = pyxrt.xclbin("snn_kernel.xclbin")
device.load_xclbin(xclbin)

# Allocate buffers
bo_indptr = pyxrt.bo(device, indptr_data.nbytes, pyxrt.bo.normal, kernel.group_id(0))
bo_spikes_in = pyxrt.bo(device, spikes.nbytes, pyxrt.bo.normal, kernel.group_id(1))

# Run kernel
bo_indptr.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
bo_spikes_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

run = kernel(bo_indptr, bo_spikes_in, bo_spikes_out, N_POST, NNZ)
run.wait()

bo_spikes_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
```

**Deliverable**: `run_snn_fpga` executable that accepts test data and prints latency/accuracy.

---

## Phase V: Dynamic Partial Reconfiguration (Optional)

### Step 5.1: Partition Design

**Static Region**:
- PCIe / HBM / DDR interface
- DMA engines, global control, monitoring

**Reconfigurable Region(s)**:
- SNN cores/tiles (swap topologies, weights, etc.)

### Step 5.2: Build DPR Flow

Generate:
- Base bitstream with static FIM
- Partial bitstreams for SNN variants

Host flow:
1. Pause/quiesce traffic
2. Trigger partial reconfig via PCAP/JTAG/driver
3. Resume operation

**Result**: 800× faster than full reconfig for topology changes.

---

## Phase VI: System-Level Optimization & CI

### Step 6.1: Profile Host-Accelerator Pipeline

Measure:
- T_compute (kernel execution)
- T_pcie (data transfer)
- T_host (driver/runtime overhead)

If kernel is sub-millisecond, focus on:
- Minimize host layers
- Persistent kernel model (kernel loops, host feeds via shared buffer)
- Batch spike transfers

### Step 6.2: CI Gates

```yaml
# .github/workflows/fpga-ci.yml
fpga_tests:
  functional:
    - name: "CPU vs FPGA output match"
      tolerance: 1e-4

  performance:
    - name: "Inference latency"
      max_ms: 1.0
    - name: "Power draw"
      max_watts: 25

  resources:
    - name: "LUT utilization"
      max_percent: 80
    - name: "BRAM utilization"
      max_percent: 70
```

---

## Bridge: Software Fabric → FPGA

### Export Format

The `FPGAExporter` in `src/models/tfan/snn/fabric/export.py` produces:

```
exports/kitten_v1_fpga/
├── config.json           # Fabric metadata
├── proj_input_hidden.bin # CSR + quantized weights
├── proj_hidden_output.bin
├── neurons.bin           # Initial states
└── hls_structs.h         # C struct definitions for HLS
```

### HLS Struct Layout

```c
// hls_structs.h - Generated by FPGAExporter
typedef struct {
    int32_t N_pre;
    int32_t N_post;
    int32_t k;
    int32_t r;
    int32_t nnz;
    float scale;
} proj_header_t;

typedef struct {
    int16_t v;          // Membrane potential (Q6.10)
    int16_t v_th;       // Threshold
    int16_t alpha;      // Leak factor
    uint8_t refractory; // Refractory counter
    uint8_t _pad;
} neuron_state_t;
```

### Validation Loop

```python
# Same input → compare CPU vs FPGA
spikes_in = torch.randint(0, 2, (batch, N_pre, time_steps))

# CPU reference
cpu_out = snn_fabric_cpu(spikes_in)

# FPGA execution
fpga_out = run_snn_fpga(spikes_in)

# Compare
assert torch.allclose(cpu_out, fpga_out, atol=1e-4)
```

---

## Quick Reference: File Locations

| Component | Path |
|-----------|------|
| SNN Fabric (PyTorch) | `src/models/tfan/snn/` |
| Fabric Export | `src/models/tfan/snn/fabric/export.py` |
| RTL Modules | `rtl/` |
| HLS Kernels | `hls/` (to be created) |
| FPGA Configs | `configs/snn/*.yaml` |
| Export Scripts | `scripts/export_kitten_fpga.py` |
| Baseline Comparison | `scripts/run_sb852_baseline.py` |

---

## Checklist Summary

### Phase 0: Setup
- [ ] Choose Vitis (Xilinx) or oneAPI (Intel)
- [ ] Set target task, latency, and resource budget

### Phase I: Training
- [ ] Define SNN with LIF + CSR projections
- [ ] Add HAT hooks (quantization, surrogate grad, sparsity)
- [ ] Train to convergence
- [ ] Export checkpoint

### Phase II: Architecture
- [ ] Lock PE design and fixed-point formats
- [ ] Document BRAM layout
- [ ] Write architecture spec

### Phase III: Kernel
- [ ] Implement HLS/SYCL kernel
- [ ] Add pragmas/attributes for pipelining
- [ ] Iterate until resource/timing reports pass

### Phase IV: Deployment
- [ ] Install BSP and validate hardware
- [ ] Build bitstream/xclbin
- [ ] Implement host application
- [ ] Benchmark end-to-end

### Phase V: DPR (Optional)
- [ ] Partition static/reconfigurable regions
- [ ] Build partial bitstreams
- [ ] Test hot-swap flow

### Phase VI: CI
- [ ] Functional tests (CPU vs FPGA match)
- [ ] Performance gates (latency, power)
- [ ] Resource gates (LUT, BRAM utilization)

---

*Last updated: 2025-11-30*
*Version: 1.0*
