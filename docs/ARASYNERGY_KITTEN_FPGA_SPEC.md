# Ara-SYNERGY Kitten FPGA Specification

This document specifies the FPGA implementation of the "Kitten" SNN fabric
for deployment on Stratix-10 or similar FPGAs.

## 1. Overview

The Kitten fabric is a 4-population spiking neural network with CI-compliant
sparse connectivity. This spec covers:

1. **Export Format**: How PyTorch fabric data maps to FPGA configuration
2. **PCIe Protocol**: Host-FPGA communication interface
3. **HDL Module Hierarchy**: Verilog/SystemVerilog structure
4. **Memory Layout**: BRAM allocation for weights and state

## 2. Kitten Fabric Architecture

```
┌─────────┐     ┌──────────┐     ┌──────────┐     ┌────────┐
│  Input  │────▶│ Hidden1  │────▶│ Hidden2  │────▶│ Output │
│  4096   │     │  4096    │     │  4096    │     │  2048  │
└─────────┘     └──────────┘     └──────────┘     └────────┘
                     │                │
                     └────────────────┘
                     (recurrent connections)
```

### 2.1 Population Parameters

| Population | N    | v_th | alpha | Notes          |
|------------|------|------|-------|----------------|
| input      | 4096 | 1.0  | 0.95  | External input |
| hidden1    | 4096 | 1.0  | 0.97  | + recurrent    |
| hidden2    | 4096 | 1.0  | 0.97  | + recurrent    |
| output     | 2048 | 0.9  | 0.98  | Readout layer  |

### 2.2 Projection Parameters (CI-Compliant)

| Projection           | N_pre | N_post | k  | r  | Sparsity |
|---------------------|-------|--------|----|----|----------|
| input_to_hidden1    | 4096  | 4096   | 64 | 32 | 98.4%    |
| hidden1_to_hidden2  | 4096  | 4096   | 64 | 32 | 98.4%    |
| hidden2_to_output   | 4096  | 2048   | 64 | 32 | 98.4%    |
| hidden1_recurrent   | 4096  | 4096   | 32 | 16 | 99.2%    |
| hidden2_recurrent   | 4096  | 4096   | 32 | 16 | 99.2%    |

**CI Gates**:
- `r/N ≤ 0.02` (max rank ratio)
- `k/N ≤ 0.02` (max degree ratio)
- `sparsity ≥ 98%`

## 3. Export Format

### 3.1 Bundle Structure

```
kitten_fpga/
├── config.json           # Fabric metadata
├── proj_input_to_hidden1.bin
├── proj_hidden1_to_hidden2.bin
├── proj_hidden2_to_output.bin
├── proj_hidden1_recurrent.bin
└── proj_hidden2_recurrent.bin
```

### 3.2 config.json Schema

```json
{
  "fabric_name": "kitten_4pop",
  "time_steps": 256,
  "dt": 1.0,
  "populations": [
    {"name": "input", "N": 4096},
    {"name": "hidden1", "N": 4096},
    {"name": "hidden2", "N": 4096},
    {"name": "output", "N": 2048}
  ],
  "total_neurons": 14336,
  "total_synapses": 786432,
  "projection_count": 5
}
```

### 3.3 Projection Binary Format

Each `proj_*.bin` file:

```
[Header: 20 bytes]
  N_pre:  int32 (4 bytes)
  N_post: int32 (4 bytes)
  k:      int32 (4 bytes)
  r:      int32 (4 bytes)
  nnz:    int32 (4 bytes)

[CSR indptr: (N_post + 1) * 4 bytes]
  int32 array

[CSR indices: nnz * 4 bytes]
  int32 array

[Weight scale: 4 bytes]
  float32

[Quantized weights: nnz * 2 bytes]
  int16 array (symmetric, scale provided above)
```

### 3.4 Quantization

- **Format**: 16-bit symmetric quantization
- **Scale**: `max_abs / 32767`
- **Dequantize**: `value_float = value_int16 * scale`

## 4. PCIe Protocol

### 4.1 Memory-Mapped Registers (BAR0)

```c
// Control/Status Registers
#define REG_CTRL        0x00  // bit0: reset, bit1: start_step
#define REG_STATUS      0x04  // bit0: busy, bit1: error
#define REG_BATCH       0x08  // Batch size (uint32)
#define REG_N_INPUT     0x0C  // Input population size
#define REG_N_OUTPUT    0x10  // Output population size

// DMA Addresses
#define REG_IN_ADDR_LO  0x20  // Input buffer address (lower 32 bits)
#define REG_IN_ADDR_HI  0x24  // Input buffer address (upper 32 bits)
#define REG_OUT_ADDR_LO 0x28  // Output buffer address (lower 32 bits)
#define REG_OUT_ADDR_HI 0x2C  // Output buffer address (upper 32 bits)

// Step Control
#define REG_STEP_ID     0x30  // Step counter (incremented by host)
#define REG_TIMEOUT_CYC 0x34  // Timeout in clock cycles (optional)
```

### 4.2 Step Protocol

1. **Host**: Fill input buffer with `[batch, N_input]` float32 currents
2. **Host**: Increment `REG_STEP_ID`
3. **Host**: Set `REG_CTRL.start_step = 1`
4. **FPGA**:
   - Clear `STATUS.busy = 0`, set `busy = 1`
   - DMA-read input buffer
   - Execute one SNN timestep (LIF + CSR MatVec)
   - DMA-write spike buffer
   - Clear `busy = 0`
5. **Host**: Poll `REG_STATUS.busy` with timeout
6. **Host**: Read output buffer `[batch, N_output]` bytes

### 4.3 Polling Implementation

```c
int wait_for_step_done(struct kitten_dev *dev, uint32_t expected_step_id, int timeout_ms) {
    const int sleep_us = 50;
    int waited_us = 0;

    while (waited_us < timeout_ms * 1000) {
        uint32_t status = bar0_read32(dev, REG_STATUS);
        uint32_t step_id = bar0_read32(dev, REG_STEP_ID);

        bool busy = status & 0x1;
        if (!busy && step_id == expected_step_id)
            return 0;  // Success

        usleep(sleep_us);
        waited_us += sleep_us;
    }
    return -ETIMEDOUT;
}
```

### 4.4 Data Formats

**Input**: Dense float32 currents `[batch, N_input]`
- Size: `batch * 4096 * 4 = 16KB` per step (batch=1)

**Output**: Byte-per-neuron spikes `[batch, N_output]`
- Size: `batch * 2048 * 1 = 2KB` per step (batch=1)
- Values: 0 = no spike, 1 = spike

## 5. HDL Module Hierarchy

### 5.1 Top-Level Structure

```
kitten_fpga_top
├── pcie_axi_bridge          # PCIe to AXI conversion
├── kitten_control_regs      # BAR0 register block
├── kitten_dma_in            # Host → FPGA DMA engine
├── kitten_dma_out           # FPGA → Host DMA engine
├── kitten_step_scheduler    # Orchestrates one timestep
└── kitten_fabric_tile       # The actual SNN fabric
    ├── lif_population[0]    # Input population
    ├── lif_population[1]    # Hidden1 population
    ├── lif_population[2]    # Hidden2 population
    ├── lif_population[3]    # Output population
    ├── csr_projection[0]    # input_to_hidden1
    ├── csr_projection[1]    # hidden1_to_hidden2
    ├── csr_projection[2]    # hidden2_to_output
    ├── csr_projection[3]    # hidden1_recurrent
    ├── csr_projection[4]    # hidden2_recurrent
    └── fabric_fsm           # Execution controller
```

### 5.2 kitten_fabric_tile

```systemverilog
module kitten_fabric_tile #(
    parameter N_INPUT  = 4096,
    parameter N_H1     = 4096,
    parameter N_H2     = 4096,
    parameter N_OUTPUT = 2048
) (
    input  wire         clk,
    input  wire         rst,

    // Control
    input  wire         i_start,
    output wire         o_done,

    // Input currents (AXIS)
    input  wire [31:0]  i_input_current,
    input  wire         i_input_valid,
    output wire         o_input_ready,

    // Output spikes (AXIS)
    output wire [7:0]   o_spike_data,
    output wire         o_spike_valid,
    input  wire         i_spike_ready
);
```

### 5.3 lif_population

```systemverilog
module lif_population #(
    parameter N     = 4096,
    parameter V_TH  = 16'h4000,  // 1.0 in Q15
    parameter ALPHA = 16'h7999   // 0.95 in Q15
) (
    input  wire         clk,
    input  wire         rst,

    // Control
    input  wire         i_step_en,
    output wire         o_step_done,

    // Current input (streaming)
    input  wire [31:0]  i_current,
    input  wire         i_current_valid,
    output wire         o_current_ready,

    // Spike output (streaming)
    output wire         o_spike,
    output wire         o_spike_valid,
    input  wire         i_spike_ready
);

// Internal BRAMs
// mem_v[N]: Membrane potential (32-bit fixed-point)
// Processing: time-multiplexed over ceil(N/DSP_UNITS) cycles
```

### 5.4 csr_projection

```systemverilog
module csr_projection #(
    parameter N_PRE  = 4096,
    parameter N_POST = 4096,
    parameter K_MAX  = 64,
    parameter ADDRW  = 13  // log2(N_POST + 1)
) (
    input  wire         clk,
    input  wire         rst,

    // Control
    input  wire         i_start,
    output wire         o_done,

    // Pre-synaptic spikes (bit vector or stream)
    input  wire [N_PRE-1:0] i_spike_vec,

    // BRAM interfaces for CSR data
    output wire [ADDRW-1:0] o_indptr_addr,
    input  wire [31:0]      i_indptr_data,

    output wire [ADDRW-1:0] o_indices_addr,
    input  wire [31:0]      i_indices_data,

    output wire [ADDRW-1:0] o_values_addr,
    input  wire [15:0]      i_values_data,

    // Scale factor (loaded at config time)
    input  wire [31:0]      i_scale,

    // Post-synaptic current output (streaming)
    output wire [31:0]      o_current,
    output wire             o_current_valid,
    input  wire             i_current_ready
);
```

**CSR MatVec Algorithm**:
1. For each row `r` in 0..N_POST-1:
   - Read `start = indptr[r]`, `end = indptr[r+1]`
   - For `i` in `start..end`:
     - Read `col = indices[i]`
     - If `spike_vec[col] == 1`:
       - Read `w = values[i]`
       - Accumulate `current[r] += w * scale`
2. Stream accumulated currents to output

## 6. BRAM Allocation

### 6.1 Per-Projection Storage

| Component | Size (bytes) | BRAMs (18Kb) |
|-----------|-------------|--------------|
| indptr    | (N_post+1)*4 | 1-2          |
| indices   | nnz * 4      | varies       |
| values    | nnz * 2      | varies       |

### 6.2 Per-Population Storage

| Component | Size (bytes) | BRAMs (18Kb) |
|-----------|-------------|--------------|
| v (membrane) | N * 4    | 2-4          |
| refractory   | N * 1    | 0.5          |

### 6.3 Total Estimates (Kitten)

```
Projections:
  5 projections × ~3 BRAMs each = ~15 BRAMs for CSR data

Populations:
  4 populations × ~3 BRAMs each = ~12 BRAMs for state

Total: ~27 BRAMs minimum (well within Stratix-10 capacity)
```

## 7. Execution Flow

### 7.1 Fabric FSM States

```
IDLE → LOAD_INPUT → RUN_PROJECTIONS → UPDATE_NEURONS → WRITE_OUTPUT → DONE
```

### 7.2 RUN_PROJECTIONS Substates

```
P0_INPUT_TO_H1 → P1_H1_TO_H2 → P2_H2_TO_OUT → P3_H1_RECUR → P4_H2_RECUR
```

Each projection substep:
1. Read spike vector from pre-population
2. Execute CSR MatVec
3. Accumulate currents into post-population buffer

### 7.3 UPDATE_NEURONS Substates

```
N0_INPUT → N1_HIDDEN1 → N2_HIDDEN2 → N3_OUTPUT
```

Each population update:
1. For each neuron (time-multiplexed):
   - Read `v`, `i_syn`, `refractory`
   - Compute: `v_new = alpha * v + (1-alpha) * (v_rest + i_syn)`
   - Check: `spike = (v_new >= v_th) && (refractory == 0)`
   - Update: `v = spike ? v_reset : v_new`
   - Write back

## 8. Python Host API

### 8.1 FpgaFabric Class

```python
class FpgaFabric:
    def __init__(self, pcie_dev, N_input, N_output, batch):
        self.dev = pcie_dev
        self.batch = batch

        # Allocate pinned buffers
        self.in_buf = self.dev.alloc_pinned(batch * N_input * 4)
        self.out_buf = self.dev.alloc_pinned(batch * N_output)

        # Configure registers
        self.dev.write_reg("BATCH", batch)
        self.dev.write_reg("N_INPUT", N_input)
        self.dev.write_reg("N_OUTPUT", N_output)
        self.dev.write_reg("IN_ADDR", self.in_buf.phys_addr)
        self.dev.write_reg("OUT_ADDR", self.out_buf.phys_addr)

    def step(self, current: np.ndarray) -> np.ndarray:
        """Execute one SNN timestep on FPGA."""
        # Copy input
        np.copyto(self.in_buf.view(), current)

        # Trigger step
        step_id = self.dev.increment_step_id()
        self.dev.start_step()

        # Poll with timeout
        if self.dev.wait_done(step_id, timeout_ms=100) != 0:
            raise TimeoutError("FPGA step timeout")

        # Return spikes
        return self.out_buf.view().copy()
```

### 8.2 Integration with Training

```python
# Select backend based on config
if cfg.device == "fpga":
    fabric = FpgaFabric.from_export("artifacts/kitten_fpga")
    model = FpgaFabricModel(fabric)
else:
    fabric = build_fabric_from_config(cfg)
    model = SNNFabricModel(fabric)

# Same API for both
out, aux = model(x)
```

## 9. Validation Checklist

- [ ] Export bundle generates valid JSON + binary files
- [ ] Config JSON matches fabric structure
- [ ] Quantized weights within ±0.1% of float originals
- [ ] CSR indptr/indices match TLS mask
- [ ] PCIe registers read/write correctly
- [ ] DMA transfers complete without corruption
- [ ] Single step produces identical spikes (CPU vs FPGA)
- [ ] 256-step simulation matches CPU within tolerance
- [ ] Timeout handling works (no infinite blocking)
- [ ] CI audit passes for exported fabric

## 10. Performance Targets

| Metric | Target |
|--------|--------|
| Latency per step | < 1 ms |
| Throughput | > 1000 steps/sec |
| Power | < 50W |
| BRAM utilization | < 50% |

## Appendix A: CSR MatVec Pseudocode

```python
def csr_matvec_fpga(indptr, indices, values, scale, spike_vec):
    """
    CSR sparse matrix-vector multiply for SNN projection.

    indptr:  [N_post + 1] int32 - Row pointers
    indices: [nnz] int32 - Column indices
    values:  [nnz] int16 - Quantized weights
    scale:   float32 - Dequantization scale
    spike_vec: [N_pre] bit - Pre-synaptic spikes

    Returns: [N_post] float32 - Post-synaptic currents
    """
    current = np.zeros(N_post, dtype=np.float32)

    for row in range(N_post):
        start = indptr[row]
        end = indptr[row + 1]

        acc = 0.0
        for i in range(start, end):
            col = indices[i]
            if spike_vec[col]:
                w = values[i] * scale
                acc += w

        current[row] = acc

    return current
```

## Appendix B: LIF Update Pseudocode

```python
def lif_update_fpga(v, i_syn, refractory, alpha, v_th, v_reset):
    """
    Leaky Integrate-and-Fire neuron update.

    All arrays: [N] neurons
    Returns: (v_new, spike, refractory_new)
    """
    spike = np.zeros(N, dtype=np.uint8)
    v_new = v.copy()
    ref_new = refractory.copy()

    for n in range(N):
        if refractory[n] > 0:
            ref_new[n] = refractory[n] - 1
            continue

        # Membrane update
        v_new[n] = alpha * v[n] + (1 - alpha) * i_syn[n]

        # Spike check
        if v_new[n] >= v_th:
            spike[n] = 1
            v_new[n] = v_reset
            ref_new[n] = REFRACTORY_STEPS

    return v_new, spike, ref_new
```
