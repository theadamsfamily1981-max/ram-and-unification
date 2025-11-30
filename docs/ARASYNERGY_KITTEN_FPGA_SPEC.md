# Ara-SYNERGY "Kitten" FPGA SNN Fabric

**Version**: v0.2
**Status**: Draft hardware spec (software fabric + export already implemented)

---

## 1. Overview

This document defines the FPGA implementation of the **Kitten SNN fabric** used by the Ara-SYNERGY project.

The design maps the **software SNN fabric layer** (already implemented in the TF-A-N repo) onto a Stratix-10 class FPGA as a **fixed SNN tile**:

- 4 populations: `input → hidden1 → hidden2 → output`
- Projection graph:
  - `input → hidden1`
  - `hidden1 → hidden2`
  - `hidden2 → output`
  - `hidden1 → hidden1` (recur)
  - `hidden2 → hidden2` (recur)
- All synapses implemented as **CSR (Compressed Sparse Row) + low-rank masked weights**
- Neuron model: **discrete-time LIF** (leaky integrate-and-fire)
- Host access via **PCIe + AXI**, with DMA for input currents and spike outputs

This spec is the contract between:

- The **software fabric export** (Python → `artifacts/kitten_fpga/…`)
- The **FPGA bitstream** and
- The **host runtime** (Linux driver / user-space library)

---

## 2. Compute Model

### 2.1 Populations

Four neuron populations:

| Name    | Size (N) |
|---------|----------|
| input   | 4096     |
| hidden1 | 4096     |
| hidden2 | 4096     |
| output  | 2048     |

Each population consists of `N` LIF neurons, updated every **simulation step**:

- Membrane potential `v[i]`
- Input current/drive `I[i]`
- Spike output `s[i] ∈ {0,1}`

Reference LIF update (software):

```python
v = alpha * v + I - v_th * spike
spike = (v >= v_th).float()
```

The FPGA implementation may use a time-multiplexed update (e.g. 1–32 neurons updated per cycle) depending on resource budget.

### 2.2 Projections

Each projection implements:

```
I_post = I_post + W · s_pre
```

Where:
- `s_pre` = spike vector from pre-population
- `W` = sparse weight matrix (CSR, low-rank masked in training)
- `I_post` = accumulated current for post-population

Projections in Kitten:

| # | Name                 | Shape        |
|---|----------------------|--------------|
| 1 | input_to_hidden1     | 4096 → 4096  |
| 2 | hidden1_to_hidden2   | 4096 → 4096  |
| 3 | hidden2_to_output    | 4096 → 2048  |
| 4 | hidden1_recur        | 4096 → 4096  |
| 5 | hidden2_recur        | 4096 → 4096  |

On FPGA these appear as CSR projection engines sharing a common pattern.

---

## 3. Host-FPGA Interface

### 3.1 PCIe + BAR Layout

The FPGA exposes:
- One BAR for control & status (BAR0, AXI-Lite)
- DMA engines for:
  - Input buffer (currents)
  - Output buffer (spikes)

All control is done by memory-mapped registers in BAR0.

### 3.1.1 Control & Status Registers (BAR0)

All offsets 32-bit aligned.

| Offset | Name        | Dir | Description                              |
|--------|-------------|-----|------------------------------------------|
| 0x00   | CTRL        | R/W | Control bits                             |
| 0x04   | STATUS      | R   | Status bits                              |
| 0x08   | BATCH       | R/W | Batch size in samples                    |
| 0x0C   | N_INPUT     | R/W | Number of input neurons (default 4096)   |
| 0x10   | N_OUTPUT    | R/W | Number of output neurons (default 2048)  |
| 0x14   | STEP_ID     | R/W | Step counter (host increments, FPGA echoes) |
| 0x20   | IN_ADDR_LO  | R/W | Input buffer low 32 bits                 |
| 0x24   | IN_ADDR_HI  | R/W | Input buffer high 32 bits                |
| 0x28   | OUT_ADDR_LO | R/W | Output buffer low 32 bits                |
| 0x2C   | OUT_ADDR_HI | R/W | Output buffer high 32 bits               |
| 0x30   | TIMEOUT_CYC | R/W | Optional internal timeout (cycles)       |

**Bitfields:**

- **CTRL**:
  - bit0: `SOFT_RESET` (write 1 to reset internal FSMs)
  - bit1: `START_STEP` (write 1 to start a new step)

- **STATUS**:
  - bit0: `BUSY` (1 while step in progress)
  - bit1: `ERROR` (1 if any internal error)
  - bit2: `TIMEOUT` (1 if internal timeout triggered)
  - bit3: `DONE` (1 when last step completed; clears on new START_STEP)

### 3.2 Data Buffers

Host allocates pinned / DMA-capable buffers:

**Input current buffer**: float32 or Qm.n fixed-point (see 4.1)
- Shape (logical): `[BATCH, N_INPUT]`
- Layout: row-major, contiguous

**Output spike buffer**: uint8 (bit-packed or byte-per-neuron)
- Option A: 1 byte per neuron (0 or 1)
- Option B: bit-packed (8 neurons per byte)

The initial implementation uses 1 byte per neuron for simplicity.

---

## 4. Data Representation

### 4.1 Fixed-Point Weights

Synaptic weights are exported from Python as:
- `values_q`: 16-bit signed fixed-point (e.g. Q1.14)
- `scale_q`: per-projection scale factor (float32)

On FPGA:

```
w_real ≈ values_q * 2^(-14) * scale_q
```

This allows us to:
- Pack weights tightly in BRAM (16 bits)
- Keep the dynamic range of the trained model via `scale_q`

### 4.2 CSR Layout in BRAM

For each projection (pre → post):

Let:
- `N_pre` = number of presynaptic neurons
- `nnz` = number of non-zero synapses

We store 3 main arrays in on-chip RAM:

1. **indptr**: length `N_pre + 1`, 32-bit unsigned
   - Row pointers (CSR)
   - For pre neuron `j`, synapses live in `[indptr[j], indptr[j+1])`

2. **indices**: length `nnz`, 32-bit unsigned
   - Post neuron indices `i`

3. **values_q**: length `nnz`, 16-bit signed
   - Quantized synaptic weights

These are conceptually single-port or dual-port BRAMs with simple address/data interfaces.

Each projection also has a small set of scalar registers:
- `scale_q`: float32 or fixed-point (e.g. Q1.14)
- `N_pre`, `N_post`: sizes

These are filled by the fabric export script and a board-specific loader (JTAG/UART/PCIe).

---

## 5. Top-Level Module

### 5.1 kitten_fpga_top

Top entity (simplified):

```systemverilog
module kitten_fpga_top (
    input  wire         pcie_refclk_p,
    input  wire         pcie_refclk_n,
    input  wire [15:0]  pcie_rx_p,
    input  wire [15:0]  pcie_rx_n,
    output wire [15:0]  pcie_tx_p,
    output wire [15:0]  pcie_tx_n,
    input  wire         sys_reset_n
);
```

Internally instantiates:
- `pcie_axi_bridge`
- `kitten_control_regs` (BAR0 mapping)
- `kitten_dma_in` / `kitten_dma_out`
- `kitten_step_scheduler`
- `kitten_fabric_tile`

Main responsibilities:
1. Expose a PCIe endpoint
2. Translate BAR0 accesses → control registers
3. Orchestrate:
   - Input DMA
   - Fabric step
   - Output DMA
4. Report BUSY, DONE, ERROR bits to host

---

## 6. Fabric Tile

### 6.1 kitten_fabric_tile Interface

```systemverilog
module kitten_fabric_tile #(
    parameter N_INPUT  = 4096,
    parameter N_H1     = 4096,
    parameter N_H2     = 4096,
    parameter N_OUTPUT = 2048
) (
    input  wire         clk,
    input  wire         rst,

    // Step control
    input  wire         i_start,
    output wire         o_done,

    // Streaming input currents for `input` population
    input  wire [31:0]  i_input_curr,
    input  wire         i_input_valid,
    output wire         o_input_ready,

    // Streaming spike outputs for `output` population
    output wire [7:0]   o_spike_data,
    output wire         o_spike_valid,
    input  wire         i_spike_ready
);
```

The tile is a step engine:
- On `i_start`:
  1. Load input currents from AXIS stream
  2. Run projections and LIF updates for all populations
  3. Emit packed spike bytes for output population
  4. Assert `o_done`

The internal design is broken into:
- Population modules (`lif_population`)
- Projection engines (`csr_projection`)
- Fabric scheduler FSM (`fabric_ctrl_fsm`)

---

## 7. Control Flow for One Step

### 7.1 High-Level Flow

1. **Host**:
   - Writes input buffer → `IN_ADDR_*`
   - Writes output buffer → `OUT_ADDR_*`
   - Programs `BATCH`, `N_INPUT`, `N_OUTPUT`
   - Increments `STEP_ID`
   - Writes `CTRL.START_STEP = 1`

2. **FPGA (scheduler)**:
   - Sets `STATUS.BUSY = 1`
   - Launches `kitten_dma_in` to read input buffer
   - Streams currents into `kitten_fabric_tile`
   - Waits for `fabric_tile.o_done`
   - Launches `kitten_dma_out` to write output spikes
   - Sets `STATUS.DONE = 1`, clears `STATUS.BUSY`

3. **Host**:
   - Polls `STATUS.BUSY → 0` with timeout
   - Verifies `STATUS.DONE == 1` and `STATUS.ERROR == 0`
   - Reads output buffer

### 7.2 Host Timeout (ncio-lock style)

Host-side logic must not block indefinitely. A typical pattern:

```c
int wait_for_step_done(struct kitten_dev *dev,
                       uint32_t step_id,
                       int timeout_ms);
```

Implementation:
- Poll `STATUS.BUSY` and `STEP_ID` every few microseconds
- Abort if `timeout_ms` exceeded
- Set error flag if `STATUS.ERROR` or `STATUS.TIMEOUT` is set

---

## 8. Configuration Export

The software fabric export (already implemented) writes:
- `kitten_fabric_config.json`
- Per-projection `.npy` arrays for `indptr`, `indices`, `values_q`
- Per-projection `scale_q`

This spec assumes a board-specific loader (e.g. small Python script + JTAG/PCIe utility) that:
1. Parses `kitten_fabric_config.json`
2. For each projection:
   - Writes `indptr` → BRAM0
   - Writes `indices` → BRAM1
   - Writes `values_q` → BRAM2
   - Writes `scale_q` → scalar register

After this one-time load at boot, the board is ready to execute steps.

---

## 9. Implementation Notes

- **Time-multiplexing** is allowed:
  - One `csr_projection` block reused for multiple projections
  - One `lif_population` block reused for multiple populations

- The **functional requirement** is:
  - Compute the same forward step as the software `SNNFabricModel`
  - Within reasonable numeric variation from fixed-point quantization

- **CI on hardware** can mimic software gates:
  - Log per-step spike rate, sparsity, non-zero currents
  - Optionally expose debug counters over extra BAR registers

---

## Appendix A: Memory Estimates

### Per-Projection Storage

| Component | Size (bytes)           | BRAMs (18Kb) |
|-----------|------------------------|--------------|
| indptr    | (N_pre + 1) × 4        | 1-2          |
| indices   | nnz × 4                | varies       |
| values_q  | nnz × 2                | varies       |

### Per-Population Storage

| Component  | Size (bytes) | BRAMs (18Kb) |
|------------|--------------|--------------|
| v (membrane)| N × 4       | 2-4          |
| refractory | N × 1        | 0.5          |

### Total Estimates (Kitten)

```
Projections:
  5 projections × ~3 BRAMs each = ~15 BRAMs for CSR data

Populations:
  4 populations × ~3 BRAMs each = ~12 BRAMs for state

Total: ~27 BRAMs minimum (well within Stratix-10 capacity)
```

---

## Appendix B: Performance Targets

| Metric              | Target         |
|---------------------|----------------|
| Latency per step    | < 1 ms         |
| Throughput          | > 1000 steps/s |
| Power               | < 50W          |
| BRAM utilization    | < 50%          |

---

## Appendix C: Validation Checklist

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
