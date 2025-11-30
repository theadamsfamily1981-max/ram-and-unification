# CSR Projection Engine RTL Specification

**Version**: v0.1
**Status**: Detailed cycle-level spec for FPGA implementation

---

## 1. Purpose

The `csr_projection` module implements sparse matrix-vector multiplication for SNN projections.

Given:
- A **spike vector** `s_pre ∈ {0,1}^{N_pre}` (presynaptic spikes)
- CSR representation of `W ∈ ℝ^{N_post×N_pre}` (weight matrix)

Compute:
```
I_post[i] += Σ_j W[i, j] * s_pre[j]
```

In CSR row-major form (iterating over presynaptic neurons):
```python
for each presynaptic neuron j:
    if s_pre[j] == 1:
        for k in indptr[j] .. indptr[j+1]-1:
            i = indices[k]
            I_post[i] += dequant(values_q[k])
```

---

## 2. Interface

Assuming time-multiplexed over presynaptic neurons.

```systemverilog
module csr_projection #(
    parameter N_PRE   = 4096,
    parameter N_POST  = 4096,
    parameter ADDRW_R = 13,  // ceil_log2(N_PRE+1) for indptr
    parameter ADDRW_C = 18   // ceil_log2(nnz_max) for indices/values
) (
    input  wire              clk,
    input  wire              rst,

    // Control
    input  wire              i_start,
    output reg               o_done,

    // Presynaptic spikes: sequential access (one j per cycle)
    input  wire              i_spike_valid,
    input  wire              i_spike,          // 1-bit spike for neuron j
    input  wire [15:0]       i_spike_idx,      // j
    output reg               o_spike_ready,

    // CSR arrays in BRAMs (read-only)
    output reg  [ADDRW_R-1:0] o_indptr_addr,
    input  wire [31:0]        i_indptr_data,   // indptr[j] or indptr[j+1]

    output reg  [ADDRW_C-1:0] o_indices_addr,
    input  wire [31:0]        i_indices_data,  // indices[k]

    output reg  [ADDRW_C-1:0] o_values_addr,
    input  wire [15:0]        i_values_q_data, // values_q[k]

    input  wire [15:0]        i_scale_q,       // fixed-point scale param

    // Output: postsynaptic current updates (stream)
    output reg                o_curr_valid,
    output reg [31:0]         o_curr_value,    // ΔI
    output reg [15:0]         o_curr_idx,      // postsynaptic neuron index i
    input  wire               i_curr_ready
);
```

Downstream, a `current_accumulator` module will fold these `(idx, ΔI)` pairs into `I_post[i]`.

---

## 3. State Machine

### 3.1 States

```
IDLE              // Wait for i_start
NEXT_SPIKE        // Accept next presynaptic spike
READ_ROW_PTRS     // Fetch indptr[j], indptr[j+1]
CHECK_SPIKE       // If s_pre[j]==0 → skip row, else iterate
ITERATE_COLS      // Loop over k from row_start to row_end-1
WAIT_ACC_READY    // Backpressure when i_curr_ready=0
DONE              // Signal completion
```

### 3.2 Registers

```systemverilog
reg [2:0]  state;
reg [15:0] curr_j;            // Current presynaptic index
reg        curr_spike;        // Spike bit for curr_j
reg [31:0] row_start;         // indptr[j]
reg [31:0] row_end;           // indptr[j+1]
reg [31:0] k;                 // CSR index iterator
reg        indptr_phase;      // 0: reading indptr[j], 1: reading indptr[j+1]
```

---

## 4. Cycle-by-Cycle Behavior

### 4.1 IDLE

```systemverilog
always @(posedge clk or posedge rst) begin
    if (rst) begin
        state <= IDLE;
        o_done <= 0;
        o_curr_valid <= 0;
        o_spike_ready <= 0;
    end
    else if (state == IDLE) begin
        if (i_start) begin
            o_done <= 0;
            o_curr_valid <= 0;
            o_spike_ready <= 1;  // Ready to accept spikes
            state <= NEXT_SPIKE;
        end
    end
end
```

### 4.2 NEXT_SPIKE

Accept the next presynaptic spike from the input stream.

```systemverilog
// In NEXT_SPIKE state:
if (i_spike_valid && o_spike_ready) begin
    curr_j <= i_spike_idx;
    curr_spike <= i_spike;
    o_spike_ready <= 0;
    indptr_phase <= 0;
    state <= READ_ROW_PTRS;
end
else if (!i_spike_valid) begin
    // No more spikes - done with this projection
    state <= DONE;
end
```

### 4.3 READ_ROW_PTRS

Two-phase read of `indptr[j]` and `indptr[j+1]`.

```systemverilog
// Phase 0: Read indptr[j]
if (indptr_phase == 0) begin
    o_indptr_addr <= curr_j;
    indptr_phase <= 1;
end
// Phase 1: Latch indptr[j], read indptr[j+1]
else if (indptr_phase == 1) begin
    row_start <= i_indptr_data;
    o_indptr_addr <= curr_j + 1;
    indptr_phase <= 2;
end
// Phase 2: Latch indptr[j+1], go to CHECK_SPIKE
else begin
    row_end <= i_indptr_data;
    state <= CHECK_SPIKE;
end
```

**Note**: This requires 3 cycles. Can be pipelined to 2 cycles with dual-port BRAM or careful address scheduling.

### 4.4 CHECK_SPIKE

```systemverilog
if (curr_spike == 0) begin
    // No spike - skip this row
    o_spike_ready <= 1;
    state <= NEXT_SPIKE;
end
else if (row_start == row_end) begin
    // Empty row (no connections)
    o_spike_ready <= 1;
    state <= NEXT_SPIKE;
end
else begin
    // Spike and non-empty row - iterate columns
    k <= row_start;
    state <= ITERATE_COLS;
end
```

### 4.5 ITERATE_COLS

For each non-zero synapse in the row:

```systemverilog
// Set BRAM addresses for indices[k] and values[k]
o_indices_addr <= k[ADDRW_C-1:0];
o_values_addr <= k[ADDRW_C-1:0];

// Wait one cycle for BRAM read
// (in pipelined version, overlap reads)

// After BRAM latency:
// i_indices_data = indices[k] (postsynaptic neuron index)
// i_values_q_data = values_q[k] (quantized weight)

// Compute dequantized weight: ΔI = values_q * scale
// For Q1.14 format: ΔI = (values_q * scale_q) >> 14
wire [31:0] delta_I;
assign delta_I = ($signed(i_values_q_data) * $signed({1'b0, i_scale_q})) >>> 14;

// Emit current update
if (i_curr_ready) begin
    o_curr_valid <= 1;
    o_curr_idx <= i_indices_data[15:0];
    o_curr_value <= delta_I;

    // Advance k
    k <= k + 1;

    if (k + 1 >= row_end) begin
        // Row finished
        o_curr_valid <= 0;
        o_spike_ready <= 1;
        state <= NEXT_SPIKE;
    end
end
else begin
    // Backpressure - wait for accumulator
    state <= WAIT_ACC_READY;
end
```

### 4.6 WAIT_ACC_READY

```systemverilog
// Keep output stable
if (i_curr_ready) begin
    o_curr_valid <= 0;

    // Advance k
    k <= k + 1;

    if (k + 1 >= row_end) begin
        o_spike_ready <= 1;
        state <= NEXT_SPIKE;
    end
    else begin
        state <= ITERATE_COLS;
    end
end
// Else stay in WAIT_ACC_READY
```

### 4.7 DONE

```systemverilog
o_done <= 1;
o_curr_valid <= 0;
o_spike_ready <= 0;

// Wait for i_start to deassert, then return to IDLE
if (!i_start) begin
    o_done <= 0;
    state <= IDLE;
end
```

---

## 5. Timing Diagram

```
Cycle:  1    2    3    4    5    6    7    8    9   10   11
State:  IDLE NEXT READ READ READ CHK  ITER ITER ITER NEXT ...
             SPIK PTR0 PTR1 PTR2

i_start:  1    -    -    -    -    -    -    -    -    -
spike_v:  -    1    -    -    -    -    -    -    -    1
spike:    -    1    -    -    -    -    -    -    -    0
spike_i:  -    0    -    -    -    -    -    -    -    1
spike_r:  -    1    0    0    0    0    0    0    0    1

indptr_a: -    -    0    1    -    -    -    -    -    -
indptr_d: -    -    -    100  105  -    -    -    -    -

row_start:-    -    -    -    100  100  100  100  100  -
row_end:  -    -    -    -    -    105  105  105  105  -
k:        -    -    -    -    -    -    100  101  102  -

indices_a:-    -    -    -    -    -    100  101  102  -
values_a: -    -    -    -    -    -    100  101  102  -

curr_v:   0    0    0    0    0    0    0    1    1    0
curr_idx: -    -    -    -    -    -    -    42   87   -
curr_val: -    -    -    -    -    -    -   0.5  0.3   -
```

---

## 6. Complexity Analysis

For one presynaptic row with:
- `nnz_row = indptr[j+1] - indptr[j]` nonzeros

Cycles per row:
- 1 cycle: Accept spike
- 3 cycles: Read row pointers
- 1 cycle: Check spike
- `nnz_row` cycles: Iterate columns (assuming no backpressure)

**Total**: ~5 + nnz_row cycles per spiking neuron

For Kitten with k=64 max degree:
- Worst case per row: ~70 cycles
- With 10% spike rate, ~400 spiking neurons per population
- Total per projection: ~28,000 cycles
- At 200 MHz: ~140 μs per projection
- 5 projections: ~700 μs (well under 1 ms target)

---

## 7. Downstream Accumulator

The `current_accumulator` module receives `(idx, ΔI)` pairs and accumulates them:

```systemverilog
module current_accumulator #(
    parameter N_POST = 4096
) (
    input  wire         clk,
    input  wire         rst,

    // Input stream from csr_projection
    input  wire         i_curr_valid,
    input  wire [15:0]  i_curr_idx,
    input  wire [31:0]  i_curr_value,
    output wire         o_curr_ready,

    // Memory interface for I_post
    output reg  [15:0]  o_mem_addr,
    output reg          o_mem_we,
    output reg  [31:0]  o_mem_din,
    input  wire [31:0]  i_mem_dout
);
```

### 7.1 Accumulator State Machine

```
IDLE
READ_CURR     // Read I_post[idx] from BRAM
ADD           // Add ΔI to I_post[idx]
WRITE_BACK    // Write updated value
```

### 7.2 Accumulator Logic

```systemverilog
reg [1:0] acc_state;
reg [15:0] pending_idx;
reg [31:0] pending_delta;

// Ready when not in middle of read-modify-write
assign o_curr_ready = (acc_state == IDLE);

always @(posedge clk) begin
    case (acc_state)
        IDLE: begin
            if (i_curr_valid) begin
                pending_idx <= i_curr_idx;
                pending_delta <= i_curr_value;
                o_mem_addr <= i_curr_idx;
                o_mem_we <= 0;
                acc_state <= READ_CURR;
            end
        end

        READ_CURR: begin
            // BRAM read latency - wait one cycle
            acc_state <= ADD;
        end

        ADD: begin
            // i_mem_dout now has I_post[idx]
            o_mem_din <= i_mem_dout + pending_delta;
            o_mem_addr <= pending_idx;
            o_mem_we <= 1;
            acc_state <= WRITE_BACK;
        end

        WRITE_BACK: begin
            o_mem_we <= 0;
            acc_state <= IDLE;
        end
    endcase
end
```

---

## 8. Optimization Notes

### 8.1 Pipelining

The basic design can be pipelined to overlap:
- BRAM reads for next k while emitting current k
- Multiple projections sharing one accumulator bank

### 8.2 Parallel Accumulation

For higher throughput:
- Use multiple accumulator banks (4-8)
- Hash `idx` to select bank
- Reduces conflicts for random access patterns

### 8.3 SIMD/Vectorization

If DSP resources allow:
- Process 4-8 synapses per cycle
- Requires wider BRAM ports or multi-bank

---

## 9. Test Vectors

### 9.1 Simple Test Case

```python
# 4x4 matrix, k=2 per row
N_pre = 4
N_post = 4

indptr = [0, 2, 4, 6, 8]  # 2 connections per row
indices = [0, 2, 1, 3, 0, 2, 1, 3]  # pattern
values_q = [100, 200, 150, 250, 175, 225, 125, 275]  # Q1.14
scale_q = 16384  # = 1.0 in Q1.14

# Spike vector: neurons 0 and 2 spike
s_pre = [1, 0, 1, 0]

# Expected output:
# I_post[0] += values_q[0] + values_q[4] = 100 + 175 = 275
# I_post[1] += values_q[6] = 125
# I_post[2] += values_q[1] + values_q[5] = 200 + 225 = 425
# I_post[3] += values_q[7] = 275
```

### 9.2 Verification

1. Initialize BRAM with test CSR data
2. Feed spike sequence: `(0, 1), (1, 0), (2, 1), (3, 0)`
3. Capture output stream
4. Verify accumulated I_post matches expected

---

## 10. Integration with Fabric Tile

The `csr_projection` module is instantiated for each projection:

```systemverilog
// In kitten_fabric_tile:

// Projection 0: input → hidden1
csr_projection #(.N_PRE(4096), .N_POST(4096)) proj_in_h1 (
    .clk(clk),
    .rst(rst),
    .i_start(proj_start[0]),
    .o_done(proj_done[0]),
    .i_spike_valid(input_spike_valid),
    .i_spike(input_spike),
    .i_spike_idx(input_spike_idx),
    .o_spike_ready(proj_spike_ready[0]),
    // BRAM interfaces...
    .o_curr_valid(h1_curr_valid),
    .o_curr_value(h1_curr_value),
    .o_curr_idx(h1_curr_idx),
    .i_curr_ready(h1_acc_ready)
);

// Similar for other projections...
```

Time-multiplexed version would share one `csr_projection` instance across all projections, switching BRAM banks between steps.
