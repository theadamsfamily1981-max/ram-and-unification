// =============================================================================
// csr_projection.sv
//
// CSR Sparse Matrix-Vector Multiply Engine for SNN Projections
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// Computes: I_post[i] += Σ_j W[i,j] * s_pre[j]  for spiking neurons j
// Using CSR format: indptr, indices, values_q arrays
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module csr_projection #(
    parameter N_PRE      = 4096,
    parameter N_POST     = 4096,
    parameter ADDRW_R    = 13,      // ceil_log2(N_PRE+1) for indptr
    parameter ADDRW_C    = 18,      // ceil_log2(nnz_max) for indices/values
    parameter BRAM_DELAY = 1        // BRAM read latency in cycles
) (
    input  wire              clk,
    input  wire              rst,

    // Control
    input  wire              i_start,
    output reg               o_done,

    // Presynaptic spikes: sequential access (one j per cycle)
    input  wire              i_spike_valid,
    input  wire              i_spike,            // 1-bit spike for neuron j
    input  wire [15:0]       i_spike_idx,        // j
    output reg               o_spike_ready,

    // CSR arrays in BRAMs (read-only)
    output reg  [ADDRW_R-1:0] o_indptr_addr,
    input  wire [31:0]        i_indptr_data,     // indptr[j] or indptr[j+1]

    output reg  [ADDRW_C-1:0] o_indices_addr,
    input  wire [31:0]        i_indices_data,    // indices[k]

    output reg  [ADDRW_C-1:0] o_values_addr,
    input  wire [15:0]        i_values_q_data,   // values_q[k] (Q1.14)

    input  wire [15:0]        i_scale_q,         // fixed-point scale param

    // Output: postsynaptic current updates (stream)
    output reg               o_curr_valid,
    output reg  [31:0]       o_curr_value,       // ΔI (signed)
    output reg  [15:0]       o_curr_idx,         // postsynaptic neuron index i
    input  wire              i_curr_ready
);

    // =========================================================================
    // State Machine
    // =========================================================================
    localparam [3:0] S_IDLE          = 4'd0;
    localparam [3:0] S_NEXT_SPIKE    = 4'd1;
    localparam [3:0] S_READ_PTR0     = 4'd2;  // Issue read for indptr[j]
    localparam [3:0] S_READ_PTR1     = 4'd3;  // Latch indptr[j], issue indptr[j+1]
    localparam [3:0] S_READ_PTR2     = 4'd4;  // Latch indptr[j+1]
    localparam [3:0] S_CHECK_SPIKE   = 4'd5;
    localparam [3:0] S_ITER_ISSUE    = 4'd6;  // Issue BRAM read for indices[k], values[k]
    localparam [3:0] S_ITER_WAIT     = 4'd7;  // Wait for BRAM latency
    localparam [3:0] S_ITER_EMIT     = 4'd8;  // Emit current update
    localparam [3:0] S_WAIT_ACC      = 4'd9;  // Backpressure wait
    localparam [3:0] S_DONE          = 4'd10;

    reg [3:0]   state;
    reg [3:0]   state_next;

    // Working registers
    reg [15:0]  curr_j;             // Current presynaptic index
    reg         curr_spike;         // Spike bit for curr_j
    reg [31:0]  row_start;          // indptr[j]
    reg [31:0]  row_end;            // indptr[j+1]
    reg [31:0]  k;                  // CSR index iterator
    reg [BRAM_DELAY-1:0] wait_cnt;  // BRAM latency counter

    // Dequantized weight computation
    // Q1.14 format: ΔI = (values_q * scale_q) >>> 14
    wire signed [31:0] values_q_ext;
    wire signed [31:0] scale_q_ext;
    wire signed [47:0] product;
    wire signed [31:0] delta_I;

    assign values_q_ext = {{16{i_values_q_data[15]}}, i_values_q_data};  // Sign extend
    assign scale_q_ext  = {16'b0, i_scale_q};                            // Zero extend (unsigned scale)
    assign product      = values_q_ext * scale_q_ext;
    assign delta_I      = product[45:14];  // Arithmetic right shift by 14

    // =========================================================================
    // State Register
    // =========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= S_IDLE;
        end else begin
            state <= state_next;
        end
    end

    // =========================================================================
    // Next State Logic
    // =========================================================================
    always @(*) begin
        state_next = state;

        case (state)
            S_IDLE: begin
                if (i_start) begin
                    state_next = S_NEXT_SPIKE;
                end
            end

            S_NEXT_SPIKE: begin
                if (i_spike_valid && o_spike_ready) begin
                    state_next = S_READ_PTR0;
                end else if (!i_spike_valid) begin
                    state_next = S_DONE;
                end
            end

            S_READ_PTR0: begin
                state_next = S_READ_PTR1;
            end

            S_READ_PTR1: begin
                state_next = S_READ_PTR2;
            end

            S_READ_PTR2: begin
                state_next = S_CHECK_SPIKE;
            end

            S_CHECK_SPIKE: begin
                if (!curr_spike || (row_start == row_end)) begin
                    state_next = S_NEXT_SPIKE;
                end else begin
                    state_next = S_ITER_ISSUE;
                end
            end

            S_ITER_ISSUE: begin
                state_next = S_ITER_WAIT;
            end

            S_ITER_WAIT: begin
                if (wait_cnt == 0) begin
                    state_next = S_ITER_EMIT;
                end
            end

            S_ITER_EMIT: begin
                if (i_curr_ready) begin
                    if (k + 1 >= row_end) begin
                        state_next = S_NEXT_SPIKE;
                    end else begin
                        state_next = S_ITER_ISSUE;
                    end
                end else begin
                    state_next = S_WAIT_ACC;
                end
            end

            S_WAIT_ACC: begin
                if (i_curr_ready) begin
                    if (k + 1 >= row_end) begin
                        state_next = S_NEXT_SPIKE;
                    end else begin
                        state_next = S_ITER_ISSUE;
                    end
                end
            end

            S_DONE: begin
                if (!i_start) begin
                    state_next = S_IDLE;
                end
            end

            default: state_next = S_IDLE;
        endcase
    end

    // =========================================================================
    // Datapath
    // =========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            o_done        <= 1'b0;
            o_spike_ready <= 1'b0;
            o_curr_valid  <= 1'b0;
            o_curr_value  <= 32'd0;
            o_curr_idx    <= 16'd0;
            o_indptr_addr <= {ADDRW_R{1'b0}};
            o_indices_addr<= {ADDRW_C{1'b0}};
            o_values_addr <= {ADDRW_C{1'b0}};
            curr_j        <= 16'd0;
            curr_spike    <= 1'b0;
            row_start     <= 32'd0;
            row_end       <= 32'd0;
            k             <= 32'd0;
            wait_cnt      <= {BRAM_DELAY{1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    o_done        <= 1'b0;
                    o_curr_valid  <= 1'b0;
                    o_spike_ready <= 1'b0;
                    if (i_start) begin
                        o_spike_ready <= 1'b1;
                    end
                end

                S_NEXT_SPIKE: begin
                    o_curr_valid <= 1'b0;
                    if (i_spike_valid && o_spike_ready) begin
                        curr_j        <= i_spike_idx;
                        curr_spike    <= i_spike;
                        o_spike_ready <= 1'b0;
                        // Issue first indptr read
                        o_indptr_addr <= i_spike_idx[ADDRW_R-1:0];
                    end else if (!i_spike_valid) begin
                        o_spike_ready <= 1'b0;
                    end
                end

                S_READ_PTR0: begin
                    // Wait for BRAM; issue read for indptr[j+1]
                    // Nothing to latch yet
                end

                S_READ_PTR1: begin
                    // Latch indptr[j]
                    row_start     <= i_indptr_data;
                    // Issue read for indptr[j+1]
                    o_indptr_addr <= curr_j[ADDRW_R-1:0] + 1'b1;
                end

                S_READ_PTR2: begin
                    // Latch indptr[j+1]
                    row_end <= i_indptr_data;
                end

                S_CHECK_SPIKE: begin
                    if (!curr_spike || (row_start == row_end)) begin
                        // Skip this row
                        o_spike_ready <= 1'b1;
                    end else begin
                        // Initialize iterator
                        k <= row_start;
                    end
                end

                S_ITER_ISSUE: begin
                    // Issue BRAM reads for indices[k] and values[k]
                    o_indices_addr <= k[ADDRW_C-1:0];
                    o_values_addr  <= k[ADDRW_C-1:0];
                    wait_cnt       <= BRAM_DELAY - 1;
                end

                S_ITER_WAIT: begin
                    if (wait_cnt > 0) begin
                        wait_cnt <= wait_cnt - 1;
                    end
                end

                S_ITER_EMIT: begin
                    // BRAM data is now valid
                    o_curr_valid <= 1'b1;
                    o_curr_idx   <= i_indices_data[15:0];
                    o_curr_value <= delta_I;

                    if (i_curr_ready) begin
                        k <= k + 1;
                        if (k + 1 >= row_end) begin
                            o_curr_valid  <= 1'b0;
                            o_spike_ready <= 1'b1;
                        end
                    end
                end

                S_WAIT_ACC: begin
                    // Keep output stable during backpressure
                    if (i_curr_ready) begin
                        o_curr_valid <= 1'b0;
                        k <= k + 1;
                        if (k + 1 >= row_end) begin
                            o_spike_ready <= 1'b1;
                        end
                    end
                end

                S_DONE: begin
                    o_done        <= 1'b1;
                    o_curr_valid  <= 1'b0;
                    o_spike_ready <= 1'b0;
                    if (!i_start) begin
                        o_done <= 1'b0;
                    end
                end

                default: begin
                    // Reset to safe state
                    o_done        <= 1'b0;
                    o_curr_valid  <= 1'b0;
                    o_spike_ready <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
