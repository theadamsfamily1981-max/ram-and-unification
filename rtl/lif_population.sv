// =============================================================================
// lif_population.sv
//
// Leaky Integrate-and-Fire (LIF) Neuron Population Update
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// Implements discrete-time LIF dynamics:
//   v_new = alpha * v + I - v_th * spike
//   spike = (v >= v_th) ? 1 : 0
//
// Processes neurons sequentially (time-multiplexed) and outputs packed spikes.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module lif_population #(
    parameter N          = 4096,
    parameter ADDRW      = 12,      // ceil_log2(N)
    parameter PACK_WIDTH = 8,       // Spikes packed per output word
    parameter BRAM_DELAY = 1        // BRAM read latency
) (
    input  wire         clk,
    input  wire         rst,

    // Control
    input  wire         i_start,
    output reg          o_done,

    // LIF parameters (fixed-point Q1.14 or configurable)
    input  wire [15:0]  i_alpha,        // Membrane decay (e.g., 0.95 → ~15564)
    input  wire [15:0]  i_v_th,         // Threshold (e.g., 1.0 → 16384)

    // Input current memory interface (read)
    output reg  [ADDRW-1:0]  o_curr_addr,
    input  wire [31:0]       i_curr_data,    // I[n]

    // Membrane potential memory interface (read/write)
    output reg  [ADDRW-1:0]  o_v_addr,
    output reg               o_v_we,
    output reg  [31:0]       o_v_din,
    input  wire [31:0]       i_v_dout,       // v[n]

    // Packed spike output stream
    output reg               o_spike_valid,
    output reg  [PACK_WIDTH-1:0] o_spike_data,
    input  wire              i_spike_ready
);

    // =========================================================================
    // State Machine
    // =========================================================================
    localparam [2:0] S_IDLE      = 3'd0;
    localparam [2:0] S_ISSUE_RD  = 3'd1;  // Issue BRAM reads for v[n], I[n]
    localparam [2:0] S_WAIT_RD   = 3'd2;  // Wait for BRAM latency
    localparam [2:0] S_COMPUTE   = 3'd3;  // Compute LIF update
    localparam [2:0] S_WRITE_V   = 3'd4;  // Write back v[n]
    localparam [2:0] S_PACK      = 3'd5;  // Accumulate spike into pack buffer
    localparam [2:0] S_EMIT      = 3'd6;  // Emit packed spike word
    localparam [2:0] S_DONE      = 3'd7;

    reg [2:0] state;
    reg [2:0] state_next;

    // Working registers
    reg [ADDRW-1:0]       neuron_idx;       // Current neuron index
    reg [$clog2(PACK_WIDTH)-1:0] pack_idx;  // Position within pack
    reg [PACK_WIDTH-1:0]  spike_pack;       // Spike accumulator
    reg [BRAM_DELAY-1:0]  wait_cnt;

    // LIF computation signals
    reg  [31:0] v_reg;              // Latched membrane potential
    reg  [31:0] I_reg;              // Latched input current
    wire [47:0] v_decay;            // alpha * v (48-bit product)
    wire [31:0] v_decayed;          // Truncated to 32-bit
    wire [31:0] v_new_no_reset;     // v_decayed + I
    wire        spike_out;          // Spike decision
    wire [31:0] v_reset;            // v_th as 32-bit for subtraction
    wire [31:0] v_new;              // Final v after reset

    // Q1.14 fixed-point multiplication
    // v_decay = alpha * v; result is Q1.28, we want Q1.14
    assign v_decay    = $signed({{16{i_alpha[15]}}, i_alpha}) * $signed(v_reg);
    assign v_decayed  = v_decay[45:14];  // Arithmetic right shift by 14

    // Add input current
    assign v_new_no_reset = $signed(v_decayed) + $signed(I_reg);

    // Spike detection: v >= v_th
    // v_th is 16-bit Q1.14, extend to 32-bit
    wire [31:0] v_th_ext;
    assign v_th_ext = {{16{i_v_th[15]}}, i_v_th};
    assign spike_out = ($signed(v_new_no_reset) >= $signed(v_th_ext));

    // Reset: v = v - v_th if spiked (subtract mode)
    assign v_reset = spike_out ? ($signed(v_new_no_reset) - $signed(v_th_ext)) : v_new_no_reset;
    assign v_new = v_reset;

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
                    state_next = S_ISSUE_RD;
                end
            end

            S_ISSUE_RD: begin
                state_next = S_WAIT_RD;
            end

            S_WAIT_RD: begin
                if (wait_cnt == 0) begin
                    state_next = S_COMPUTE;
                end
            end

            S_COMPUTE: begin
                state_next = S_WRITE_V;
            end

            S_WRITE_V: begin
                state_next = S_PACK;
            end

            S_PACK: begin
                // Check if pack is full or we're done with all neurons
                if (pack_idx == PACK_WIDTH - 1 || neuron_idx == N - 1) begin
                    state_next = S_EMIT;
                end else if (neuron_idx < N - 1) begin
                    state_next = S_ISSUE_RD;
                end else begin
                    state_next = S_DONE;
                end
            end

            S_EMIT: begin
                if (i_spike_ready) begin
                    if (neuron_idx >= N - 1) begin
                        state_next = S_DONE;
                    end else begin
                        state_next = S_ISSUE_RD;
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
            o_curr_addr   <= {ADDRW{1'b0}};
            o_v_addr      <= {ADDRW{1'b0}};
            o_v_we        <= 1'b0;
            o_v_din       <= 32'd0;
            o_spike_valid <= 1'b0;
            o_spike_data  <= {PACK_WIDTH{1'b0}};
            neuron_idx    <= {ADDRW{1'b0}};
            pack_idx      <= {$clog2(PACK_WIDTH){1'b0}};
            spike_pack    <= {PACK_WIDTH{1'b0}};
            wait_cnt      <= {BRAM_DELAY{1'b0}};
            v_reg         <= 32'd0;
            I_reg         <= 32'd0;
        end else begin
            case (state)
                S_IDLE: begin
                    o_done        <= 1'b0;
                    o_spike_valid <= 1'b0;
                    o_v_we        <= 1'b0;

                    if (i_start) begin
                        neuron_idx <= {ADDRW{1'b0}};
                        pack_idx   <= {$clog2(PACK_WIDTH){1'b0}};
                        spike_pack <= {PACK_WIDTH{1'b0}};
                    end
                end

                S_ISSUE_RD: begin
                    // Issue parallel reads for v[n] and I[n]
                    o_v_addr    <= neuron_idx;
                    o_curr_addr <= neuron_idx;
                    o_v_we      <= 1'b0;
                    wait_cnt    <= BRAM_DELAY - 1;
                end

                S_WAIT_RD: begin
                    if (wait_cnt > 0) begin
                        wait_cnt <= wait_cnt - 1;
                    end
                end

                S_COMPUTE: begin
                    // Latch BRAM outputs
                    v_reg <= i_v_dout;
                    I_reg <= i_curr_data;
                    // Computation happens combinationally
                end

                S_WRITE_V: begin
                    // Write back new membrane potential
                    o_v_addr <= neuron_idx;
                    o_v_din  <= v_new;
                    o_v_we   <= 1'b1;
                end

                S_PACK: begin
                    o_v_we <= 1'b0;

                    // Add spike to pack buffer
                    spike_pack[pack_idx] <= spike_out;

                    // Check if pack is complete or last neuron
                    if (pack_idx == PACK_WIDTH - 1 || neuron_idx == N - 1) begin
                        // Prepare for emit
                        o_spike_data <= spike_pack;
                        o_spike_data[pack_idx] <= spike_out;  // Include current spike
                    end else begin
                        // Continue to next neuron
                        pack_idx   <= pack_idx + 1;
                        neuron_idx <= neuron_idx + 1;
                    end
                end

                S_EMIT: begin
                    o_spike_valid <= 1'b1;

                    if (i_spike_ready) begin
                        o_spike_valid <= 1'b0;
                        spike_pack    <= {PACK_WIDTH{1'b0}};
                        pack_idx      <= {$clog2(PACK_WIDTH){1'b0}};

                        if (neuron_idx < N - 1) begin
                            neuron_idx <= neuron_idx + 1;
                        end
                    end
                end

                S_DONE: begin
                    o_done        <= 1'b1;
                    o_spike_valid <= 1'b0;
                    o_v_we        <= 1'b0;

                    if (!i_start) begin
                        o_done <= 1'b0;
                    end
                end

                default: begin
                    o_v_we        <= 1'b0;
                    o_spike_valid <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
