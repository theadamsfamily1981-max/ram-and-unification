// =============================================================================
// current_accumulator.sv
//
// Postsynaptic Current Accumulator for SNN Projections
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// Receives (idx, ΔI) pairs from csr_projection and performs:
//   I_post[idx] += ΔI
// Using read-modify-write to BRAM
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module current_accumulator #(
    parameter N_POST     = 4096,
    parameter ADDRW      = 12,      // ceil_log2(N_POST)
    parameter BRAM_DELAY = 1        // BRAM read latency in cycles
) (
    input  wire         clk,
    input  wire         rst,

    // Input stream from csr_projection
    input  wire         i_curr_valid,
    input  wire [15:0]  i_curr_idx,
    input  wire [31:0]  i_curr_value,   // ΔI (signed)
    output wire         o_curr_ready,

    // Memory interface for I_post BRAM
    output reg  [ADDRW-1:0]  o_mem_addr,
    output reg               o_mem_we,
    output reg  [31:0]       o_mem_din,
    input  wire [31:0]       i_mem_dout,

    // Control
    input  wire         i_clear,        // Clear all I_post to zero
    output reg          o_clear_done
);

    // =========================================================================
    // State Machine
    // =========================================================================
    localparam [2:0] S_IDLE       = 3'd0;
    localparam [2:0] S_READ_CURR  = 3'd1;  // Issue read for I_post[idx]
    localparam [2:0] S_WAIT_READ  = 3'd2;  // Wait for BRAM latency
    localparam [2:0] S_ADD        = 3'd3;  // Compute I_post[idx] + ΔI
    localparam [2:0] S_WRITE_BACK = 3'd4;  // Write result
    localparam [2:0] S_CLEAR      = 3'd5;  // Clear loop

    reg [2:0] state;
    reg [2:0] state_next;

    // Working registers
    reg [15:0]       pending_idx;
    reg [31:0]       pending_delta;
    reg [ADDRW-1:0]  clear_addr;
    reg [BRAM_DELAY-1:0] wait_cnt;

    // Ready when idle
    assign o_curr_ready = (state == S_IDLE) && !i_clear;

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
                if (i_clear) begin
                    state_next = S_CLEAR;
                end else if (i_curr_valid) begin
                    state_next = S_READ_CURR;
                end
            end

            S_READ_CURR: begin
                state_next = S_WAIT_READ;
            end

            S_WAIT_READ: begin
                if (wait_cnt == 0) begin
                    state_next = S_ADD;
                end
            end

            S_ADD: begin
                state_next = S_WRITE_BACK;
            end

            S_WRITE_BACK: begin
                state_next = S_IDLE;
            end

            S_CLEAR: begin
                if (clear_addr == N_POST - 1) begin
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
            o_mem_addr    <= {ADDRW{1'b0}};
            o_mem_we      <= 1'b0;
            o_mem_din     <= 32'd0;
            o_clear_done  <= 1'b0;
            pending_idx   <= 16'd0;
            pending_delta <= 32'd0;
            clear_addr    <= {ADDRW{1'b0}};
            wait_cnt      <= {BRAM_DELAY{1'b0}};
        end else begin
            case (state)
                S_IDLE: begin
                    o_mem_we     <= 1'b0;
                    o_clear_done <= 1'b0;

                    if (i_clear) begin
                        // Start clear sequence
                        clear_addr <= {ADDRW{1'b0}};
                        o_mem_addr <= {ADDRW{1'b0}};
                        o_mem_din  <= 32'd0;
                        o_mem_we   <= 1'b1;
                    end else if (i_curr_valid) begin
                        // Latch input and issue read
                        pending_idx   <= i_curr_idx;
                        pending_delta <= i_curr_value;
                        o_mem_addr    <= i_curr_idx[ADDRW-1:0];
                        o_mem_we      <= 1'b0;
                    end
                end

                S_READ_CURR: begin
                    // Wait for BRAM read
                    wait_cnt <= BRAM_DELAY - 1;
                end

                S_WAIT_READ: begin
                    if (wait_cnt > 0) begin
                        wait_cnt <= wait_cnt - 1;
                    end
                end

                S_ADD: begin
                    // Compute sum: I_post[idx] + ΔI
                    // Signed addition
                    o_mem_din  <= $signed(i_mem_dout) + $signed(pending_delta);
                    o_mem_addr <= pending_idx[ADDRW-1:0];
                    o_mem_we   <= 1'b1;
                end

                S_WRITE_BACK: begin
                    // Write is happening this cycle
                    o_mem_we <= 1'b0;
                end

                S_CLEAR: begin
                    // Write zero to current address
                    o_mem_we  <= 1'b1;
                    o_mem_din <= 32'd0;

                    if (clear_addr == N_POST - 1) begin
                        o_clear_done <= 1'b1;
                        o_mem_we     <= 1'b0;
                    end else begin
                        clear_addr <= clear_addr + 1;
                        o_mem_addr <= clear_addr + 1;
                    end
                end

                default: begin
                    o_mem_we <= 1'b0;
                end
            endcase
        end
    end

endmodule

`default_nettype wire
