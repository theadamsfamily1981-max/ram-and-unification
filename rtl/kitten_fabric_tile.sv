// =============================================================================
// kitten_fabric_tile.sv
//
// One "Kitten" SNN Tile - Top Module
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// Dataflow:
//   pre_spikes → CSR projection → current accumulator → LIF population → post_spikes
//
// Control Phases:
//   1) T_PROJ: Host streams presynaptic spikes. csr_projection computes
//      ΔI updates, current_accumulator folds them into I_post BRAM.
//   2) T_POP: LIF population reads I_post, updates v_mem, clears I_post,
//      and emits packed postsynaptic spikes.
//
// Host drives:
//   - i_step_start: Begin one SNN step
//   - i_proj_done:  No more presynaptic spikes for this step
// Tile reports:
//   - o_step_done:  Step finished (LIF sweep complete)
//
// All BRAMs are external ports for Vivado block RAM instantiation.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module kitten_fabric_tile #(
    parameter N_PRE       = 4096,
    parameter N_POST      = 4096,
    parameter ADDRW_PRE   = 13,       // ceil_log2(N_PRE+1) for indptr
    parameter ADDRW_CSR   = 18,       // ceil_log2(nnz_max) for CSR arrays
    parameter ADDRW_POST  = 12,       // ceil_log2(N_POST) for I_post / v_mem
    parameter PACK_WIDTH  = 8,        // Spikes packed per output word
    parameter BRAM_DELAY  = 1         // BRAM read latency
) (
    input  wire                  clk,
    input  wire                  rst,

    // ========================================================================
    // Step Control (host-visible)
    // ========================================================================
    input  wire                  i_step_start,   // Pulse or level to start a step
    input  wire                  i_proj_done,    // Host says: "no more pre spikes"
    output reg                   o_step_done,    // Tile says: "LIF sweep done"

    // ========================================================================
    // Presynaptic Spike Stream (from host / previous tile)
    // ========================================================================
    input  wire                  i_pre_spike_valid,
    input  wire [15:0]           i_pre_spike_idx,
    input  wire                  i_pre_spike,
    output wire                  o_pre_spike_ready,

    // ========================================================================
    // Postsynaptic Spike Stream (to host / next tile)
    // ========================================================================
    output wire                  o_post_spike_valid,
    output wire [PACK_WIDTH-1:0] o_post_spike_data,
    input  wire                  i_post_spike_ready,

    // ========================================================================
    // CSR Memory: Row Pointers (indptr) - Read Only
    // ========================================================================
    output wire [ADDRW_PRE-1:0]  o_indptr_addr,
    input  wire [31:0]           i_indptr_data,

    // ========================================================================
    // CSR Memory: Column Indices - Read Only
    // ========================================================================
    output wire [ADDRW_CSR-1:0]  o_indices_addr,
    input  wire [31:0]           i_indices_data,

    // ========================================================================
    // CSR Memory: Quantized Values - Read Only
    // ========================================================================
    output wire [ADDRW_CSR-1:0]  o_values_addr,
    input  wire [15:0]           i_values_q_data,

    // ========================================================================
    // Per-Projection Quantization Scale
    // ========================================================================
    input  wire [15:0]           i_scale_q,

    // ========================================================================
    // I_post Memory (shared: accumulator writes, LIF reads/clears)
    // ========================================================================
    output wire [ADDRW_POST-1:0] o_I_addr,
    output wire                  o_I_we,
    output wire [31:0]           o_I_din,
    input  wire [31:0]           i_I_dout,

    // ========================================================================
    // v_mem Memory (owned by LIF population)
    // ========================================================================
    output wire [ADDRW_POST-1:0] o_v_addr,
    output wire                  o_v_we,
    output wire [31:0]           o_v_din,
    input  wire [31:0]           i_v_dout,

    // ========================================================================
    // LIF Parameters
    // ========================================================================
    input  wire [15:0]           i_alpha,        // Leak factor (Q1.14)
    input  wire [15:0]           i_v_th          // Threshold (Q1.14)
);

    // ========================================================================
    // Tile Phase FSM
    // ========================================================================
    localparam [1:0] T_IDLE = 2'd0;
    localparam [1:0] T_PROJ = 2'd1;
    localparam [1:0] T_POP  = 2'd2;
    localparam [1:0] T_DONE = 2'd3;

    reg [1:0] state, state_next;

    // Submodule control signals
    reg  proj_start;
    reg  pop_start;
    wire lif_done;

    // ========================================================================
    // CSR Projection → Current Accumulator Wires
    // ========================================================================
    wire        proj_curr_valid;
    wire [15:0] proj_curr_idx;
    wire [31:0] proj_curr_value;
    wire        acc_curr_ready;

    // ========================================================================
    // I_post Memory Mux: Accumulator (PROJ phase) vs LIF (POP phase)
    // ========================================================================
    // Accumulator side
    wire [ADDRW_POST-1:0] I_addr_acc;
    wire                  I_we_acc;
    wire [31:0]           I_din_acc;

    // LIF side
    wire [ADDRW_POST-1:0] I_addr_lif;
    wire                  I_we_lif;
    wire [31:0]           I_din_lif;

    // Mux: In T_POP phase, LIF owns I_post; otherwise accumulator
    assign o_I_addr = (state == T_POP) ? I_addr_lif : I_addr_acc;
    assign o_I_we   = (state == T_POP) ? I_we_lif   : I_we_acc;
    assign o_I_din  = (state == T_POP) ? I_din_lif  : I_din_acc;

    // ========================================================================
    // Instantiate: csr_projection
    // ========================================================================
    csr_projection #(
        .N_PRE      (N_PRE),
        .N_POST     (N_POST),
        .ADDRW_R    (ADDRW_PRE),
        .ADDRW_C    (ADDRW_CSR),
        .BRAM_DELAY (BRAM_DELAY)
    ) u_csr_projection (
        .clk            (clk),
        .rst            (rst),

        // Control
        .i_start        (proj_start),
        .o_done         (/* unused - host gates via i_proj_done */),

        // Presynaptic spike input
        .i_spike_valid  (i_pre_spike_valid),
        .i_spike        (i_pre_spike),
        .i_spike_idx    (i_pre_spike_idx),
        .o_spike_ready  (o_pre_spike_ready),

        // CSR BRAM interfaces
        .o_indptr_addr  (o_indptr_addr),
        .i_indptr_data  (i_indptr_data),

        .o_indices_addr (o_indices_addr),
        .i_indices_data (i_indices_data),

        .o_values_addr  (o_values_addr),
        .i_values_q_data(i_values_q_data),

        .i_scale_q      (i_scale_q),

        // Output to accumulator
        .o_curr_valid   (proj_curr_valid),
        .o_curr_idx     (proj_curr_idx),
        .o_curr_value   (proj_curr_value),
        .i_curr_ready   (acc_curr_ready)
    );

    // ========================================================================
    // Instantiate: current_accumulator
    // ========================================================================
    current_accumulator #(
        .N_POST     (N_POST),
        .ADDRW      (ADDRW_POST),
        .BRAM_DELAY (BRAM_DELAY)
    ) u_current_accumulator (
        .clk         (clk),
        .rst         (rst),

        // Input from csr_projection
        .i_curr_valid(proj_curr_valid),
        .i_curr_idx  (proj_curr_idx),
        .i_curr_value(proj_curr_value),
        .o_curr_ready(acc_curr_ready),

        // I_post memory interface
        .o_mem_addr  (I_addr_acc),
        .o_mem_we    (I_we_acc),
        .o_mem_din   (I_din_acc),
        .i_mem_dout  (i_I_dout),

        // Clear control (not used - LIF clears I_post)
        .i_clear     (1'b0),
        .o_clear_done(/* unused */)
    );

    // ========================================================================
    // Instantiate: lif_population
    // ========================================================================
    lif_population #(
        .N          (N_POST),
        .ADDRW      (ADDRW_POST),
        .PACK_WIDTH (PACK_WIDTH),
        .BRAM_DELAY (BRAM_DELAY)
    ) u_lif_population (
        .clk            (clk),
        .rst            (rst),

        // Control
        .i_start        (pop_start),
        .o_done         (lif_done),

        // LIF parameters
        .i_alpha        (i_alpha),
        .i_v_th         (i_v_th),

        // I_post memory (read + clear)
        .o_curr_addr    (I_addr_lif),
        .o_curr_we      (I_we_lif),
        .o_curr_din     (I_din_lif),
        .i_curr_data    (i_I_dout),

        // v_mem memory
        .o_v_addr       (o_v_addr),
        .o_v_we         (o_v_we),
        .o_v_din        (o_v_din),
        .i_v_dout       (i_v_dout),

        // Spike output stream
        .o_spike_valid  (o_post_spike_valid),
        .o_spike_data   (o_post_spike_data),
        .i_spike_ready  (i_post_spike_ready)
    );

    // ========================================================================
    // Tile FSM: State Register
    // ========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= T_IDLE;
        end else begin
            state <= state_next;
        end
    end

    // ========================================================================
    // Tile FSM: Next State + Output Logic
    // ========================================================================
    always @(*) begin
        state_next  = state;
        proj_start  = 1'b0;
        pop_start   = 1'b0;
        o_step_done = 1'b0;

        case (state)
            T_IDLE: begin
                if (i_step_start) begin
                    state_next = T_PROJ;
                end
            end

            T_PROJ: begin
                // Projection phase: csr_projection + accumulator running
                proj_start = 1'b1;

                // When host asserts i_proj_done, transition to POP phase
                if (i_proj_done) begin
                    state_next = T_POP;
                end
            end

            T_POP: begin
                // Population phase: LIF update sweep
                pop_start = 1'b1;

                if (lif_done) begin
                    state_next = T_DONE;
                end
            end

            T_DONE: begin
                o_step_done = 1'b1;

                // Wait for host to drop i_step_start to re-arm
                if (!i_step_start) begin
                    state_next = T_IDLE;
                end
            end

            default: begin
                state_next = T_IDLE;
            end
        endcase
    end

endmodule

`default_nettype wire
