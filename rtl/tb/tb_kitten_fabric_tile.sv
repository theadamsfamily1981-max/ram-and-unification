// =============================================================================
// tb_kitten_fabric_tile.sv
//
// Minimal Testbench for kitten_fabric_tile
// Tests a 4×4 identity network with all weights = 1
//
// Ara-SYNERGY Kitten Fabric - Simulation
// =============================================================================

`timescale 1ns/1ps

module tb_kitten_fabric_tile;

    // ========================================================================
    // Parameters for tiny test
    // ========================================================================
    localparam int N_PRE      = 4;
    localparam int N_POST     = 4;
    localparam int ADDRW_PRE  = 3;   // enough for N_PRE+1 = 5
    localparam int ADDRW_CSR  = 3;   // enough for nnz=4
    localparam int ADDRW_POST = 2;   // enough for 4 neurons
    localparam int V_WIDTH    = 32;
    localparam int I_WIDTH    = 32;
    localparam int PACK_WIDTH = 4;

    // ========================================================================
    // Clock / Reset
    // ========================================================================
    logic clk;
    logic rst;

    initial begin
        clk = 0;
        forever #5 clk = ~clk;  // 100 MHz
    end

    initial begin
        rst = 1;
        #50;
        rst = 0;
    end

    // ========================================================================
    // DUT IO
    // ========================================================================
    logic                  i_step_start;
    logic                  i_proj_done;
    logic                  o_step_done;

    logic                  i_pre_spike_valid;
    logic [15:0]           i_pre_spike_idx;
    logic                  i_pre_spike;
    logic                  o_pre_spike_ready;

    logic                  o_post_spike_valid;
    logic [PACK_WIDTH-1:0] o_post_spike_data;
    logic                  i_post_spike_ready;

    logic [ADDRW_PRE-1:0]  o_indptr_addr;
    logic [31:0]           i_indptr_data;

    logic [ADDRW_CSR-1:0]  o_indices_addr;
    logic [31:0]           i_indices_data;

    logic [ADDRW_CSR-1:0]  o_values_addr;
    logic [15:0]           i_values_q_data;

    logic [15:0]           i_scale_q;

    logic [ADDRW_POST-1:0] o_I_addr;
    logic                  o_I_we;
    logic [I_WIDTH-1:0]    o_I_din;
    logic [I_WIDTH-1:0]    i_I_dout;

    logic [ADDRW_POST-1:0] o_v_addr;
    logic                  o_v_we;
    logic [V_WIDTH-1:0]    o_v_din;
    logic [V_WIDTH-1:0]    i_v_dout;

    logic [15:0]           i_alpha;
    logic [15:0]           i_v_th;

    // ========================================================================
    // Tiny BRAM models for CSR & state
    // ========================================================================

    // CSR indptr: length N_PRE+1 = 5
    // Identity network:
    //   row 0 -> nnz index 0
    //   row 1 -> nnz index 1
    //   ...
    // indptr = [0,1,2,3,4]
    logic [31:0] indptr_mem [0:N_PRE]; // 0..4

    // CSR indices: 0,1,2,3 (identity mapping)
    logic [31:0] indices_mem [0:N_POST-1];

    // CSR values: all 1.0 in Q1.14 format (16384 = 1.0)
    logic [15:0] values_mem [0:N_POST-1];

    // I_post memory: length N_POST
    logic [I_WIDTH-1:0] I_mem [0:N_POST-1];

    // v_mem memory: length N_POST
    logic [V_WIDTH-1:0] v_mem_array [0:N_POST-1];

    // Combinational read for CSR RAMs (zero-latency for simplicity)
    always_comb begin
        i_indptr_data   = indptr_mem[o_indptr_addr];
        i_indices_data  = indices_mem[o_indices_addr];
        i_values_q_data = values_mem[o_values_addr];
    end

    // Synchronous read/write for I_post and v_mem
    always_ff @(posedge clk) begin
        // I_post
        i_I_dout <= I_mem[o_I_addr];
        if (o_I_we) begin
            I_mem[o_I_addr] <= o_I_din;
        end

        // v_mem
        i_v_dout <= v_mem_array[o_v_addr];
        if (o_v_we) begin
            v_mem_array[o_v_addr] <= o_v_din;
        end
    end

    // ========================================================================
    // DUT Instance
    // ========================================================================
    kitten_fabric_tile #(
        .N_PRE      (N_PRE),
        .N_POST     (N_POST),
        .ADDRW_PRE  (ADDRW_PRE),
        .ADDRW_CSR  (ADDRW_CSR),
        .ADDRW_POST (ADDRW_POST),
        .PACK_WIDTH (PACK_WIDTH)
    ) dut (
        .clk                (clk),
        .rst                (rst),

        .i_step_start       (i_step_start),
        .i_proj_done        (i_proj_done),
        .o_step_done        (o_step_done),

        .i_pre_spike_valid  (i_pre_spike_valid),
        .i_pre_spike_idx    (i_pre_spike_idx),
        .i_pre_spike        (i_pre_spike),
        .o_pre_spike_ready  (o_pre_spike_ready),

        .o_post_spike_valid (o_post_spike_valid),
        .o_post_spike_data  (o_post_spike_data),
        .i_post_spike_ready (i_post_spike_ready),

        .o_indptr_addr      (o_indptr_addr),
        .i_indptr_data      (i_indptr_data),

        .o_indices_addr     (o_indices_addr),
        .i_indices_data     (i_indices_data),

        .o_values_addr      (o_values_addr),
        .i_values_q_data    (i_values_q_data),

        .i_scale_q          (i_scale_q),

        .o_I_addr           (o_I_addr),
        .o_I_we             (o_I_we),
        .o_I_din            (o_I_din),
        .i_I_dout           (i_I_dout),

        .o_v_addr           (o_v_addr),
        .o_v_we             (o_v_we),
        .o_v_din            (o_v_din),
        .i_v_dout           (i_v_dout),

        .i_alpha            (i_alpha),
        .i_v_th             (i_v_th)
    );

    // ========================================================================
    // Simple Spike Driver + Monitor
    // ========================================================================

    // Always ready to accept output spikes in this TB
    assign i_post_spike_ready = 1'b1;

    // Init memories & params
    initial begin : init_mem
        integer i;

        // CSR indptr (identity: each row has exactly 1 nnz)
        indptr_mem[0] = 0;
        indptr_mem[1] = 1;
        indptr_mem[2] = 2;
        indptr_mem[3] = 3;
        indptr_mem[4] = 4;

        // CSR indices (identity mapping: row j connects to col j)
        indices_mem[0] = 0;
        indices_mem[1] = 1;
        indices_mem[2] = 2;
        indices_mem[3] = 3;

        // CSR values: all 1.0 in Q1.14 (16384 = 1.0)
        for (i = 0; i < N_POST; i = i + 1) begin
            values_mem[i] = 16'd16384;
        end

        // I_post and v_mem: zero
        for (i = 0; i < N_POST; i = i + 1) begin
            I_mem[i] = '0;
            v_mem_array[i] = '0;
        end

        // LIF params in Q1.14:
        // alpha = 0.9 → ~14746 (0.9 * 16384)
        // v_th  = 0.5 → 8192
        i_alpha = 16'd14746;
        i_v_th  = 16'd8192;

        // Quant scale = 1.0 in Q1.14
        i_scale_q = 16'd16384;
    end

    // Spike sender task
    task send_spike(input [15:0] idx);
        begin
            @(posedge clk);
            i_pre_spike_idx   <= idx;
            i_pre_spike       <= 1'b1;
            i_pre_spike_valid <= 1'b1;

            // Wait for ready
            while (!o_pre_spike_ready) @(posedge clk);
            @(posedge clk);

            i_pre_spike_valid <= 1'b0;
            i_pre_spike       <= 1'b0;
        end
    endtask

    // Post spike monitor
    always_ff @(posedge clk) begin
        if (!rst && o_post_spike_valid) begin
            $display("[%0t] POST spikes: %b", $time, o_post_spike_data);
        end
    end

    // ========================================================================
    // Main Stimulus
    // ========================================================================
    initial begin : stim
        // Default
        i_step_start      = 0;
        i_proj_done       = 0;
        i_pre_spike_valid = 0;
        i_pre_spike       = 0;
        i_pre_spike_idx   = 0;

        // VCD dump for GTKWave
        $dumpfile("tb_kitten_fabric_tile.vcd");
        $dumpvars(0, tb_kitten_fabric_tile);

        // Wait for reset deassert
        @(negedge rst);
        $display("[%0t] Reset deasserted", $time);

        // ====================================================================
        // Step 1: Send spikes on neurons 0 and 2
        // ====================================================================
        @(posedge clk);
        i_step_start = 1'b1;
        $display("[%0t] Step 1 started", $time);

        // Send spike on pre-neuron 0
        send_spike(16'd0);
        $display("[%0t] Sent spike on neuron 0", $time);

        // Send spike on pre-neuron 2
        send_spike(16'd2);
        $display("[%0t] Sent spike on neuron 2", $time);

        // Indicate no more spikes this step
        @(posedge clk);
        i_proj_done = 1'b1;
        $display("[%0t] Projection done signaled", $time);

        // Wait for step_done
        wait (o_step_done == 1'b1);
        $display("[%0t] Step 1 done!", $time);

        // Drop step_start so FSM can return to IDLE
        @(posedge clk);
        i_step_start = 1'b0;
        i_proj_done  = 1'b0;

        // Some extra cycles
        repeat (5) @(posedge clk);

        // ====================================================================
        // Step 2: Send spikes on all neurons
        // ====================================================================
        @(posedge clk);
        i_step_start = 1'b1;
        $display("[%0t] Step 2 started", $time);

        // Send spikes on all pre-neurons
        send_spike(16'd0);
        send_spike(16'd1);
        send_spike(16'd2);
        send_spike(16'd3);

        @(posedge clk);
        i_proj_done = 1'b1;

        wait (o_step_done == 1'b1);
        $display("[%0t] Step 2 done!", $time);

        @(posedge clk);
        i_step_start = 1'b0;
        i_proj_done  = 1'b0;

        repeat (10) @(posedge clk);

        // ====================================================================
        // Final report
        // ====================================================================
        $display("");
        $display("=== Final membrane potentials ===");
        for (int i = 0; i < N_POST; i++) begin
            $display("  v_mem[%0d] = %0d (0x%08h)", i, $signed(v_mem_array[i]), v_mem_array[i]);
        end

        $display("");
        $display("Simulation complete.");
        $finish;
    end

endmodule
