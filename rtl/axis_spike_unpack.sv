// =============================================================================
// axis_spike_unpack.sv
//
// AXI-Stream → Presynaptic Spike Events Unpacker
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// Encoding convention:
//   tdata[15:0]  = presynaptic neuron index
//   tdata[16]    = spike bit (1 = spike, 0 = no spike)
//   tdata[31:17] = reserved
//
// Bridges PCIe H2C AXIS stream to kitten_fabric_tile presyn interface.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module axis_spike_unpack #(
    parameter AXIS_DATA_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst,

    // ========================================================================
    // AXI-Stream Slave (from PCIe/DMA H2C)
    // ========================================================================
    input  wire [AXIS_DATA_WIDTH-1:0]   s_axis_tdata,
    input  wire                         s_axis_tvalid,
    output wire                         s_axis_tready,
    input  wire                         s_axis_tlast,   // Optional, unused
    input  wire [3:0]                   s_axis_tkeep,   // Optional, unused

    // ========================================================================
    // Spike Event Output (to kitten_fabric_tile)
    // ========================================================================
    output reg                          o_spike_valid,
    output reg  [15:0]                  o_spike_idx,
    output reg                          o_spike_bit,
    input  wire                         i_spike_ready
);

    // Simple pass-through handshake:
    //   - We are ready iff downstream is ready
    //   - Spike valid = AXIS valid & ready
    assign s_axis_tready = i_spike_ready;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            o_spike_valid <= 1'b0;
            o_spike_idx   <= 16'd0;
            o_spike_bit   <= 1'b0;
        end else begin
            // Default: no spike
            o_spike_valid <= 1'b0;

            if (s_axis_tvalid && s_axis_tready) begin
                // Consume one AXIS word, emit one spike event
                o_spike_valid <= 1'b1;
                o_spike_idx   <= s_axis_tdata[15:0];
                o_spike_bit   <= s_axis_tdata[16];
            end
        end
    end

endmodule

`default_nettype wire
