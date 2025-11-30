// =============================================================================
// spike_axis_pack.sv
//
// Postsynaptic Spike Packer → AXI-Stream
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// Encoding convention:
//   tdata[PACK_WIDTH-1:0] = packed spike bits from LIF population
//   tdata[31:PACK_WIDTH]  = zero-padded
//
// Bridges kitten_fabric_tile postsyn interface to PCIe C2H AXIS stream.
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module spike_axis_pack #(
    parameter PACK_WIDTH      = 8,
    parameter AXIS_DATA_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst,

    // ========================================================================
    // Spike Input (from kitten_fabric_tile)
    // ========================================================================
    input  wire                         i_spike_valid,
    input  wire [PACK_WIDTH-1:0]        i_spike_data,
    output wire                         o_spike_ready,

    // ========================================================================
    // AXI-Stream Master (to PCIe/DMA C2H)
    // ========================================================================
    output reg  [AXIS_DATA_WIDTH-1:0]   m_axis_tdata,
    output reg                          m_axis_tvalid,
    input  wire                         m_axis_tready,
    output reg                          m_axis_tlast,
    output reg  [3:0]                   m_axis_tkeep
);

    // Simple 1:1 mapping: one spike pack → one AXIS word
    assign o_spike_ready = m_axis_tready;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tdata  <= {AXIS_DATA_WIDTH{1'b0}};
            m_axis_tlast  <= 1'b0;
            m_axis_tkeep  <= 4'hF;
        end else begin
            // Default: no valid output
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;

            if (i_spike_valid && m_axis_tready) begin
                // Emit one AXIS word with packed spikes in low bits
                m_axis_tvalid <= 1'b1;
                m_axis_tdata  <= {{(AXIS_DATA_WIDTH-PACK_WIDTH){1'b0}}, i_spike_data};
                m_axis_tkeep  <= 4'hF;
                // tlast could be used for framing if needed
            end
        end
    end

endmodule

`default_nettype wire
