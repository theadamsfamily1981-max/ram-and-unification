// =============================================================================
// kitten_axil_regs.sv
//
// AXI-Lite Register Block for Kitten Fabric Tile Control
// Ara-SYNERGY Kitten Fabric - Synthesizable RTL
//
// Register Map:
//   0x00  CONTROL   R/W  bit0=step_start, bit1=proj_done
//   0x04  STATUS    R    bit0=step_done, bit1=busy
//   0x08  ALPHA     R/W  LIF alpha parameter (16-bit Q1.14)
//   0x0C  V_TH      R/W  LIF threshold (16-bit Q1.14)
//   0x10  SCALE_Q   R/W  Projection quantization scale (16-bit)
//
// AXI-Lite interface follows ARM AMBA spec (no burst, 32-bit data).
// =============================================================================

`timescale 1ns / 1ps
`default_nettype none

module kitten_axil_regs #(
    parameter ADDR_WIDTH = 8,
    parameter DATA_WIDTH = 32
) (
    input  wire                     clk,
    input  wire                     rst,

    // ========================================================================
    // AXI-Lite Slave Interface
    // ========================================================================
    // Write address channel
    input  wire [ADDR_WIDTH-1:0]    s_axil_awaddr,
    input  wire                     s_axil_awvalid,
    output reg                      s_axil_awready,

    // Write data channel
    input  wire [DATA_WIDTH-1:0]    s_axil_wdata,
    input  wire [3:0]               s_axil_wstrb,
    input  wire                     s_axil_wvalid,
    output reg                      s_axil_wready,

    // Write response channel
    output reg  [1:0]               s_axil_bresp,
    output reg                      s_axil_bvalid,
    input  wire                     s_axil_bready,

    // Read address channel
    input  wire [ADDR_WIDTH-1:0]    s_axil_araddr,
    input  wire                     s_axil_arvalid,
    output reg                      s_axil_arready,

    // Read data channel
    output reg  [DATA_WIDTH-1:0]    s_axil_rdata,
    output reg  [1:0]               s_axil_rresp,
    output reg                      s_axil_rvalid,
    input  wire                     s_axil_rready,

    // ========================================================================
    // Control Outputs (to kitten_fabric_tile)
    // ========================================================================
    output reg                      o_step_start,
    output reg                      o_proj_done,
    output reg  [15:0]              o_alpha,
    output reg  [15:0]              o_v_th,
    output reg  [15:0]              o_scale_q,

    // ========================================================================
    // Status Inputs (from kitten_fabric_tile)
    // ========================================================================
    input  wire                     i_step_done,
    input  wire                     i_busy
);

    // ========================================================================
    // Register Addresses
    // ========================================================================
    localparam [7:0] ADDR_CONTROL = 8'h00;
    localparam [7:0] ADDR_STATUS  = 8'h04;
    localparam [7:0] ADDR_ALPHA   = 8'h08;
    localparam [7:0] ADDR_V_TH    = 8'h0C;
    localparam [7:0] ADDR_SCALE_Q = 8'h10;

    // ========================================================================
    // AXI Response Codes
    // ========================================================================
    localparam [1:0] RESP_OKAY   = 2'b00;
    localparam [1:0] RESP_SLVERR = 2'b10;

    // ========================================================================
    // Internal Registers
    // ========================================================================
    reg [DATA_WIDTH-1:0] reg_control;
    reg [DATA_WIDTH-1:0] reg_alpha;
    reg [DATA_WIDTH-1:0] reg_v_th;
    reg [DATA_WIDTH-1:0] reg_scale_q;

    // Latched addresses for pipelined access
    reg [ADDR_WIDTH-1:0] aw_addr;
    reg [ADDR_WIDTH-1:0] ar_addr;

    // ========================================================================
    // Write State Machine
    // ========================================================================
    localparam [1:0] W_IDLE    = 2'd0;
    localparam [1:0] W_ADDR    = 2'd1;
    localparam [1:0] W_DATA    = 2'd2;
    localparam [1:0] W_RESP    = 2'd3;

    reg [1:0] w_state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            w_state         <= W_IDLE;
            s_axil_awready  <= 1'b0;
            s_axil_wready   <= 1'b0;
            s_axil_bvalid   <= 1'b0;
            s_axil_bresp    <= RESP_OKAY;
            aw_addr         <= {ADDR_WIDTH{1'b0}};

            // Default register values
            reg_control     <= 32'd0;
            reg_alpha       <= 32'd14746;  // 0.9 in Q1.14
            reg_v_th        <= 32'd8192;   // 0.5 in Q1.14
            reg_scale_q     <= 32'd16384;  // 1.0 in Q1.14
        end else begin
            case (w_state)
                W_IDLE: begin
                    s_axil_awready <= 1'b1;
                    s_axil_wready  <= 1'b0;
                    s_axil_bvalid  <= 1'b0;

                    if (s_axil_awvalid && s_axil_awready) begin
                        aw_addr        <= s_axil_awaddr;
                        s_axil_awready <= 1'b0;
                        s_axil_wready  <= 1'b1;
                        w_state        <= W_DATA;
                    end
                end

                W_DATA: begin
                    if (s_axil_wvalid && s_axil_wready) begin
                        s_axil_wready <= 1'b0;

                        // Write to register based on address
                        case (aw_addr[7:0])
                            ADDR_CONTROL: begin
                                if (s_axil_wstrb[0]) reg_control[7:0]   <= s_axil_wdata[7:0];
                                if (s_axil_wstrb[1]) reg_control[15:8]  <= s_axil_wdata[15:8];
                                if (s_axil_wstrb[2]) reg_control[23:16] <= s_axil_wdata[23:16];
                                if (s_axil_wstrb[3]) reg_control[31:24] <= s_axil_wdata[31:24];
                                s_axil_bresp <= RESP_OKAY;
                            end
                            ADDR_STATUS: begin
                                // STATUS is read-only
                                s_axil_bresp <= RESP_SLVERR;
                            end
                            ADDR_ALPHA: begin
                                if (s_axil_wstrb[0]) reg_alpha[7:0]   <= s_axil_wdata[7:0];
                                if (s_axil_wstrb[1]) reg_alpha[15:8]  <= s_axil_wdata[15:8];
                                s_axil_bresp <= RESP_OKAY;
                            end
                            ADDR_V_TH: begin
                                if (s_axil_wstrb[0]) reg_v_th[7:0]   <= s_axil_wdata[7:0];
                                if (s_axil_wstrb[1]) reg_v_th[15:8]  <= s_axil_wdata[15:8];
                                s_axil_bresp <= RESP_OKAY;
                            end
                            ADDR_SCALE_Q: begin
                                if (s_axil_wstrb[0]) reg_scale_q[7:0]   <= s_axil_wdata[7:0];
                                if (s_axil_wstrb[1]) reg_scale_q[15:8]  <= s_axil_wdata[15:8];
                                s_axil_bresp <= RESP_OKAY;
                            end
                            default: begin
                                s_axil_bresp <= RESP_SLVERR;
                            end
                        endcase

                        s_axil_bvalid <= 1'b1;
                        w_state       <= W_RESP;
                    end
                end

                W_RESP: begin
                    if (s_axil_bready && s_axil_bvalid) begin
                        s_axil_bvalid <= 1'b0;
                        w_state       <= W_IDLE;
                    end
                end

                default: w_state <= W_IDLE;
            endcase
        end
    end

    // ========================================================================
    // Read State Machine
    // ========================================================================
    localparam [1:0] R_IDLE = 2'd0;
    localparam [1:0] R_ADDR = 2'd1;
    localparam [1:0] R_DATA = 2'd2;

    reg [1:0] r_state;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            r_state         <= R_IDLE;
            s_axil_arready  <= 1'b0;
            s_axil_rvalid   <= 1'b0;
            s_axil_rdata    <= 32'd0;
            s_axil_rresp    <= RESP_OKAY;
            ar_addr         <= {ADDR_WIDTH{1'b0}};
        end else begin
            case (r_state)
                R_IDLE: begin
                    s_axil_arready <= 1'b1;
                    s_axil_rvalid  <= 1'b0;

                    if (s_axil_arvalid && s_axil_arready) begin
                        ar_addr        <= s_axil_araddr;
                        s_axil_arready <= 1'b0;
                        r_state        <= R_DATA;
                    end
                end

                R_DATA: begin
                    // Read from register based on address
                    case (ar_addr[7:0])
                        ADDR_CONTROL: begin
                            s_axil_rdata <= reg_control;
                            s_axil_rresp <= RESP_OKAY;
                        end
                        ADDR_STATUS: begin
                            s_axil_rdata <= {30'd0, i_busy, i_step_done};
                            s_axil_rresp <= RESP_OKAY;
                        end
                        ADDR_ALPHA: begin
                            s_axil_rdata <= reg_alpha;
                            s_axil_rresp <= RESP_OKAY;
                        end
                        ADDR_V_TH: begin
                            s_axil_rdata <= reg_v_th;
                            s_axil_rresp <= RESP_OKAY;
                        end
                        ADDR_SCALE_Q: begin
                            s_axil_rdata <= reg_scale_q;
                            s_axil_rresp <= RESP_OKAY;
                        end
                        default: begin
                            s_axil_rdata <= 32'hDEADBEEF;
                            s_axil_rresp <= RESP_SLVERR;
                        end
                    endcase

                    s_axil_rvalid <= 1'b1;
                    r_state       <= R_IDLE;

                    // Wait for rready
                    if (!(s_axil_rready)) begin
                        r_state <= R_DATA;  // Stay until accepted
                    end
                end

                default: r_state <= R_IDLE;
            endcase

            // Clear rvalid when accepted
            if (s_axil_rvalid && s_axil_rready) begin
                s_axil_rvalid <= 1'b0;
            end
        end
    end

    // ========================================================================
    // Output Mapping
    // ========================================================================
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            o_step_start <= 1'b0;
            o_proj_done  <= 1'b0;
            o_alpha      <= 16'd14746;
            o_v_th       <= 16'd8192;
            o_scale_q    <= 16'd16384;
        end else begin
            o_step_start <= reg_control[0];
            o_proj_done  <= reg_control[1];
            o_alpha      <= reg_alpha[15:0];
            o_v_th       <= reg_v_th[15:0];
            o_scale_q    <= reg_scale_q[15:0];
        end
    end

endmodule

`default_nettype wire
