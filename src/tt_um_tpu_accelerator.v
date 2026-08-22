// ============================================================================
// tt_um_tpu_accelerator.v — Tiny Tapeout top level
//
// A serial-I/O-fed 2x2 INT4 weight-stationary systolic array AI MAC engine
// on a single 1x1 Tiny Tapeout tile (160 um x 100 um, SkyWater 130 nm).
//
// ==========================================================================
// PINOUT
// ==========================================================================
// ui_in[7:6]  OP    opcode: 00=NOP 01=LOAD_WEIGHT 10=PUSH_ACT 11=READ_RESULT
// ui_in[5:4]  WADDR weight address (LOAD_WEIGHT): addr = k + 2*n
//                    0=W[0][0] 1=W[1][0] 2=W[0][1] 3=W[1][1]
// ui_in[5]    ACC   accumulate flag (PUSH_ACT, latched on 1st push of group)
// ui_in[3:0]  DATA  INT4 two's-complement nibble (weight or activation)
//
// uo_out[7]   RDV   1 while READ_RESULT streaming is active
// uo_out[6]   BUSY  1 while queueing/computing (do not push new tiles)
// uo_out[5]   DONE  1 when results are latched and readable
// uo_out[4:1] BCNT  bit_idx[5:2] (readout word/bit position, debug aid)
// uo_out[0]   RBIT  serial result bit, MSB-first, 48 bits per result set:
//                    word0=C[0][0], word1=C[0][1], word2=C[1][0], word3=C[1][1]
//                    (each 12-bit two's-complement, MSB first; stream wraps
//                    every 48 clocks and restarts at bit 0 on re-entry)
//
// uio_out[7:6] ST   FSM state (0=IDLE 1=RUN 2=DONE 3=READ)
// uio_out[5]   RV   res_valid (mirror)
// uio_out[4]   AM   acc_mode (mirror)
// uio_out[3:2] QC   activation queue fill level (0..3)
// uio_out[1]   WSTB weight write strobe (live)
// uio_out[0]   CSTB result capture strobe (live)
// uio_oe       all outputs enabled (debug/status observability bus)
// uio_in       unused
// ==========================================================================
//
// Usage (one matrix product C = A x W, 2x2 INT4 each):
//   1. LOAD_WEIGHT x4  (one per PE, any order)
//   2. PUSH_ACT x4 back-to-back: A[0][0], A[0][1], A[1][0], A[1][1]
//      (bit 5 of the first push = 0 fresh / 1 accumulate onto results)
//   3. wait DONE (6-cycle systolic run, hands-off)
//   4. READ_RESULT: hold opcode 11, sample RBIT on every clock (48 clocks)
//
// Multi-tile K-chaining: repeat step 2 with ACC=1 to accumulate
// C += A x W with 12-bit saturating arithmetic.
//
// Part of the Tiny Tapeout 1x1-tile TPU project.
// ============================================================================
`default_nettype none

module tt_um_tpu_accelerator (
    input  wire [7:0] ui_in,    // dedicated inputs
    output wire [7:0] uo_out,   // dedicated outputs
    input  wire [7:0] uio_in,   // bidirectional pins: input path (unused)
    output wire [7:0] uio_out,  // bidirectional pins: output path
    output wire [7:0] uio_oe,   // bidirectional pins: enable (1 = drive)
    input  wire       ena,      // tile enable (global logic freeze when 0)
    input  wire       clk,      // clock
    input  wire       rst_n     // active-low synchronous reset
);

    // ------------------------------------------------------------------
    // command field extraction
    // ------------------------------------------------------------------
    wire [1:0] cmd      = ui_in[7:6];
    wire [1:0] waddr    = ui_in[5:4];
    wire       acc_flag = ui_in[5];
    wire [3:0] data_in  = ui_in[3:0];

    // ------------------------------------------------------------------
    // internal busses
    // ------------------------------------------------------------------
    wire        q_we, q_re_we;      // queue write strobe
    wire [1:0]  q_widx, q_ridx;
    wire [1:0]  q_count;
    wire [3:0]  q_dout;

    wire [3:0]  we;
    wire [3:0]  wdata;
    wire [3:0]  row_act_0, row_act_1;

    wire        cap_en;
    wire [1:0]  cap_sel;
    wire        acc_mode;

    wire signed [11:0] psum_south_0, psum_south_1;

    wire [5:0]  bit_idx;
    wire        busy, res_valid, rd_active;
    wire [1:0]  dbg_state;

    wire        rd_bit;

    // ------------------------------------------------------------------
    // control / datapath instantiation
    // ------------------------------------------------------------------
    control_fsm u_ctrl (
        .clk          (clk),
        .rst_n        (rst_n),
        .ena          (ena),
        .cmd          (cmd),
        .waddr        (waddr),
        .acc_flag     (acc_flag),
        .data_in      (data_in),
        .q_we         (q_we),
        .q_widx       (q_widx),
        .q_ridx       (q_ridx),
        .q_count      (q_count),
        .we           (we),
        .wdata        (wdata),
        .row_act_0    (row_act_0),
        .row_act_1    (row_act_1),
        .q_dout       (q_dout),
        .cap_en       (cap_en),
        .cap_sel      (cap_sel),
        .acc_mode     (acc_mode),
        .psum_south_0 (psum_south_0),
        .psum_south_1 (psum_south_1),
        .bit_idx      (bit_idx),
        .busy         (busy),
        .res_valid    (res_valid),
        .rd_active    (rd_active),
        .dbg_state    (dbg_state)
    );

    systolic_array u_array (
        .clk          (clk),
        .rst_n        (rst_n),
        .ena          (ena),
        .we           (we),
        .wdata        (wdata),
        .row_act_0    (row_act_0),
        .row_act_1    (row_act_1),
        .psum_south_0 (psum_south_0),
        .psum_south_1 (psum_south_1)
    );

    io_serializer u_ser (
        .clk          (clk),
        .rst_n        (rst_n),
        .ena          (ena),
        .q_we         (q_we),
        .q_widx       (q_widx),
        .q_din        (data_in),
        .q_ridx       (q_ridx),
        .q_dout       (q_dout),
        .cap_en       (cap_en),
        .cap_sel      (cap_sel),
        .cap_val      (cap_sel[0] ? psum_south_1 : psum_south_0),
        .acc_mode     (acc_mode),
        .bit_idx      (bit_idx),
        .rd_bit       (rd_bit),
        .dbg_results  ()
    );

    // ------------------------------------------------------------------
    // output pin mapping
    // ------------------------------------------------------------------
    assign uo_out = {
        rd_active,          // [7] RDV
        busy,               // [6] BUSY
        res_valid,          // [5] DONE
        bit_idx[5:2],       // [4:1] BCNT (word/bit position)
        rd_active ? rd_bit : 1'b0  // [0] RBIT
    };

    assign uio_out = {
        dbg_state,          // [7:6] ST
        res_valid,          // [5]   RV
        acc_mode,           // [4]   AM
        q_count,            // [3:2] QC
        (|we),              // [1]   WSTB
        cap_en              // [0]   CSTB
    };

    // Drive the debug bus only while the tile is selected. Deriving OE from
    // the ena input (instead of a constant 8'hFF) avoids tie cells whose HI
    // pins merge into the VPWR rail during layout extraction — a known LVS
    // pin-matching hazard for constant-driven output pins.
    assign uio_oe = {8{ena}};
    // uio_in intentionally unused

endmodule

`default_nettype wire
