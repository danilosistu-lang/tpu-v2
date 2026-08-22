// ============================================================================
// io_serializer.v — input activation queue + result accumulation buffer
//
// Input side:
//   4-deep x 4-bit activation queue. The control FSM pushes nibbles at
//   q_widx and reads them back combinationally at q_ridx during the
//   systolic injection phase (single shared storage, no separate FIFO
//   pointers — area discipline for the 1x1 tile).
//
// Output side:
//   4 x 12-bit signed result registers with a *single shared* saturating
//   accumulate adder (time-multiplexed across the four capture slots —
//   one adder instead of four). Readout is pointer-indexed over the
//   48-bit flattened register file, so results can be re-read any number
//   of times with zero extra flops and no destructive shifting.
//
// Saturation range: [-2048, +2047] (12-bit signed), applied on every
// capture. A fresh (non-accumulate) tile can never saturate (|C| <= 128),
// so saturation is only reachable through accumulate-mode chaining, and
// the bit-exact golden model applies the identical clip.
//
// Part of the Tiny Tapeout 1x1-tile TPU project.
// ============================================================================
`default_nettype none

module io_serializer (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               ena,

    // ---- input queue port (owned by control_fsm) --------------------------
    input  wire               q_we,     // write strobe
    input  wire        [1:0]  q_widx,   // write index (0..3)
    input  wire        [3:0]  q_din,    // INT4 activation nibble
    input  wire        [1:0]  q_ridx,   // read index (injection phase)
    output wire        [3:0]  q_dout,   // combinational read-back

    // ---- result capture port ----------------------------------------------
    input  wire               cap_en,   // capture strobe (one per result slot)
    input  wire        [1:0]  cap_sel,  // result slot 0..3 (C00,C01,C10,C11)
    input  wire signed [11:0] cap_val,  // south-edge psum, column-muxed
    input  wire               acc_mode, // 1 = accumulate onto previous tile

    // ---- serial readout port ----------------------------------------------
    input  wire        [5:0]  bit_idx,  // 0..47, MSB-first pointer (from FSM)
    output wire               rd_bit,   // flattened[47 - bit_idx]

    // ---- debug / verification ---------------------------------------------
    output wire        [47:0] dbg_results
);

    // ------------------------------------------------------------------
    // input activation queue storage
    // ------------------------------------------------------------------
    reg [3:0] q_mem [0:3];

    assign q_dout = q_mem[q_ridx];

    always @(posedge clk) begin
        if (!rst_n) begin
            q_mem[0] <= 4'd0;
            q_mem[1] <= 4'd0;
            q_mem[2] <= 4'd0;
            q_mem[3] <= 4'd0;
        end else if (ena) begin
            if (q_we)
                q_mem[q_widx] <= q_din;
        end
    end

    // ------------------------------------------------------------------
    // result registers + shared saturating accumulate adder
    // ------------------------------------------------------------------
    reg signed [11:0] r0, r1, r2, r3;   // C00, C01, C10, C11

    wire signed [11:0] cap_prior =
        (cap_sel == 2'd0) ? r0 :
        (cap_sel == 2'd1) ? r1 :
        (cap_sel == 2'd2) ? r2 : r3;

    // accumulate mode gates the prior value; fresh tile starts from zero
    wire signed [11:0] cap_base = acc_mode ? cap_prior : 12'sd0;

    // 13-bit extended sum -> saturate to [-2048, +2047]
    wire signed [12:0] cap_sum = {cap_base[11], cap_base} +
                                 {cap_val[11],  cap_val};

    wire signed [11:0] cap_sat =
        (cap_sum > 13'sd2047)  ?  12'sd2047 :
        (cap_sum < -13'sd2048) ? -12'sd2048 :
                                  cap_sum[11:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            r0 <= 12'sd0;
            r1 <= 12'sd0;
            r2 <= 12'sd0;
            r3 <= 12'sd0;
        end else if (ena) begin
            if (cap_en) begin
                case (cap_sel)
                    2'd0: r0 <= cap_sat;
                    2'd1: r1 <= cap_sat;
                    2'd2: r2 <= cap_sat;
                    default: r3 <= cap_sat;
                endcase
            end
        end
    end

    // ------------------------------------------------------------------
    // pointer-indexed serial readout (non-destructive)
    // ------------------------------------------------------------------
    // bit 47 = r0[11] (MSB of C00): r0 streams first, matching the spec
    wire [47:0] results_flat = {r0, r1, r2, r3};

    assign rd_bit      = results_flat[6'd47 - bit_idx];
    assign dbg_results = results_flat;

endmodule

`default_nettype wire
