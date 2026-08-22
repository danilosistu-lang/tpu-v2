// ============================================================================
// systolic_array.v — 2x2 weight-stationary systolic array
//
// Grid layout (weights stationary, PE(k,n) holds W[k][n]):
//
//            north psum = 0          north psum = 0
//                 |                       |
//   row_act_0 --> PE(0,0).W[0][0]  PE(0,1).W[0][1]
//                    |  a_out ->--.         |
//                    |            |         |
//   row_act_1 --> PE(1,0).W[1][0]  PE(1,1).W[1][1]
//                    |                       |
//             psum_south_0              psum_south_1
//             (= C[m][0])               (= C[m][1])
//
// Activations are injected skewed at the west edge (row k carries the
// A[m][k] taps), partial sums enter row 0 from the north tied to zero and
// emerge on the south edge in row-major result order:
//   t+2: C[0][0]   t+3: C[0][1]   t+4: C[1][0]   t+5: C[1][1]
// (t = cycle in which the first skewed activation is presented).
//
// Weight-load address map (one-hot we[]):
//   we[0] -> PE(0,0) = W[0][0]     we[1] -> PE(1,0) = W[1][0]
//   we[2] -> PE(0,1) = W[0][1]     we[3] -> PE(1,1) = W[1][1]
// i.e. external address = k + 2*n  (k = contraction row, n = output column).
//
// Part of the Tiny Tapeout 1x1-tile TPU project.
// ============================================================================
`default_nettype none

module systolic_array (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               ena,

    // ---- weight-load interface (decoded by control_fsm) -------------------
    input  wire        [3:0]  we,        // one-hot PE write strobes
    input  wire        [3:0]  wdata,     // INT4 weight nibble (shared bus)

    // ---- skewed activation injection at the west edge ---------------------
    input  wire        [3:0]  row_act_0, // row k=0 taps: A[m][0] on even slots
    input  wire        [3:0]  row_act_1, // row k=1 taps: A[m][1] on odd slots

    // ---- south-edge partial sums (final MAC results) ----------------------
    output wire signed [11:0] psum_south_0, // column n=0: C[m][0]
    output wire signed [11:0] psum_south_1  // column n=1: C[m][1]
);

    // internal southward psum wires (row 0 -> row 1)
    wire signed [11:0] p_mid_col0;
    wire signed [11:0] p_mid_col1;

    // internal eastward activation wires (column 0 -> column 1)
    wire [3:0] a_mid_row0;
    wire [3:0] a_mid_row1;

    // ---- row 0 (k = 0 taps) ------------------------------------------------
    pe u_pe00 (
        .clk   (clk),
        .rst_n (rst_n),
        .ena   (ena),
        .we    (we[0]),
        .wdata (wdata),
        .a_in  (row_act_0),
        .a_out (a_mid_row0),
        .p_in  (12'sd0),           // north edge tied to zero
        .p_out (p_mid_col0)
    );

    pe u_pe01 (
        .clk   (clk),
        .rst_n (rst_n),
        .ena   (ena),
        .we    (we[2]),
        .wdata (wdata),
        .a_in  (a_mid_row0),
        .a_out (),                // east edge of the array (unused)
        .p_in  (12'sd0),           // north edge tied to zero
        .p_out (p_mid_col1)
    );

    // ---- row 1 (k = 1 taps) ------------------------------------------------
    pe u_pe10 (
        .clk   (clk),
        .rst_n (rst_n),
        .ena   (ena),
        .we    (we[1]),
        .wdata (wdata),
        .a_in  (row_act_1),
        .a_out (a_mid_row1),
        .p_in  (p_mid_col0),      // southward chain, column 0
        .p_out (psum_south_0)
    );

    pe u_pe11 (
        .clk   (clk),
        .rst_n (rst_n),
        .ena   (ena),
        .we    (we[3]),
        .wdata (wdata),
        .a_in  (a_mid_row1),
        .a_out (),                // east edge of the array (unused)
        .p_in  (p_mid_col1),      // southward chain, column 1
        .p_out (psum_south_1)
    );

endmodule

`default_nettype wire
