// ============================================================================
// pe.v — Weight-stationary systolic Processing Element
//
// Dataflow (textbook TPU-style, shrunk to INT4):
//   * Activations flow WEST -> EAST (registered pass-through, 1-cycle delay)
//   * Partial sums flow NORTH -> SOUTH (accumulate-and-register)
//   * One INT4 weight is held stationary in the PE
//
// Arithmetic: prod = a_in * w   (INT4 x INT4 -> INT8, range -56..+64)
//            p_out = p_in + prod (12-bit signed)
//
// Width analysis (Milestone-3): the internal chain provably never exceeds
// |psum| <= 128 (9 bits). A measured experiment narrowing the chain to
// 9 bits + sign-extension at the boundary synthesised 3% LARGER than the
// plain 12-bit chain, because ABC already exploits the sign-redundancy of
// the upper bits -- so the simple 12-bit datapath is kept (measured area
// 8523 um^2 vs 8765 um^2 for the narrowed variant).
//
// The PE needs no accumulator clear: the north edge of row-0 is tied to
// zero, so every output slot emerges pre-initialised (stale partial sums
// are overwritten slot-by-slot and never contaminate results).
//
// Part of the Tiny Tapeout 1x1-tile TPU project.
// ============================================================================
`default_nettype none

module pe (
    input  wire               clk,
    input  wire               rst_n,
    input  wire               ena,

    // ---- weight-load port (one-cycle command strobe) ----------------------
    input  wire               we,      // write enable for stationary weight
    input  wire        [3:0]  wdata,   // INT4 two's-complement weight

    // ---- activation stream: west -> east ----------------------------------
    input  wire        [3:0]  a_in,
    output reg         [3:0]  a_out,

    // ---- partial-sum stream: north -> south -------------------------------
    input  wire signed [11:0] p_in,
    output reg  signed [11:0] p_out
);

    reg  signed [3:0]  w;                     // stationary INT4 weight
    wire signed [3:0]  a = a_in;              // reinterpret as signed
    wire signed [7:0]  prod = a * w;          // INT8 product (-56 .. +64)

    always @(posedge clk) begin
        if (!rst_n) begin
            w     <= 4'sd0;
            a_out <= 4'd0;
            p_out <= 12'sd0;
        end else if (ena) begin
            if (we)
                w <= wdata;
            a_out <= a_in;                    // systolic eastward delay
            p_out <= p_in + prod;             // systolic southward accumulate
        end
    end

endmodule

`default_nettype wire
