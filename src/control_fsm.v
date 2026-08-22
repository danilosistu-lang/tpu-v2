// ============================================================================
// control_fsm.v — command decoder + sequencer
//
// Serial command encoding on ui_in:
//   ui_in[7:6] = opcode:  00 = NOP, 01 = LOAD_WEIGHT,
//                         10 = PUSH_ACT, 11 = READ_RESULT
//   ui_in[5:4] = WADDR (LOAD_WEIGHT): addr = k + 2*n
//   ui_in[5]   = ACC   (PUSH_ACT): accumulate flag (latched on 1st push)
//   ui_in[3:0] = DATA  nibble
//
// FSM (4 states, 2 FF):
//   S_IDLE -> (4th PUSH_ACT) -> S_RUN -> S_DONE <-> S_READ
//     S_RUN: 6 cycles. cnt 0..3 = skewed injection (idx=cnt, even->row0,
//            odd->row1); cnt 4..5 = drain. Result captures at cnt 2..5
//            map to slots 0..3 (C00,C01,C10,C11); south column = cnt[0].
//     S_DONE: results valid (res_valid=1); accepts LOAD_WEIGHT / PUSH_ACT
//            (new tile) / READ_RESULT (enter readout).
//     S_READ: streams one result bit per clock MSB-first (bit_idx 0..47,
//            wraps). Exits to S_DONE on any opcode != READ; re-entering
//            restarts at bit 0 (non-destructive pointer readout).
//
// Flop budget notes (1x1 tile area discipline): no separate "fill" state —
// queueing happens in S_IDLE/S_DONE with a 2-bit counter; a single 3-bit
// run counter doubles as injection index, capture scheduler and drain
// timer; bit_idx is the only readout state.
//
// Part of the Tiny Tapeout 1x1-tile TPU project.
// ============================================================================
`default_nettype none

module control_fsm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        ena,

    // ---- decoded command interface ----------------------------------------
    input  wire [1:0]  cmd,       // ui_in[7:6]
    input  wire [1:0]  waddr,     // ui_in[5:4]
    input  wire        acc_flag,  // ui_in[5]
    input  wire [3:0]  data_in,   // ui_in[3:0]

    // ---- activation queue (storage lives in io_serializer) ----------------
    output wire        q_we,
    output wire [1:0]  q_widx,
    output wire [1:0]  q_ridx,
    output reg  [1:0]  q_count,   // owned here, storage in io_serializer

    // ---- systolic array control -------------------------------------------
    output wire [3:0]  we,        // one-hot weight write strobes
    output wire [3:0]  wdata,
    output wire [3:0]  row_act_0, // skewed west-edge injection
    output wire [3:0]  row_act_1,
    input  wire [3:0]  q_dout,    // queue read-back for injection

    // ---- result capture (io_serializer) ------------------------------------
    output wire        cap_en,
    output wire [1:0]  cap_sel,
    output wire        acc_mode,
    input  wire signed [11:0] psum_south_0,
    input  wire signed [11:0] psum_south_1,

    // ---- serial readout -----------------------------------------------------
    output wire [5:0]  bit_idx,

    // ---- status -------------------------------------------------------------
    output wire        busy,
    output wire        res_valid,
    output wire        rd_active,

    // ---- debug bus (uio) ----------------------------------------------------
    output wire [1:0]  dbg_state
);

    // ------------------------------------------------------------------
    // opcodes / states
    // ------------------------------------------------------------------
    localparam [1:0] CMD_NOP   = 2'b00,
                     CMD_LOADW = 2'b01,
                     CMD_PUSH  = 2'b10,
                     CMD_READ  = 2'b11;

    localparam [1:0] S_IDLE = 2'd0,
                     S_RUN  = 2'd1,
                     S_DONE = 2'd2,
                     S_READ = 2'd3;

    reg [1:0] state;
    reg [2:0] run_cnt;      // 0..5: inject x4 + drain x2
    reg [5:0] bit_cnt;      // 0..47 readout pointer
    reg       acc_mode_r;   // latched on first push of a group
    reg       res_valid_r;

    // ------------------------------------------------------------------
    // command acceptance
    // ------------------------------------------------------------------
    wire accepting = (state == S_IDLE) || (state == S_DONE);

    // weight-load decode: one-cycle one-hot strobe
    assign we = (cmd == CMD_LOADW) && accepting ? (4'b0001 << waddr) : 4'b0000;

    // activation push decode
    wire push = (cmd == CMD_PUSH) && accepting;

    assign q_we   = push;
    assign q_widx = q_count;

    // ------------------------------------------------------------------
    // queue count (owned here; storage + combinational read in serializer)
    // ------------------------------------------------------------------
    wire last_push = push && (q_count == 2'd3);     // 4th nibble -> launch

    always @(posedge clk) begin
        if (!rst_n)
            q_count <= 2'd0;
        else if (ena) begin
            if (last_push)
                q_count <= 2'd0;                    // consumed by RUN
            else if (push)
                q_count <= q_count + 2'd1;
        end
    end

    // ------------------------------------------------------------------
    // main sequencer
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= S_IDLE;
            run_cnt     <= 3'd0;
            bit_cnt     <= 6'd0;
            acc_mode_r  <= 1'b0;
            res_valid_r <= 1'b0;
        end else if (ena) begin
            case (state)
                S_IDLE, S_DONE: begin
                    if (last_push) begin
                        state   <= S_RUN;
                        run_cnt <= 3'd0;
                        res_valid_r <= 1'b0;        // new tile invalidates
                        // first push of this group latched the acc flag
                    end
                    if (push && (q_count == 2'd0))
                        acc_mode_r <= acc_flag;     // latched on 1st push
                    if (!push && (cmd == CMD_READ) && res_valid_r &&
                        (state == S_DONE)) begin
                        state   <= S_READ;
                        bit_cnt <= 6'd0;
                    end
                end

                S_RUN: begin
                    if (run_cnt == 3'd5) begin
                        state       <= S_DONE;
                        res_valid_r <= 1'b1;
                        run_cnt     <= 3'd0;
                    end else
                        run_cnt <= run_cnt + 3'd1;
                end

                S_READ: begin
                    if (cmd != CMD_READ)
                        state <= S_DONE;            // exit -> restartable
                    else
                        bit_cnt <= (bit_cnt == 6'd47) ? 6'd0
                                                     : bit_cnt + 6'd1;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

    // ------------------------------------------------------------------
    // systolic injection (combinational from queue read port)
    // ------------------------------------------------------------------
    wire inject_phase = (state == S_RUN) && (run_cnt < 3'd4);

    assign q_ridx    = run_cnt[1:0];
    assign row_act_0 = inject_phase && !run_cnt[0] ? q_dout : 4'd0;
    assign row_act_1 = inject_phase &&  run_cnt[0] ? q_dout : 4'd0;

    // ------------------------------------------------------------------
    // result capture scheduling (cnt 2..5 -> slots 0..3, column = cnt[0])
    // ------------------------------------------------------------------
    wire cap_window = (state == S_RUN) && (run_cnt >= 3'd2);

    assign cap_en  = cap_window;
    assign cap_sel = run_cnt[1:0] - 2'd2;

    // ------------------------------------------------------------------
    // outputs
    // ------------------------------------------------------------------
    assign wdata     = data_in;
    assign bit_idx   = bit_cnt;
    assign acc_mode  = acc_mode_r;
    assign busy      = (state == S_RUN) || (q_count != 2'd0);
    assign res_valid = res_valid_r;
    assign rd_active = (state == S_READ);

    assign dbg_state = state;

endmodule

`default_nettype wire
