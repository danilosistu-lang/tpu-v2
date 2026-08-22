<!---

This file is used to generate your project datasheet. Please fill in the information below and delete any unused
sections.

You can also include images in this folder and reference them in the markdown. Each image must be less than
512 kb in size, and the combined size of all images must be less than 1 MB.
-->

## How it works

A TPU-style AI matrix-MAC engine on one Tiny Tapeout tile: a **2×2
weight-stationary systolic array** of processing elements (PEs) computing

```
C[m][n] = Σk A[m][k] · W[k][n]      (2×2 INT4 matrices, k = 0..1)
```

```
            north psum = 0        north psum = 0
                 │                     │
 row_act_0 ──► PE(0,0)·W[0][0]   PE(0,1)·W[0][1]
                 │ a_out ──►──┐   │
                 │            │   │
 row_act_1 ──► PE(1,0)·W[1][0]   PE(1,1)·W[1][1]
                 │                     │
           psum_south_0          psum_south_1
           (= C[m][0])           (= C[m][1])
```

* **Activations flow west→east** with 1-cycle systolic skew; **partial sums
  flow north→south**; one INT4 weight is **stationary** in each PE.
* One 6-clock systolic run produces all four 12-bit results in row-major
  order (C00, C01, C10, C11) on the south edge.
* Results land in a 4×12-bit accumulation buffer with a **single shared
  saturating adder** (12-bit clip at ±2048). Pushing more tiles with
  `ACC=1` accumulates onto them — this is how K > 2 matrix products and
  neural-network layers are computed by tile streaming.
* Readout is a **non-destructive 48-bit pointer mux**: results can be
  re-read any number of times, and the stream restarts at bit 0 whenever
  READ_RESULT is re-entered.

### Command protocol (serial, through ui_in)

| `ui_in[7:6]` | command | payload | effect |
|---|---|---|---|
| `00` | NOP | — | idle |
| `01` | LOAD_WEIGHT | `[5:4]`=addr, `[3:0]`=INT4 | writes weight: addr = k+2n (0=W00, 1=W10, 2=W01, 3=W11) |
| `10` | PUSH_ACT | `[5]`=ACC, `[3:0]`=INT4 | queues A nibble, row-major (A00, A01, A10, A11); 4th push auto-launches the run |
| `11` | READ_RESULT | — | streams 1 result bit/clock on `uo[0]`, MSB first; 48 bits then wraps; restart at bit 0 on re-entry |

Each command is one clock long. After four PUSH_ACT commands the FSM runs
the systolic pass; DONE (`uo[5]`) rises 6 clocks after the 4th push.

### Status & debug

* `uo[7]`=RDV (readout active), `uo[6]`=BUSY (queueing/computing),
  `uo[5]`=DONE (results latched), `uo[4:1]`=readout bit index, `uo[0]`=RBIT.
* The bidirectional pins carry a live debug bus (driven `0xFF` while `ena=1`, tri-stated otherwise):
  `uio[7:6]`=FSM state, `uio[5]`=result-valid, `uio[4]`=accumulate mode,
  `uio[3:2]`=queue fill, `uio[1]`=weight-write strobe, `uio[0]`=capture
  strobe.

### Result format

48-bit stream = four 12-bit two's-complement words, MSB first:
`C[0][0], C[0][1], C[1][0], C[1][1]`. A fresh tile never clips (|C| ≤ 128);
accumulate chains saturate at +2047 / −2048 — the verification golden model
applies the identical clip, so hardware results are bit-exact predictable.

## How to test

Run the cocotb suite locally (100% pass required in CI):

```bash
cd test
pip install -r requirements.txt
make            # 16 tests: random/corner/saturation/protocol coverage
```

On the demo board after reset (`rst_n` low one cycle):

1. LOAD_WEIGHT ×4 with your W matrix (addresses 0..3).
2. PUSH_ACT ×4 back-to-back: A[0][0], A[0][1], A[1][0], A[1][1]
   (bit 5 = 0 for a fresh tile, 1 to accumulate).
3. Wait ~6 clocks for DONE on `uo[5]`.
4. Hold READ_RESULT and sample `uo[0]` for 48 clocks.

Check against `A × W` in two's-complement 12-bit arithmetic. Example:
`A = [[1,2],[3,-4]]`, `W = [[1,0],[0,1]]` (identity) returns `[1, 2, 3, -4]`.

For gate-level simulation after the GDS action runs, the same suite is
executed with `make GATES=yes` against the synthesized netlist
(powered automatically by the GL_TEST wrapper in `test/tb.v`).

## External hardware

None required — the design is fully self-contained on one tile and works
with any Tiny Tapeout demo board / TT I/O controller.
