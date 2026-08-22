# SPDX-FileCopyrightText: © 2024 Tiny Tapeout
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# test.py — cocotb verification of tt_um_tpu_accelerator (through tb.v)
#
# Drives the real serial protocol over ui_in and checks bit-exact agreement
# with the inline NumPy golden model across:
#   * reset/idle status behaviour
#   * directed + 100 random 2x2 INT4 matrix products
#   * INT4 corner values (all -8, all +7, +-128 magnitude tiles)
#   * weight nibble sweep and weight rewrites between tiles
#   * multi-tile accumulate chains incl. 12-bit saturation (+2047 / -2048)
#   * readout restart / re-read / mid-stream exit / bit-exact 48-bit stream
#   * commands ignored while busy, BUSY timing, ena freeze mid-queue
#   * random session soak (interleaved loads, acc/fresh tiles, re-reads)
# ============================================================================

import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import FallingEdge, RisingEdge

import numpy as np

# ---------------------------------------------------------------------------
# opcodes / pin map (see src/tt_um_tpu_accelerator.v header)
# ---------------------------------------------------------------------------
CMD_NOP, CMD_LOADW, CMD_PUSH, CMD_READ = 0b00, 0b01, 0b10, 0b11

UO_RDV, UO_BUSY, UO_DONE, UO_RBIT = 7, 6, 5, 0

CLK_NS = 10
rng = random.Random(0xC0FFEE)          # deterministic test data


# ---------------------------------------------------------------------------
# golden reference model (bit-exact twin of the RTL semantics)
# ---------------------------------------------------------------------------
ACC_MIN, ACC_MAX = -2048, 2047


def clip12(v):
    return int(np.clip(v, ACC_MIN, ACC_MAX))


def sign4(nibble):
    nibble &= 0xF
    return nibble - 16 if nibble & 0x8 else nibble


def signed12(unsigned):
    unsigned &= 0xFFF
    return unsigned - 4096 if unsigned & 0x800 else unsigned


def unsigned12(value):
    return value & 0xFFF


class TpuGoldenModel:
    def __init__(self):
        self.W = np.zeros((2, 2), dtype=np.int32)     # W[k][n]
        self.R = [0, 0, 0, 0]                         # C00 C01 C10 C11

    def load_weight(self, addr, nibble):
        k, n = addr & 1, addr >> 1                    # addr = k + 2n
        self.W[k][n] = sign4(nibble)

    def load_weights(self, W):
        for addr in range(4):
            self.load_weight(addr, int(W[addr & 1][addr >> 1]) & 0xF)

    def push_tile(self, A, acc):
        C = np.asarray(A, dtype=np.int32) @ self.W    # exact integer math
        flat = (int(C[0][0]), int(C[0][1]), int(C[1][0]), int(C[1][1]))
        for i, v in enumerate(flat):
            self.R[i] = clip12(self.R[i] + v if acc else v)
        return list(self.R)

    def result_bitstream(self):
        bits = []
        for word in self.R:
            for b in range(11, -1, -1):
                bits.append((unsigned12(word) >> b) & 1)
        return bits


# ---------------------------------------------------------------------------
# bus functional model of the serial command protocol
# ---------------------------------------------------------------------------
def uo_val(dut):
    return int(dut.uo_out.value)


def uo_bit(dut, idx):
    return (uo_val(dut) >> idx) & 1


async def do_reset(dut, cycles=4):
    dut.ui_in.value = 0
    dut.uio_in.value = 0
    dut.ena.value = 1
    dut.rst_n.value = 0
    cocotb.start_soon(Clock(dut.clk, CLK_NS, unit="ns").start())
    for _ in range(cycles):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.rst_n.value = 1
    for _ in range(2):
        await RisingEdge(dut.clk)
    await FallingEdge(dut.clk)


class TpuDriver:
    def __init__(self, dut):
        self.dut = dut

    async def cmd(self, ui):
        """Drive one command for exactly one clock, then NOP."""
        await FallingEdge(self.dut.clk)
        self.dut.ui_in.value = ui
        await FallingEdge(self.dut.clk)
        self.dut.ui_in.value = 0

    async def idle(self, n=1):
        await FallingEdge(self.dut.clk)
        self.dut.ui_in.value = 0
        for _ in range(n - 1):
            await FallingEdge(self.dut.clk)

    async def load_weights(self, W):
        """LOAD_WEIGHT x4: addr = k + 2*n."""
        for addr in range(4):
            k, n = addr & 1, addr >> 1
            await self.cmd((CMD_LOADW << 6) | (addr << 4) | (int(W[k][n]) & 0xF))

    async def push_tile(self, A, acc):
        """PUSH_ACT x4 back-to-back (one clock each), then NOP."""
        vals = (int(A[0][0]), int(A[0][1]), int(A[1][0]), int(A[1][1]))
        for v in vals:
            await FallingEdge(self.dut.clk)
            self.dut.ui_in.value = (CMD_PUSH << 6) | (acc << 5) | (v & 0xF)
        await FallingEdge(self.dut.clk)
        self.dut.ui_in.value = 0          # never park a command into S_DONE

    async def wait_done(self, timeout=60):
        for _ in range(timeout):
            if uo_bit(self.dut, UO_DONE):
                return
            await FallingEdge(self.dut.clk)
        raise AssertionError("timeout waiting for DONE")

    async def read_results(self):
        """Hold READ_RESULT for 48 clocks, sample RBIT at every negedge."""
        dut = self.dut
        await FallingEdge(dut.clk)
        dut.ui_in.value = CMD_READ << 6
        bits = []
        for _ in range(48):
            await FallingEdge(dut.clk)
            assert uo_bit(dut, UO_RDV), "RDV deasserted during readout"
            bits.append(uo_bit(dut, UO_RBIT))
        await FallingEdge(dut.clk)
        dut.ui_in.value = 0
        return [signed12(sum(b << i for i, b in
                             enumerate(reversed(bits[w * 12:(w + 1) * 12]))))
                for w in range(4)]

    async def run_tile(self, A, acc):
        await self.push_tile(A, acc)
        await self.idle(2)
        await self.wait_done()
        return await self.read_results()

    async def full_op(self, W, A, acc=0):
        await self.load_weights(W)
        return await self.run_tile(A, acc)


def rand_i4():
    return [[rng.randint(-8, 7) for _ in range(2)] for _ in range(2)]


def expect(model, got, ctx):
    exp = list(model.R)
    assert got == exp, f"{ctx}: DUT={got} != golden={exp}"


# ===========================================================================
# tests
# ===========================================================================

@cocotb.test()
async def test_reset_and_idle_status(dut):
    """After reset: not busy, no result, no readout activity."""
    await do_reset(dut)
    assert not uo_bit(dut, UO_BUSY)
    assert not uo_bit(dut, UO_DONE)
    assert not uo_bit(dut, UO_RDV)
    assert not uo_bit(dut, UO_RBIT)
    assert int(dut.uio_oe.value) == 0xFF, "debug bus must drive when ena=1"
    dut.ena.value = 0
    await FallingEdge(dut.clk)
    assert int(dut.uio_oe.value) == 0x00, "debug bus must tri-state when ena=0"
    dut.ena.value = 1
    await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    assert not uo_bit(dut, UO_BUSY), "spurious busy after reset"


@cocotb.test()
async def test_identity_matrix_product(dut):
    """Hand-computed sanity: identity weights return A unchanged."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[1, 0], [0, 1]]
    A = [[1, 2], [3, -4]]
    model.load_weights(W)
    model.push_tile(A, acc=0)
    got = await drv.full_op(W, A, acc=0)
    expect(model, got, "identity")
    assert got == [1, 2, 3, -4]


@cocotb.test()
async def test_directed_mixed(dut):
    """Directed non-trivial product with negatives on every row/column."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[1, -2], [-3, 4]]
    A = [[5, -6], [7, -8]]
    model.load_weights(W)
    model.push_tile(A, acc=0)
    got = await drv.full_op(W, A, acc=0)
    expect(model, got, "directed")
    assert got == [23, -34, 31, -46]


@cocotb.test()
async def test_corner_values(dut):
    """INT4 extremes: all -8, all +7, and maximum +-128 tile magnitudes."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    corners = [
        ([[-8, -8], [-8, -8]], [[-8, -8], [-8, -8]]),   # C = 128 everywhere
        ([[7, 7], [7, 7]],     [[7, 7], [7, 7]]),       # C = 98
        ([[-8, 7], [7, -8]],   [[-8, -8], [7, 7]]),     # mixed extremes
        ([[7, 7], [7, 7]],     [[-8, -8], [-8, -8]]),   # C = -112
        ([[0, 0], [0, 0]],     [[5, -5], [5, -5]]),     # zero activations
        ([[5, -5], [5, -5]],   [[0, 0], [0, 0]]),       # zero weights
    ]
    for i, (W, A) in enumerate(corners):
        model.load_weights(W)
        model.push_tile(A, acc=0)
        got = await drv.full_op(W, A, acc=0)
        expect(model, got, f"corner case {i}")


@cocotb.test()
async def test_weight_nibble_sweep(dut):
    """Sweep every weight nibble value with a fixed probing activation."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    A = [[-8, 7], [7, -8]]
    for w00 in range(-8, 8):
        W = [[w00, -3], [5, w00]]
        model.load_weights(W)
        model.push_tile(A, acc=0)
        got = await drv.full_op(W, A, acc=0)
        expect(model, got, f"weight sweep w={w00}")


@cocotb.test()
async def test_random_matmuls(dut):
    """100 random (W, A) pairs — fresh weights and activations each time."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    for i in range(100):
        W, A = rand_i4(), rand_i4()
        model.load_weights(W)
        model.push_tile(A, acc=0)
        got = await drv.full_op(W, A, acc=0)
        expect(model, got, f"random matmul #{i}")


@cocotb.test()
async def test_accumulate_chain_positive_saturation(dut):
    """K-chaining with ACC=1: +128/tile until the +2047 saturation clamp."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[-8, -8], [-8, -8]]
    A = [[-8, -8], [-8, -8]]                  # C = +128 per tile
    model.load_weights(W)
    await drv.load_weights(W)
    for t in range(20):
        acc = 0 if t == 0 else 1
        model.push_tile(A, acc)
        got = await drv.run_tile(A, acc)
        expect(model, got, f"acc-sat tile {t}")
    assert model.R[0] == 2047, "model should be saturated"


@cocotb.test()
async def test_accumulate_chain_negative_saturation(dut):
    """K-chaining to the -2048 rail: -112/tile with directed operands."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[-8, -8], [-8, -8]]
    A = [[7, 7], [7, 7]]                      # C = -112 per tile
    model.load_weights(W)
    await drv.load_weights(W)
    for t in range(20):
        acc = 0 if t == 0 else 1
        model.push_tile(A, acc)
        got = await drv.run_tile(A, acc)
        expect(model, got, f"neg-sat tile {t}")
    assert model.R[0] == -2048


@cocotb.test()
async def test_accumulate_random_chain(dut):
    """40 random tiles accumulated onto one result register file."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = rand_i4()
    model.load_weights(W)
    await drv.load_weights(W)
    for t in range(40):
        A = rand_i4()
        acc = 0 if t == 0 else 1
        model.push_tile(A, acc)
        got = await drv.run_tile(A, acc)
        expect(model, got, f"random acc chain tile {t}")


@cocotb.test()
async def test_weight_rewrite_between_tiles(dut):
    """Weights may be rewritten between tiles without resetting."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    for i in range(10):
        W, A = rand_i4(), rand_i4()
        model.load_weights(W)
        model.push_tile(A, acc=0)
        got = await drv.full_op(W, A, acc=0)
        expect(model, got, f"weight rewrite #{i}")


@cocotb.test()
async def test_readout_restart_and_reread(dut):
    """Readout can exit early, restart at bit 0, and re-read identically."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[3, -1], [4, 2]]
    A = [[-5, 6], [7, -2]]
    model.load_weights(W)
    model.push_tile(A, acc=0)
    await drv.load_weights(W)
    await drv.push_tile(A, 0)
    await drv.idle(2)
    await drv.wait_done()

    # partial read: 10 bits, then exit mid-stream
    dut.ui_in.value = CMD_READ << 6
    for _ in range(10):
        await FallingEdge(dut.clk)
    await FallingEdge(dut.clk)
    dut.ui_in.value = 0                      # exit mid-stream
    await drv.idle(2)

    # two clean full reads must both match the golden bitstream
    r1 = await drv.read_results()
    r2 = await drv.read_results()
    expect(model, r1, "re-read #1")
    expect(model, r2, "re-read #2")


@cocotb.test()
async def test_readout_bitstream_matches_golden(dut):
    """Every one of the 48 serial bits matches the model bit-for-bit."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[-7, 3], [2, -5]]
    A = [[6, -8], [-1, 4]]
    model.load_weights(W)
    model.push_tile(A, acc=0)
    await drv.load_weights(W)
    await drv.push_tile(A, 0)
    await drv.idle(2)
    await drv.wait_done()

    exp_bits = model.result_bitstream()
    dut.ui_in.value = CMD_READ << 6
    for i in range(48):
        await FallingEdge(dut.clk)
        assert uo_bit(dut, UO_RBIT) == exp_bits[i], f"bit {i} mismatch"
    await FallingEdge(dut.clk)
    dut.ui_in.value = 0


@cocotb.test()
async def test_push_while_busy_ignored(dut):
    """Extra pushes during compute are dropped; the tile still checks out."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[2, -3], [1, 4]]
    A = [[-6, 5], [3, -7]]
    model.load_weights(W)
    model.push_tile(A, acc=0)
    await drv.load_weights(W)
    await drv.push_tile(A, 0)

    # spam junk pushes while the FSM is running
    for junk in range(4):
        await FallingEdge(dut.clk)
        dut.ui_in.value = (CMD_PUSH << 6) | (junk & 0xF)
    await drv.idle(2)
    await drv.wait_done()
    got = await drv.read_results()
    expect(model, got, "busy-spam")


@cocotb.test()
async def test_busy_timing(dut):
    """BUSY rises with the first push and falls exactly at DONE."""
    await do_reset(dut)
    drv = TpuDriver(dut)
    await drv.load_weights([[1, 0], [0, 1]])
    assert not uo_bit(dut, UO_BUSY), "idle before pushes"
    await drv.push_tile([[1, 1], [1, 1]], 0)   # 4 back-to-back pushes
    saw_busy = False
    for _ in range(12):
        saw_busy |= bool(uo_bit(dut, UO_BUSY))
        if uo_bit(dut, UO_DONE):
            break
        await FallingEdge(dut.clk)
    assert saw_busy, "BUSY never asserted during compute"
    assert uo_bit(dut, UO_DONE)
    await drv.idle(2)
    assert not uo_bit(dut, UO_BUSY), "BUSY stuck after DONE"


@cocotb.test()
async def test_ena_freeze_gates_logic(dut):
    """ena=0 freezes all sequential logic; commands have no effect."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    W = [[4, -2], [3, 5]]
    A = [[-3, 2], [6, -1]]
    model.load_weights(W)
    model.push_tile(A, acc=0)
    await drv.load_weights(W)

    # freeze mid-queueing: push 2 nibbles, freeze, spam commands, unfreeze
    await FallingEdge(dut.clk)
    dut.ui_in.value = (CMD_PUSH << 6) | (A[0][0] & 0xF)
    await FallingEdge(dut.clk)
    dut.ui_in.value = (CMD_PUSH << 6) | (A[0][1] & 0xF)
    await FallingEdge(dut.clk)
    dut.ui_in.value = 0                        # clear command before freeze
    dut.ena.value = 0
    for _ in range(5):                         # frozen spam: must be ignored
        dut.ui_in.value = (CMD_PUSH << 6) | 0x5
        await FallingEdge(dut.clk)
    dut.ui_in.value = 0                        # no stale command at unfreeze
    dut.ena.value = 1
    await FallingEdge(dut.clk)
    # only 2 nibbles made it; finish the tile properly
    dut.ui_in.value = (CMD_PUSH << 6) | (A[1][0] & 0xF)
    await FallingEdge(dut.clk)
    dut.ui_in.value = (CMD_PUSH << 6) | (A[1][1] & 0xF)
    await FallingEdge(dut.clk)
    dut.ui_in.value = 0
    await drv.idle(2)
    await drv.wait_done()
    got = await drv.read_results()
    expect(model, got, "ena freeze")


@cocotb.test()
async def test_random_session_soak(dut):
    """Random command soak: interleaved weight loads, acc/fresh tiles, reads."""
    await do_reset(dut)
    drv, model = TpuDriver(dut), TpuGoldenModel()
    model.load_weights(rand_i4())
    await drv.load_weights(model.W.tolist())
    for t in range(60):
        action = rng.random()
        if action < 0.15:
            W = rand_i4()
            model.load_weights(W)
            await drv.load_weights(W)
        A = rand_i4()
        acc = 1 if rng.random() < 0.5 else 0
        if acc and rng.random() < 0.3:
            acc = 0                             # occasional fresh overwrite
        model.push_tile(A, acc)
        got = await drv.run_tile(A, acc)
        expect(model, got, f"soak step {t}")
        if rng.random() < 0.25:
            again = await drv.read_results()
            expect(model, again, f"soak re-read {t}")
