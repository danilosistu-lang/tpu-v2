# Test

RTL simulation with [cocotb](https://www.cocotb.org/) + Icarus Verilog:

```bash
pip install -r requirements.txt
make            # RTL: 16 tests through the tb.v wrapper (must be 100%)
make GATES=yes  # gate-level, after the gds action copied gate_level_netlist.v
```

`test.py` drives the full serial command protocol of
`tt_um_tpu_accelerator` (weights → activations → readout) and compares
every result **bit-exact** against a NumPy golden model with identical
12-bit saturating semantics. Coverage: random/corner INT4 matmuls,
saturation chains, weight rewrites, readout restart semantics, busy/ena
behaviour, and a random-session soak. The VCD from the run (`tb.vcd`) can
be viewed with gtkwave (layout in `tb.gtkw`) or surfer.
