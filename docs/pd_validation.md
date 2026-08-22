# Local Physical-Design Validation Record

Full OpenLane 2.3.10 flow executed natively (no Docker) with
`scripts/setup_eda.sh` + `scripts/run_pd.sh` against this exact repository
state (sky130A / `sky130_fd_sc_hd`, explicit 160 µm × 100 µm 1×1-tile
floorplan injected locally; the official `tt-gds-action` supplies the tile
geometry from `info.yaml`).

| check | result |
|---|---|
| Flow | `Flow complete.` — all 70 steps, no deferred errors |
| Die area | **160.0 × 100.0 µm** (bbox `0 0 160 100`) — one tile |
| Placement utilization | 77.0 % |
| Magic DRC (GDS) | **COUNT: 0** |
| Netgen LVS | **match** (`Netlists match uniquely`, all corners) |
| Setup slack (9 corners) | WNS = 0 — **no violations** |
| Hold slack (9 corners) | WNS = 0 — **no violations** |
| Achievable frequency | clk period_min 5.06 ns → **fmax ≈ 197 MHz** (target 20 MHz) |
| Max slew / max cap (9 corners) | **0 / 0** violations |
| Antenna | **0** violations (24 diodes + jumpers) |
| Power grid | PSM: all shapes connected; worst IR drop ≈ 26.5 µV |
| GDSII | streamed (Magic + KLayout writers) |

Notes:

* `uio_oe` is driven from the `ena` input (`{8{ena}}`) rather than a
  constant — tie-cell constants merge into the VPWR rail during layout
  extraction and break LVS pin matching under some flow configurations.
  The RTL-level fix is flow-robust (verified: LVS passes with both
  `MAGIC_DEF_LABELS` settings).
* `max_fanout` liberty soft-guideline notes on CTS clock buffers
  (fanout 11–16 vs soft limit 10) remain, as is standard for clock trees;
  electrical validity is proven by slew/cap/setup/hold all clean at every
  corner.

Rerun the validation at any time:

```bash
./scripts/setup_eda.sh          # once (~25 min, ~2.5 GB)
./scripts/run_pd.sh             # flow + scripts/check_tile.py (~4 min)
```
