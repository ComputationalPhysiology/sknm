# sknm

A Python implementation of the **Simplified Kirchhoff Network Model** (SKNM) of Jæger & Tveito,
*Scientific Reports* **13**:16434 (2023), together with **KNM** and **SKNM(uₑ=0)** so that the
paper's model comparisons can be reproduced.

The domain is cell-based cardiac electrophysiology: excitable cells coupled through gap junctions,
resolved per cell rather than homogenized into a continuum. See [`CONTEXT.md`](CONTEXT.md) for the
vocabulary this codebase uses.

> **Status: early development.** The model, the solvers and the analysis are implemented, and
> the scripts in [`examples/`](examples) reproduce the paper's hiPSC-CM figures. The pancreatic
> β cell model, and the figures that use it, are not implemented.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Runtime dependencies are **numpy**, **scipy** and **pint**. `matplotlib` arrives with the
`examples` extra; `gotranx` with the `codegen` extra, which is needed only to regenerate the
membrane models (the generated Python is committed).

## Usage

Every dimensional argument carries a unit; a bare number is refused. This reproduces the
conduction velocity of Table S1 of the supplementary, 3.73 cm/s, in about two seconds:

```python
from sknm import Simulation, Variant, analysis, presets
from sknm.membrane import base_model_IM, from_gotranx
from sknm.units import ms, mV

network = presets.hipsc_sheet(40, 40, alpha=1.0)
model = from_gotranx(base_model_IM)
sim = Simulation(network, model, variant=Variant.SKNM, dt=0.02 * ms)
sim.set_parameter("stim_amplitude", presets.hipsc_stimulus_amplitude(40, 40))

path = presets.hipsc_conduction_path(40, 40, alpha=1.0)
recorder = analysis.ActivationRecorder(sim, threshold=-20 * mV, stop_when_activated=path.end)
sim.run(50 * ms, record=(), callback=recorder)

print(analysis.conduction_velocity(recorder, path))
print(recorder.max_upstroke_velocity[presets.hipsc_centre_cell(40, 40)], "V/s")
```

Run the same network as `Variant.KNM` to solve for the extracellular potential as well.

## Examples

Each script in [`examples/`](examples) reproduces one of the paper's hiPSC-CM figures, writing
it to `examples/figures/` and printing the numbers it plotted.

| Script | Reproduces |
|---|---|
| `fig02_travelling_wave.py` | Figure 2 — snapshots of a wave crossing 40×40 cells, KNM against SKNM |
| `fig03_anisotropy.py` | Figure 3 — conduction velocity against cell length-to-width ratio |
| `fig04_gap_junction_variation.py` | Figures 4 and S2 — conduction velocity against gap junction variation |
| `fig05_sources_of_difference.py` | Figure 5 — the two factors that make KNM and SKNM differ |

```bash
python -m pip install -e ".[examples]"
python examples/fig03_anisotropy.py           # a reduced sample, about a minute
python examples/fig03_anisotropy.py --full    # the paper's whole sample
```

Every script is **fast by default**: it sweeps every other point of its parameter range unless
given `--full`. The reduction is fewer points, never cheaper points — each point plotted is
computed on the paper's own 40×40 sheet at its own 0.02 ms time step, so `--full` adds markers
rather than moving them. Results are cached under `examples/results/` and reused, so restyling
a figure costs no simulation; the cache is keyed on the parameters of a run and cannot see that
`sknm` itself has changed, so pass `--no-cache` or delete the directory after changing the
library.

The gap junction draws come from a seeded generator rather than from the paper's own random
numbers, so curves that depend on them sit very close to the published ones rather than on top
of them. The authors' draws can be passed to `presets.vary_conductances` instead, if you have
them.

## Development

```bash
pytest                      # tests
pre-commit run --all-files  # ruff lint, ruff format, mypy
```

Requires Python 3.11 or newer.

## Attribution

This is an **independent Python reimplementation**. It is not the authors' code and is not
endorsed by them. The model, its equations and its reference parameters are due to:

> Jæger, K.H., Tveito, A. *The simplified Kirchhoff network model (SKNM): a cell-based
> reaction–diffusion model of excitable tissue.* Scientific Reports **13**, 16434 (2023).
> <https://doi.org/10.1038/s41598-023-43444-9>

The authors' own C++ implementation is distributed under CC-BY-4.0 and was used as the ground
truth for every numerical choice made here. Please cite the paper above in any work that uses this
package; see [`CITATION.cff`](CITATION.cff).

## Licence

MIT — see [`LICENSE`](LICENSE).
