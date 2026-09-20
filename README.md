# sknm

A Python implementation of the **Simplified Kirchhoff Network Model** (SKNM) of Jæger & Tveito,
*Scientific Reports* **13**:16434 (2023), together with **KNM** and **SKNM(uₑ=0)** so that the
paper's model comparisons can be reproduced.

The domain is cell-based cardiac and islet electrophysiology: excitable cells coupled through gap
junctions, resolved per cell rather than homogenized into a continuum. See
[`CONTEXT.md`](CONTEXT.md) for the vocabulary this codebase uses.

> **Status: early development.** The model, the solvers and the analysis are implemented, and
> the scripts in [`examples/`](examples) reproduce the paper's figures for both of its cell
> types — hiPSC-derived cardiomyocytes and pancreatic β cells. The homogenized bidomain and
> monodomain comparisons (Figures 6–9) are out of scope.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Runtime dependencies are **numpy**, **scipy** and **pint**. `matplotlib` arrives with the
`examples` extra; `gotranx` with the `codegen` extra, which is needed only to regenerate the
membrane models (the generated Python is committed).

Two membrane models ship: `base_model_IM`, an hiPSC-CM model, and `PBM`, the phantom bursting
model of the pancreatic β cell. Both are generated from the `.ode` sources in the repository
root and checked against the authors' C++ elementwise.

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

The β cell setup is the same shape, through `presets.beta_*`, and reproduces Table S4's
0.0243 cm/s:

```python
from sknm import Simulation, Variant, analysis, presets
from sknm.units import ms

network = presets.beta_sheet(15, 15)
sim = Simulation(network, presets.beta_membrane_model(), variant=Variant.SKNM, dt=0.02 * ms)
sim.set_parameter("gkatpbar", presets.beta_stimulus_conductance(15, 15))

path = presets.beta_conduction_path(15, 15)
recorder = analysis.ActivationRecorder(
    sim, threshold=presets.BETA_THRESHOLD, stop_when_activated=path.end
)
sim.run(1000 * ms, record=(), callback=recorder)

print(analysis.conduction_velocity(recorder, path))
```

`ActivationRecorder` has no default threshold. The two cell types activate at −20 mV and
−50 mV, and a β action potential peaks at about −19.5 mV, so the cardiac threshold applied to a
β sheet leaves the measurement cell unactivated. `presets.HIPSC_THRESHOLD` and
`presets.BETA_THRESHOLD` are the two values.

## Examples

Each script in [`examples/`](examples) reproduces one of the paper's figures, writing it to
`examples/figures/` and printing the numbers it plotted.

| Script | Reproduces |
|---|---|
| `fig02_travelling_wave.py` | Figure 2 — snapshots of a wave crossing 40×40 cells, KNM against SKNM |
| `fig03_anisotropy.py` | Figure 3 — conduction velocity against cell length-to-width ratio |
| `fig04_gap_junction_variation.py` | Figures 4 and S2 — conduction velocity against gap junction variation |
| `fig05_sources_of_difference.py` | Figure 5 — the two factors that make KNM and SKNM differ |
| `fig10_beta_travelling_wave.py` | Figure 10 — snapshots of a wave crossing 15×15 β cells |
| `fig11_beta_variable_coupling.py` | Figure 11 — the same, at γ = 1 and 2% extracellular volume |
| `fig12_beta_gap_junction_variation.py` | Figures 12 and S1 — β conduction velocity against gap junction variation |

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

The gap junction draws differ between the two cell types, and the reason is measurable. The
**cardiac** figures seed a generator rather than using the paper's own random numbers: on a
40×40 sheet with a 25-cell conduction path the choice of draws moves a velocity by about 2%, so
those curves sit very close to the published ones rather than on top of them. The authors' draws
can be passed to `presets.vary_conductances` instead, if you have them.

The **β** figures cannot do that. A 15×15 sheet has 420 connections and an 8-cell path, so at
γ = 1 the draws move the velocity by 20% and every seed tried fell below the axis of the
published Figure S1. Those scripts therefore read the authors' own 420 draws, committed with
their provenance under [`examples/data/`](examples/data). No number the test suite asserts
depends on them: every published target is at γ = 0, where the draws cancel out of the formula.

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
