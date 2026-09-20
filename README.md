# sknm

A Python implementation of the **Simplified Kirchhoff Network Model** (SKNM) of Jæger & Tveito,
*Scientific Reports* **13**:16434 (2023), together with **KNM** and **SKNM(uₑ=0)** so that the
paper's model comparisons can be reproduced.

The domain is cell-based cardiac electrophysiology: excitable cells coupled through gap junctions,
resolved per cell rather than homogenized into a continuum. See [`CONTEXT.md`](CONTEXT.md) for the
vocabulary this codebase uses.

> **Status: early development.** The model, the solvers and the analysis are implemented and
> reproduce the paper's published conduction velocity; the example scripts are not written yet.

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
