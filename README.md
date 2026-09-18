# sknm

A Python implementation of the **Simplified Kirchhoff Network Model** (SKNM) of Jæger & Tveito,
*Scientific Reports* **13**:16434 (2023), together with **KNM** and **SKNM(uₑ=0)** so that the
paper's model comparisons can be reproduced.

The domain is cell-based cardiac electrophysiology: excitable cells coupled through gap junctions,
resolved per cell rather than homogenized into a continuum. See [`CONTEXT.md`](CONTEXT.md) for the
vocabulary this codebase uses.

> **Status: early development.** The public API is settled but largely unimplemented.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Runtime dependencies are **numpy** and **scipy** only. `matplotlib` arrives with the `examples`
extra; `gotranx` with the `codegen` extra, which is needed only to regenerate the membrane models
(the generated Python is committed).

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
