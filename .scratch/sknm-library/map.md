# Map: SKNM Python library

Label: `wayfinder:map`

## Destination

A working, tested, pip-installable Python package `sknm` (src layout, modern `pyproject.toml`)
implementing the **Simplified Kirchhoff Network Model** of Jæger & Tveito, *Sci Rep* 13:16434
(2023) — together with **KNM** and **SKNM(uₑ=0)** so the paper's model comparisons can be made —
plus a unit-test suite and a `examples/` folder of runnable scripts that reproduce the
manuscript's KNM/SKNM figures.

Done when: `pip install -e .` works, `import sknm` works, `pytest` passes, and every example
script listed in [06-example-and-figure-catalogue](issues/06-example-and-figure-catalogue.md)
runs and emits its figure.

## Notes

**Domain**: computational cardiac/islet electrophysiology. Cell-based (not homogenised)
reaction–diffusion over a network of cells coupled by gap junctions. Read the two PDFs in
`references/` before touching the maths — the main paper for the model equations (1)–(37) and
the supplementary for the conductance derivations (S1), convergence tables (S2), SKNM(uₑ=0)
(S3), and the BD↔KNM equivalence proof (S4).

**Reference implementation**: `references/SKNM_code/` — the authors' C++/MFEM code, CC-BY-4.0.
Ground truth for every numerical choice. Note that MFEM is *not* structurally required: the
sparse matrices are hand-assembled in the drivers and the `Meshes/` are visualization-only.
A second, pure-MATLAB SKNM implementation exists at Zenodo 15798522 (Jæger, Louch & Tveito 2025)
and is an easier cross-validation target than the C++.

**Attribution**: the port is a derivative of CC-BY-4.0 material. README and `pyproject.toml`
must credit Jæger & Tveito and cite the paper DOI 10.1038/s41598-023-43444-9.

**Skills every session should consult**: `mattpocock-skills:grilling` and
`mattpocock-skills:domain-modeling` for decision tickets; `mattpocock-skills:tdd` and
`superpowers:test-driven-development` once implementation tickets graduate;
`mattpocock-skills:codebase-design` when shaping module seams; `dataviz` for the figure scripts.

**Standing preferences** (settled with the user before charting, not up for re-litigation):

- Execution is carried *inside* this map — tickets end in working code, not only a spec.
  This overrides wayfinder's plan-only default.
- Scope is **SKNM + KNM + SKNM(uₑ=0)**. Bidomain/monodomain is out of scope (see below).
- Membrane models come from **gotranx** codegen with the generated Python **committed**, so the
  runtime dependency set stays **numpy + scipy only**. matplotlib is an examples/docs extra.
- Example scripts are **fast by default** (~2 min, reduced parameters, qualitatively the same
  figure) with a `--full` flag for paper-exact runs, and results cached so replotting is instant.

## Decisions so far

<!-- one line per closed ticket: gist + link. Detail lives in the ticket, never here. -->

_(none yet — charting session only)_

## Not yet specified

Fog: in scope, but not yet sharp enough to ticket. Graduates as the frontier advances.

- **The implementation tickets themselves.** Deliberately unspecified until
  [01-core-api-and-domain-model](issues/01-core-api-and-domain-model.md) and
  [03-numerical-core-design](issues/03-numerical-core-design.md) close — the module seams decide
  how the build splits into sessions. Expect roughly: matrix assembly, the time stepper, the
  three model variants, the tissue/graph constructors, the figure scripts.
- **How to treat Table 1 (CPU scaling to 50,625 cells).** It is a table, not a figure, and a
  NumPy/SciPy port will not match optimised C++/OpenMP timings. Open question whether we
  reproduce the *scaling behaviour* (SKNM ~O(N), KNM ~O(N^1.5)) rather than the absolute
  seconds, or omit it. Depends on how fast the implementation actually turns out.
- **Performance work.** Whether the vectorised-NumPy membrane step is fast enough, or whether
  numba/gotranx's numba-friendly backend is needed for the 25-state hiPSC-CM model across
  ~1600–50,000 cells. Cannot be judged before there is something to profile.
- **Documentation beyond the README**, and whether the package is published to PyPI (the name
  `sknm` is currently free).
- **β-cell stimulation protocol details** — the paper stimulates by halving `gKATP` in the
  leftmost 5 cells, which needs per-cell parameter overrides. Whether that generalises into the
  public API or stays an example-level detail depends on ticket 01.

## Out of scope

Ruled beyond the destination. Never graduates; returns only as a fresh effort.

- **Bidomain (BD) and monodomain (MD) models — Figures 6, 7, 8, 9 and Table 2.** These are
  homogenised PDE models needing separate 2D FEM/finite-difference machinery, Gmsh meshes, and
  orders of magnitude more compute (the paper's own BD run for 50,625 cells was estimated at
  10.7 days). They exist in the paper only to argue that SKNM:KNM behaves like the familiar
  MD:BD. A package called `sknm` does not need them. Reference code is nonetheless available at
  `references/SKNM_code/BD_2D_hiPSC.cpp` should this ever be revisited.
- **Adding BD/MD via the Supplementary S4 equivalence.** Considered and rejected: at
  Δx = cell size the S4 finite-difference scheme is *provably identical* to KNM/SKNM, so it
  would produce Figures 6–9 that duplicate Figures 2–5 rather than independently reproducing
  the published FEM results. Correct code giving the wrong answer to "did we reproduce Fig 7".
- **The EMI model** and any sub-cellular spatial resolution.
