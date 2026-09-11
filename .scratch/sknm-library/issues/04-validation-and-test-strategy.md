# 04 — Validation and test strategy

Parent: [map](../map.md)
Type: grilling
Status: open
Blocked by: 01, 02

## Question

What does "this implementation is correct" mean concretely, and what does the test suite look
like?

The hard part of testing a PDE/ODE solver is that most obvious tests are either trivial or
require a 40×40 simulation. Decisions to make:

1. **Published reference values as regression targets.** The supplementary gives exact numbers
   we can assert against — this is the strongest validation available without building the C++:
   - Table S1 (hiPSC-CM, KNM *and* SKNM): at `Δt = 0.02` ms, **CV = 3.73 cm/s**,
     **(dv/dt)max = 18.81 V/s**. Full Δt sweep from 1 ms down to 0.002 ms.
   - Table S4 (β cells): at `Δt = 0.02` ms, **CV = 0.0243 cm/s**, **(dv/dt)max = 0.340 V/s**.
   - Default hiPSC-CM CV should be ≈ 4 cm/s (the paper tunes `Gg = 2e-4` mS for this).
   What tolerance is honest for a from-scratch reimplementation — 1%, 2%, 5%?
2. **Analytic / small-network tests** that need no membrane model at all:
   - two cells, no ionic current: membrane potentials equilibrate exponentially at a rate set
     by `Gi` and `Cm` — closed form.
   - a network with all conductances equal and uniform initial state must stay uniform.
   - λ from eq. (30) on a uniform network must satisfy `Ge = λ·Gi` exactly, and there
     **SKNM must equal KNM to solver tolerance** — this is the paper's central claim (Fig. 2)
     and makes an excellent test.
   - `SKNM(uₑ=0)` must equal SKNM when `λ/(1+λ) → 1`, i.e. large λ.
3. **A cheap membrane model for unit tests.** FitzHugh–Nagumo or similar, so the bulk of the
   suite doesn't drag a 25-state model through every case. Depends on ticket 01's plug-in seam.
4. **Matrix-level tests**: assembled sparse operator vs a dense brute-force build on a tiny
   network; symmetry; row sums (Laplacian rows sum to zero except where pinned).
5. **Convergence tests**: does the error actually shrink as Δt decreases, reproducing the trend
   of Table S1? Slow — mark them, or run reduced.
6. **Cross-validation against the MATLAB reference** (Zenodo 15798522, pure MATLAB, no MFEM) —
   worth doing, or is matching the published tables enough?
7. **Test runtime budget.** What must `pytest` cost with no flags? Which tests are marked slow?

## Answer

_(unresolved)_
