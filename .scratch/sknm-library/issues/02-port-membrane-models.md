# 02 — Port the two membrane models to Python

Parent: [map](../map.md)
Type: task
Status: open
Blocked by: —

## Question

Nothing to decide — the strategy is settled (gotranx codegen, generated Python committed,
runtime stays numpy + scipy). This is the manual work that unblocks every simulation ticket:
without the membrane models there is no `I_ion(v, s)` and no `F(s, v)`, so nothing can be run
or validated.

Do the work:

1. **Reconstruct `.ode` source** for both models from the Gotran-*generated* C++ in
   `references/SKNM_code/` (no `.ode` was ever published — only the output):
   - `base_model_IM.h` — hiPSC-CM wildtype, **25 states**
     (`m, j, mL, hL, Xr1, Xr2, x_Ks, q, r, d, f, f_Ca, xf, Na_i, r_RyR, cn, cc, cd, csl, cs,
     bc, bd, bs, bsl, V_m`), **83 parameters**. 1348 lines — this is the real work.
   - `PBM.h` — Bertram & Sherman phantom bursting β cell, **5 states**
     (`v, n, c, a, cer`), **25 parameters**. RHS is ~20 lines; nearly transcribable by hand.
2. **Install gotranx** (`pip install gotranx`; not currently present) and generate Python with a
   Rush–Larsen scheme — the paper uses first-order Rush–Larsen, gotranx offers
   `forward_generalized_rush_larsen` and `hybrid_rush_larsen`.
3. **Commit the generated Python** under the package so there is no runtime gotranx dependency.
   Keep the `.ode` files in-tree next to it, and record the exact regeneration command.
4. **Validate** each ported model as a *single cell*, before any tissue coupling exists:
   - integrate one cell and compare the action potential against the C++ `rhs` output. The C++
     can be compiled standalone for this — the headers have no MFEM dependency, only the
     drivers do.
   - cross-check the β-cell model against the published Bertram & Sherman equations.
   - a useful independent check: Zenodo 21914331 carries a MATLAB version of the same hiPSC-CM
     base-model family.

Record in the answer: how faithful the round-trip proved, any discrepancies found, the
regeneration command, and where the generated files landed.

## Progress (2026-09-10 — partial, ticket still open)

Step 1 done for the hiPSC-CM model; steps 2 and 4 done for it too. β cell and steps 3 not started.

**`base_model.ode` (user-supplied) is the WRONG parameterisation.** Same model family — 25 states,
same names (bar `f_Ca` → `f_Ca_B`), same `dV_m_dt = -I_tot - i_Stim` — but all 25 initial state
values and 29 of 71 shared parameters differ, including *every* channel conductance
(`g_Na` 0.36 vs 1.24186, `g_K1` 0.27 vs 0.0454092, …), `Temp` (310 K vs 296 K) and `Nao`/`Ko`/`ce`.
It is also structurally different: it has 5 Q10 parameters where the reference has 13 applied
inside the gate kinetics, and carries `lambda_*` scaling factors instead. Consistent, because both
define `Qpow = -31 + Temp/10`, so at 310 K every Q10 factor collapses to 1 and is removable; the
reference at 296 K needs them live. It also lacks `SQT1` and adds `g_KATP`, `epi`, `Na_sl`.
Separately it does not parse with gotranx 1.8.0 — lines 233–234 (`aj`, `bj`) use legacy gotran
`(cond)*(a) + ~(cond)*(b)` syntax. **Kept as-is; not used.**

**Transcribed `base_model_IM.h` → `base_model_IM.ode`** (repo root, 360 lines, 21 components,
25 states, 83 parameters, 108 assignments) via `tools/c2ode.py`. That converter turns each C
expression into valid Python, parses it with `ast`, rewrites the tree (ternary → `Conditional`,
comparisons → `Lt`/`Le`/…, `&&` → `And`, `std::pow` → `**`, `t` → `time`) and `ast.unparse`s it,
so operator precedence is Python's problem, not hand-written regex's. Rerun with
`python3 tools/c2ode.py`.

**Validation — faithful.** `tools/validate_base_model_IM.py` compiles the original C
(`tools/rhs_driver.cpp`, needs no MFEM) and compares `rhs` over 500 randomised states with `V_m`
swept across [-90, 40] mV to exercise both branches of every conditional, and `t` sampled across
the stimulus window: **12,500/12,500 comparisons agree, max relative difference 3.4e-13.**
Trajectory check: explicit Euler, 50,000 steps at dt = 1 µs, agrees to 2.7e-5 relative
(`V_m` 17.171314 C vs 17.171300 Python). C explicit Euler at dt = 0.02 NaNs exactly as Python's
does. The equations are identical.

Regeneration command:
`gotranx ode2py base_model_IM.ode -o <dest>.py --scheme generalized_rush_larsen`

### Finding that needs a decision: gotranx's Rush–Larsen is ~40× less stable here

Old gotran computes a **fully expanded** `dV_m_dt_linearized` (chain rule through all 13 currents),
so `V_m` gets a genuine exponential update. gotranx linearises the *immediate* expression only, and
`d(-I_tot - i_Stim)/dV_m = 0`, so `V_m` silently falls back to forward Euler. Measured over 300 ms
against a C explicit-Euler dt = 1e-5 reference:

| scheme | dt (ms) | result |
|---|---|---|
| C `forward_rush_larsen` | 0.02 (paper's `dt_ode`) | stable, max err 0.64 mV |
| gotranx `generalized_rush_larsen` | 0.02, 0.01, 0.005, 0.002 | **blows up at t = 1 ms** |
| gotranx `generalized_rush_larsen` | 0.001 | stable, max err 2.97 mV |
| gotranx `generalized_rush_larsen` | 0.0005 | stable, max err 0.017 mV |
| gotranx `generalized_rush_larsen` | 0.00025 | stable, max err 0.0038 mV |

It converges cleanly — it is not wrong, just far more restrictive. Matching the C's accuracy needs
dt ≈ 0.0005, i.e. **40 ODE substeps per 0.02 ms network step**. That is a direct 40× hit on the
membrane step, which the map already flags as the likely bottleneck. Candidate responses, to
decide: (a) accept K = 40 substepping; (b) inline the current expressions into `dV_m_dt` in the
`.ode` so gotranx can differentiate through them — mechanical, `tools/c2ode.py` could emit it, but
untested and produces a huge expression; (c) ask upstream whether deep linearisation is available
(the user is gotranx's author). **Raised with the user; unanswered.**

### Re-measured 2026-09-14 — the stability decision is resolved, no substepping needed

The user took option (c): gotranx now linearises on the true Jacobian diagonal (deep
linearisation, gotranx 2.0.0) and guards removable singularities by default (PR #285, merged to
gotranx `main` at `70171e8`). Re-ran the table above with that gotranx, same protocol (single cell,
`stim_amplitude` = 20, 300 ms, `V_m` every 1 ms, reference = C explicit Euler dt = 1e-5; the
recomputed reference is bit-identical to the 2026-09-10 one):

| scheme | dt (ms) | max err vs reference |
|---|---|---|
| C `forward_rush_larsen` | 0.02 | 0.6383 mV |
| gotranx `generalized_rush_larsen` | **0.02** | **0.6381 mV — stable** |
| gotranx `generalized_rush_larsen` | 0.01 | 0.4146 mV |
| gotranx `generalized_rush_larsen` | 0.005 | 0.2581 mV |
| gotranx `generalized_rush_larsen` | 0.002 | 0.1201 mV |
| gotranx `generalized_rush_larsen` | 0.001 | 0.06046 mV |
| gotranx `generalized_rush_larsen` | 0.0005 | 0.02947 mV |
| gotranx `generalized_rush_larsen` | 0.00025 | 0.01432 mV |

Stable at every dt, first order (error halves with dt below 0.002). At the paper's dt = 0.02 it
matches the C Rush–Larsen to 0.0002 mV. **Decision: one ODE step per 0.02 ms network step (K = 1);
the 40× substepping concern is gone.** Candidate (b) (inlining currents into `dV_m_dt`) is moot.

Details worth keeping:

- Singularity guarding adds exactly one guard to this model: `ibarca_j` (GHK-type Ca flux) for
  `|V_m| < 0.0193` mV. Guards on vs off changes the 300 ms trajectory by 7.7e-14 mV and costs
  ~10% per step (68–72 vs 64 µs/step, single cell, numpy) because `numpy.where` evaluates both
  branches.
- `rhs` validation re-run with the current gotranx: still **12,500/12,500, max rel diff 3.4e-13**.
  None of its random states falls inside the guard window, so checked that separately: Python and
  C agree to 9.7e-13 at |V_m| = 0.019 and 0.001 mV; the gap grows as 1/V_m below that (1.1e-10 at
  1e-6, 1.4e-7 at 1e-9) and the C is non-finite at exactly 0 — the cancellation the guard removes.
- Tooling is now self-contained (it previously imported a module from a dead session's `/tmp`):
  - `python3 tools/validate_base_model_IM.py` — `rhs` vs C, elementwise.
  - `python3 tools/stability_base_model_IM.py` — the table above (~10 min; C reference cached in
    `/tmp/sknm-tools-build`). Uses `tools/conv_driver.cpp`.
- Regeneration command unchanged; note it now guards singularities by default
  (`--no-remove-singularities` to opt out):
  `gotranx ode2py base_model_IM.ode -o <dest>.py --scheme generalized_rush_larsen`
- Ticket 01's "Facts established" were measured against gotranx **1.8.0**; the scheme signatures,
  state layout and backend list there have not been re-checked against the current gotranx.

### Remaining

- β-cell `PBM.h` → `.ode`. A throwaway hand transcription was already validated qualitatively
  (235 upstrokes, V between −66.3 and −20.4 mV, genuine phantom bursting) but was **not** checked
  against the C; redo it with `tools/c2ode.py` and the same `rhs` comparison.
- Step 3 (commit generated Python into the package) — blocked on the package existing at all.

## Answer

_(unresolved)_
