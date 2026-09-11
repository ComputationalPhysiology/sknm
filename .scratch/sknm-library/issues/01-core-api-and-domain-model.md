# 01 — Core API and domain model

Parent: [map](../map.md)
Type: grilling
Status: in progress (claimed by henriknf@simula.no)
Blocked by: —

## Question

What is the shape of `sknm`'s public API, and what are the domain's canonical names?

The physics is a graph Laplacian over cell connections. That admits two very different libraries,
and this decision propagates into every other ticket:

- **Reproduction artifact**: hardcoded structured 2D sheets, the two published cell models, the
  paper's parameter names. Smallest surface, matches the C++ reference closely.
- **General library**: an arbitrary cell *graph* (any topology, per-cell geometry, per-connection
  conductance), a pluggable membrane model, with the paper's setups as provided presets.

The recommendation going in is the **general library**, because writing the coupling as a graph
is barely more work than hardcoding a grid and it makes the model genuinely testable — a two-cell
network has an analytic answer, a 1D chain has a known conduction velocity — whereas a
sheet-only API forces every test through a 40×40 hiPSC-CM simulation. This is a recommendation,
not a settled decision; grill it.

Sub-questions to settle:

1. **Names.** What is the thing holding the cells and their connections called — `Tissue`,
   `CellNetwork`, `CellCollection`? What is one edge — `Connection`, `Junction`, `Coupling`?
   The paper says "cell connection" and indexes by `(j,k)`. Resolve against
   `mattpocock-skills:domain-modeling` and write the glossary to `CONTEXT.md`.
2. **What does `import sknm` expose at top level?** Model classes, a solve function, tissue
   constructors, the membrane models, or a deliberately thin surface?
3. **How are the three model variants expressed** — three classes, one class with a `variant`
   enum, or a `model=` argument? Note from the reference C++ that SKNM differs from KNM by one
   scalar factor `λ/(1+λ)` plus dropping a matrix block, and SKNM(uₑ=0) differs from SKNM by
   setting that factor to 1. That is suspiciously close to *one* implementation with two knobs.
4. **How does a membrane model plug in?** A Protocol/ABC with `rhs`/`forward_rush_larsen` and
   state metadata? What contract does it owe (vectorised over cells? in-place?)? This is the
   seam that lets tests use a cheap 2-state model instead of a 25-state one.
5. **Where does geometry live** — per-cell `lx, ly, lz`, `A_m`, `δe`, and per-connection
   `A_{j,k}`, `l_{j,k}`, `G_g^{j,k}`. Are these arrays on the network, or objects per cell?
6. **How are per-cell/per-connection parameter overrides expressed?** Needed for the γ
   gap-junction variation and the β-cell `gKATP` stimulus.

## Facts established (2026-09-10) — do not re-derive

From `references/SKNM_code/` and the paper PDF:

- The two drivers (`KNM_2D_hiPSC.cpp`, `KNM_2D_beta.cpp`) are **byte-identical** in all four
  matrix-assembly functions and in the time loop. They differ only in constants, membrane header,
  and stimulus mechanism. One solver, two presets.
- The matrices are **already pure graph Laplacians** over per-edge conductances; the grid enters
  only as index arithmetic (`idx±1`, `idx±Nx`). A graph formulation is nearly free.
- The variant switch is exactly two things — `matrix_factor = (λ/(1+λ))·dt/(Cm·Am)` vs
  `dt/(Cm·Am)`, and whether the `(0,1)/(1,0)/(1,1)` blocks exist:
  **KNM** = (blocks yes, factor 1); **SKNM** = (no, λ/(1+λ)); **SKNM(uₑ=0)** = (no, 1).
  The fourth combination is not a model.
- **λ** (paper eq. 30) is a global weighted least-squares fit of Gₑ ≈ λG_i over *all* connections,
  weighted by l²/A². Verified the C++ computes exactly this: `Me_x = δₑσₑ` is Gₑ·l/A, and
  `1/(1/(δᵢσᵢ) + Rg·lylz/lx)` is G_i·l/A. λ is a derived property of the conductance set.
- Boundary conditions: homogeneous Neumann for v and uₑ everywhere (free in a graph formulation),
  plus a single Dirichlet ground on uₑ at cell 0. `u_prev` is never updated in the C++, so the
  elliptic block's RHS is identically zero.
- **Both stimuli are the same operation**: override one membrane *parameter* on a subset of cells.
  hiPSC sets `stim_amplitude` = 20; β sets `gkatpbar` = 250 (half of 500).
- Correction to the map's fog note: the β stimulus is the **centre** 5 cells of the left boundary,
  2 columns deep (`x < 2lx && 5ly < y < 10ly`), not the "leftmost 5 cells". hiPSC is
  `x < 2lx && 14ly < y < 25ly` (~11 cells, though the paper's text says 10).
- **Capacitance trap**: the two membrane models use different conventions. hiPSC is
  `dV/dt = -I_tot` (implicit specific Cm = 1 µF/cm²); β is `dv/dt = -I/Cm` with Cm = 5300 fF
  absolute. The network's `Cm·Am` (5.3093e-6 µF) is made to match 5300 fF *by hand*. Nothing
  checks it. Note the splitting means the network step never sees `I_ion` — the membrane seam
  contract is only "advance (v, s) by dt" — so the two capacitances need only agree implicitly.

From gotranx 1.8.0 (installed and exercised, not read off docs):

- `rhs(t, states, parameters)` but schemes are `(states, t, dt, parameters)` — inconsistent order.
- Schemes are **functional**, not in-place; caller does `states[:] = ...`.
- **State-major `(num_states, num_cells)`.** Cell-major does not raise — it returns garbage.
- **State indices are dependency-sorted** (neither declaration order nor alphabetical); parameters
  *are* alphabetical. `state_index(name)` is mandatory — the C++'s hardcoded `V_idx` will not port.
  Consequence: `states[v_index]` is a contiguous **zero-copy row**, so state-major is also the
  layout that makes the network step cheapest.
- Generated numpy modules import **only numpy** — validates committing the generated code.
- **No numba backend** (`--backend` is `numpy|jax`); numba means `njit`-wrapping the output plus
  `--shape single`. Relevant to the map's performance fog.
- `forward_generalized_rush_larsen` is a deprecated alias; the live name is
  `generalized_rush_larsen`.

## Grilling round 1 — posted 2026-09-10, UNANSWERED

Recommendations are mine, not decisions. Nothing below is settled.

1. **Library shape.** (a) reproduction artifact / (b) general library / (c) general core, narrow
   public surface. ➡️ **(c)** — graph internals from day one, v1's documented API is the two
   presets plus a `from_edges` constructor; tests use two-cell and 1D-chain networks with analytic
   answers.
2. **Three variants expressed how?** (a) three classes / (b) one class + `variant` enum /
   (c) two independent knobs. ➡️ **(b)** — the knobs are not independent, so (c) admits nonsense.
3. **What drives the time loop?** (a) `Simulation` with public `.step()` + `.run()` /
   (b) generator / (c) `solve() -> Result` with declarative recording. ➡️ **(a) with a thin (c)**
   — CV measurement needs threshold events and early termination; easier as a callback than a DSL.
4. **λ input or derived?** ➡️ **derived lazily, overridable** — right by default, still pokeable,
   independently testable.
5. **Per-cell / per-connection overrides.** ➡️ **named API + raw-array escape hatch**
   (`set_parameter("gkatpbar", 250, cells=mask)`), because both paper stimuli are that one op.
6. **The two-capacitances trap.** (a) document only / (b) membrane model declares its capacitance,
   network validates / (c) derive one from the other. ➡️ **(b)** — (c) is impossible, the hiPSC
   model has no capacitance to read. This catches the one silent-wrong-answer bug in the design.
7. **Membrane-model seam.** (a) duck-typed gotranx module / (b) `MembraneModel` Protocol +
   `from_gotranx` adapter / (c) ABC. ➡️ **(b)**, `step` specified in-place — it is the only place
   the V-state name and capacitance can be bound to the model they belong to, and it makes a
   hand-written 15-line FitzHugh–Nagumo test model trivial. Honest caveat: in-place buys no
   performance, gotranx allocates internally regardless; it buys ownership clarity.

Deferred to round 2 (depend on the above): naming/glossary and `CONTEXT.md`, the top-level
`import sknm` surface, where geometry lives.

## Answer

_(unresolved)_
