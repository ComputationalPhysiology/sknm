# 06 — Example and figure catalogue

Parent: [map](../map.md)
Type: grilling
Status: open
Blocked by: 01, 03

## Question

Which example scripts exist, what does each reproduce, and how are they structured?

Settled already: fast-by-default (~2 min, reduced parameters, qualitatively the same figure)
with a `--full` flag for paper-exact runs, and simulation results cached so replotting is
instant and separable from solving.

The in-scope figures, all KNM/SKNM (Figs 6–9 are out of scope — see the map):

| Figure | Content | Cost at paper parameters |
|---|---|---|
| **2** | Travelling wave snapshots, 40×40 hiPSC-CM, KNM vs SKNM, t = 25/30/35 ms | 2 sims |
| **3** | CV vs cell length:width ratio α ∈ [1,4], at δe = 50/20/10/2 % | ~40 sims |
| **4** | CV vs gap-junction variation γ ∈ [0,1], at the same four δe | ~48 sims |
| **5** | F(λ) vs γ; max(uₑ)−min(uₑ) vs δe | cheap + ~5 KNM sims |
| **10** | β-cell snapshots, 15×15, t = 0.1/0.3/0.5/0.7 s, KNM vs SKNM | 2 sims |
| **11** | Same, with γ = 1 and 2 % extracellular volume | 2 sims |
| **12** | β-cell CV vs γ, four δe | ~48 sims |
| **S1** | Fig 12 plus SKNM(uₑ=0) | reuses 12 |
| **S2** | Fig 4 plus SKNM(uₑ=0) | reuses 4 |

Open decisions:

1. **One script per figure, or grouped?** `fig02_travelling_wave.py` etc. is obvious to navigate
   and matches "reproduce the figures"; grouping shares expensive sweeps (Fig 4 and S2 are the
   same sweep, as are 12 and S1).
2. **Reduced-parameter defaults**: what exactly gets reduced — sheet size, number of γ/α points,
   simulated duration? Constraint: the figure must stay *qualitatively* right, not a toy.
3. **Cache format and location.** `.npz` under a gitignored `examples/results/`? Cache
   invalidation keyed on a parameter hash? Are any cached results committed so a reader sees the
   figures without running anything?
4. **CLI**: argparse per script, a shared `examples/_common.py`, or a single `python -m
   sknm.examples` entry point? How is `--full` wired?
5. **Figure styling** — consult the `dataviz` skill. Match the paper's look (small multiples with
   "50% E"/"20% E"/… panel titles, KNM solid + SKNM dotted), or use a cleaner house style?
   Reproduction argues for matching.
6. **The random θ_{j,k} draws.** The paper reuses one fixed set across all γ values and both
   models, and the actual numbers are in `references/SKNM_code/random_picks/`. Do we consume
   those files verbatim (exact reproduction) or seed a RNG (self-contained)? Consuming them is
   the only way to match the published curves point-for-point.
7. **Are examples smoke-tested in CI**, at reduced settings?

## Answer

_(unresolved)_
