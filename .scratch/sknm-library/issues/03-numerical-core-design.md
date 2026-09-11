# 03 — Numerical core design

Parent: [map](../map.md)
Type: grilling
Status: open
Blocked by: 01

## Question

How is the solver actually structured, and which numerical choices do we inherit from the
reference C++ versus decide for ourselves?

What the reference implementation (`references/SKNM_code/KNM_2D_hiPSC.cpp`) does, as ground truth:

- **Godunov operator splitting.** Step 1 advances the membrane ODEs per cell
  (`forward_rush_larsen`, OpenMP-parallel over cells); step 2 solves the spatial linear system.
  Defaults `dt = dt_ode = 0.02` ms.
- **KNM** is a 2×2 block system (blocks `MiI`, `Mi`, `Mib`, `Mie`) with a block-diagonal
  preconditioner. **SKNM** is the single `MiI` block with no preconditioner — that is the whole
  computational saving.
- The reduction enters through one scalar per cell:
  `matrix_factor[i] = (λ/(1+λ))·dt/(Cm·Am[i])` versus KNM's `dt/(Cm·Am[i])`.
- **λ** is a least-squares ratio over all connections (paper eq. 30):
  `λ = Σ Ge·Gi·(l/A)² / Σ Gi²·(l/A)²`.
- Linear solver **BiCGSTAB**, `abstol=0`, `reltol=sqrt(1e-12)`, `maxiter=50000`.
- Matrices assembled by hand, indexed by `Nx,Ny`; the `.mesh` files are visualization-only.

Open decisions:

1. **Splitting scheme.** Keep Godunov (first-order, matches the paper and its convergence
   tables) or offer Strang as an option? Note the paper's convergence study justifying
   `Δt = 0.02` ms assumes the Godunov/Rush–Larsen combination — changing the scheme invalidates
   the comparison to Tables S1/S4.
2. **Linear solver.** BiCGSTAB to match the reference, or is the SKNM operator symmetric
   positive definite (it is a weighted graph Laplacian plus a diagonal), making **CG** the
   correct and faster choice? For KNM's saddle-point-ish block system, does a sparse direct
   factorisation (`splu`, factored once when conductances are constant in time) beat an
   iterative solve at the problem sizes we care about? Conductances *are* time-independent here,
   which makes prefactorisation very attractive.
3. **KNM block system**: solve monolithically, or eliminate uₑ via a Schur complement?
4. **The Dirichlet corner.** KNM/BD pin uₑ at the lower-left corner for uniqueness. How is that
   expressed on a general graph rather than a rectangle (ticket 01's shape decides this)?
5. **Matrix assembly**: build directly as `scipy.sparse` from the connection list — incidence
   matrix `Aᵀ diag(G) A` is the clean formulation. Confirm it reproduces the C++ stencil.
6. **State layout and dtype.** `(n_cells, n_states)` vs `(n_states, n_cells)` — the latter is
   usually better for a vectorised RHS. float64 throughout?
7. **Stimulus protocols.** hiPSC-CM: 20 µA/cm² to the centre 10 cells on the left boundary.
   β cell: halve `gKATP` in the centre 5 left-boundary cells. How are these expressed?

## Answer

_(unresolved)_
