"""The solver seam: turning an assembled matrix into something that solves against it.

Every solver here is asserted against `DirectSolver`, which is exact: a sparse LU of an SPD
matrix has no tolerance to tune, so it is the reference the iterative ones are measured by
rather than a fourth opinion. The tests that matter are therefore comparisons, not absolute
values -- and the one about `maxiter` is about what happens when a solver *cannot* answer.
"""

import dataclasses

import numpy as np
import pytest
import scipy.sparse as sp

from sknm import Variant, assemble, sheet
from sknm.linalg import (
    BiCGSTABSolver,
    CGSolver,
    ConvergenceError,
    DirectSolver,
    Solver,
    as_solver,
)
from sknm.units import cm, kohm, mS, ms, um

DELTA_E = 0.2
GG = (1 / (5e3 * kohm)).to(mS)
MEMBRANE_AREA = 1.8e-5 * cm**2
DT = 0.02 * ms

SOLVERS = [DirectSolver(), CGSolver(), BiCGSTABSolver()]
ITERATIVE = [CGSolver(), BiCGSTABSolver()]
VARIANTS = list(Variant)


@pytest.fixture
def varied_sheet():
    """A small anisotropic sheet with a different membrane area on every cell.

    Anisotropic so that a mistake which confuses the two directions cannot hide, and varied in
    area so the capacitive diagonal is a genuine scaling rather than a multiple of the identity.
    """
    nx, ny = 4, 5
    areas = np.linspace(1.0, 2.5, nx * ny) * MEMBRANE_AREA
    return sheet(
        nx,
        ny,
        lx=21 * um,
        ly=14 * um,
        lz=(1 + DELTA_E) * 14 * um,
        delta_e=DELTA_E,
        sigma_i=4.0 * mS / cm,
        sigma_e=20.0 * mS / cm,
        Gg=GG,
        membrane_area=areas,
    )


def ramp(n_cells):
    """A deterministic non-constant right-hand side, as a membrane potential in mV."""
    return np.linspace(-80.0, 20.0, n_cells)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize("variant", VARIANTS)
def test_every_solver_solves_the_assembled_system(varied_sheet, variant, solver):
    operator = assemble(varied_sheet, dt=DT, variant=variant)
    rhs = operator.rhs(ramp(varied_sheet.n_cells))

    solution = solver.factorize(operator.matrix)(rhs)

    residual = operator.matrix @ solution - rhs
    assert np.linalg.norm(residual) <= 1e-6 * np.linalg.norm(rhs)


@pytest.mark.parametrize("solver", ITERATIVE)
@pytest.mark.parametrize("variant", VARIANTS)
def test_the_iterative_solvers_agree_with_the_exact_one(varied_sheet, variant, solver):
    """Driven hard enough, an iterative solve reproduces the exact one.

    `rtol` bounds the residual, not the solution, and the KNM system is conditioned well
    enough to turn its default 1e-6 residual into a 3e-4 disagreement on the extracellular
    entries, which are near zero. Tightening the tolerance is what makes this a statement
    about the two solvers computing the same thing rather than about a condition number.
    """
    solver = dataclasses.replace(solver, rtol=1e-13)
    operator = assemble(varied_sheet, dt=DT, variant=variant)
    rhs = operator.rhs(ramp(varied_sheet.n_cells))

    exact = DirectSolver().factorize(operator.matrix)(rhs)
    approximate = solver.factorize(operator.matrix)(rhs)

    np.testing.assert_allclose(approximate, exact, rtol=1e-6, atol=1e-8 * np.abs(exact).max())


@pytest.mark.parametrize("solver", SOLVERS)
def test_one_factorization_serves_many_right_hand_sides(varied_sheet, solver):
    """The whole point of the seam: prepare once, solve repeatedly against the same matrix."""
    operator = assemble(varied_sheet, dt=DT, variant=Variant.SKNM)
    solve = solver.factorize(operator.matrix)

    for scale in (1.0, -3.0, 17.0):
        rhs = scale * operator.rhs(ramp(varied_sheet.n_cells))
        residual = operator.matrix @ solve(rhs) - rhs
        assert np.linalg.norm(residual) <= 1e-6 * np.linalg.norm(rhs)


# --- Not converging, and saying so --------------------------------------------------------


@pytest.mark.parametrize("solver", ITERATIVE)
def test_a_solver_that_runs_out_of_iterations_reports_instead_of_returning(varied_sheet, solver):
    """One iteration cannot solve this, and the answer to that is an error, not a number.

    scipy reports non-convergence in a return code that is easy to drop on the floor, which
    would leave a simulation running on a wrong potential for the rest of its length.
    """
    operator = assemble(varied_sheet, dt=DT, variant=Variant.KNM)
    solve = dataclasses.replace(solver, maxiter=1).factorize(operator.matrix)

    with pytest.raises(ConvergenceError, match="did not converge"):
        solve(operator.rhs(ramp(varied_sheet.n_cells)))


def test_the_non_convergence_message_says_how_far_it_got(varied_sheet):
    operator = assemble(varied_sheet, dt=DT, variant=Variant.KNM)
    solve = CGSolver(maxiter=1).factorize(operator.matrix)

    with pytest.raises(ConvergenceError) as failure:
        solve(operator.rhs(ramp(varied_sheet.n_cells)))

    message = str(failure.value)
    assert "CGSolver" in message
    assert "relative residual" in message
    assert "1 iterations" in message
    assert "DirectSolver" in message


def test_enough_iterations_converge_where_one_did_not(varied_sheet):
    """The failure above is about `maxiter`, not about the system being unsolvable."""
    operator = assemble(varied_sheet, dt=DT, variant=Variant.KNM)
    rhs = operator.rhs(ramp(varied_sheet.n_cells))

    solution = CGSolver().factorize(operator.matrix)(rhs)

    assert np.linalg.norm(operator.matrix @ solution - rhs) <= 1e-6 * np.linalg.norm(rhs)


# --- Preconditioning ----------------------------------------------------------------------


def iterations_needed(preconditioner, matrix, rhs, limit=200):
    """The smallest `maxiter` that converges, found by trying them.

    Measured rather than written down: the count depends on the scipy version, and it is only
    the ordering between preconditioners that is a property of the preconditioners themselves.
    """
    for iterations in range(1, limit):
        try:
            CGSolver(maxiter=iterations, preconditioner=preconditioner).factorize(matrix)(rhs)
        except ConvergenceError:
            continue
        return iterations
    return limit


def test_a_preconditioner_reduces_the_iterations_needed(varied_sheet):
    """The only thing a preconditioner does is change the iteration count.

    Nothing about the answer can tell a preconditioner that was applied from one that was
    built and then dropped, or from one assembled upside down, because the solve converges to
    the same place either way. The count is the whole observable.
    """
    operator = assemble(varied_sheet, dt=DT, variant=Variant.KNM)
    rhs = operator.rhs(ramp(varied_sheet.n_cells))
    plain = iterations_needed("none", operator.matrix, rhs)

    assert iterations_needed("ilu", operator.matrix, rhs) < plain
    assert iterations_needed("jacobi", operator.matrix, rhs) <= plain


@pytest.mark.parametrize("preconditioner", ["none", "jacobi", "ilu"])
def test_every_preconditioner_reaches_the_same_answer(varied_sheet, preconditioner):
    """A preconditioner changes how many iterations are needed, never what they converge to."""
    operator = assemble(varied_sheet, dt=DT, variant=Variant.SKNM)
    rhs = operator.rhs(ramp(varied_sheet.n_cells))
    exact = DirectSolver().factorize(operator.matrix)(rhs)

    solver = CGSolver(rtol=1e-13, preconditioner=preconditioner)
    solution = solver.factorize(operator.matrix)(rhs)

    np.testing.assert_allclose(solution, exact, rtol=1e-6)


def test_an_unknown_preconditioner_is_refused(varied_sheet):
    operator = assemble(varied_sheet, dt=DT, variant=Variant.SKNM)

    with pytest.raises(ValueError, match="preconditioner must be one of"):
        CGSolver(preconditioner="multigrid").factorize(operator.matrix)


# --- Naming a solver ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [("direct", DirectSolver), ("cg", CGSolver), ("bicgstab", BiCGSTABSolver)],
)
def test_a_solver_can_be_named_by_a_string(name, expected):
    assert isinstance(as_solver(name), expected)


def test_an_object_is_passed_through_unchanged():
    solver = CGSolver(maxiter=17)

    assert as_solver(solver) is solver


def test_an_unknown_solver_name_is_refused():
    with pytest.raises(ValueError, match="solver must be one of"):
        as_solver("gmres")


def test_something_that_cannot_solve_is_refused():
    with pytest.raises(TypeError, match="must have a factorize method"):
        as_solver(object())


# --- The objects themselves ---------------------------------------------------------------


def test_the_direct_solver_solves_the_matrix_it_was_given_and_not_its_transpose():
    """Every matrix this package assembles is symmetric, which hides a transpose completely.

    `DirectSolver` is a general sparse LU and is public, so a caller may hand it a matrix that
    is not symmetric -- and then which of the two it factorized is the whole answer.
    """
    matrix = sp.csr_array(np.array([[2.0, 1.0, 0.0], [0.0, 3.0, 1.0], [0.0, 0.0, 4.0]]))
    rhs = np.array([1.0, 2.0, 3.0])

    solution = DirectSolver().factorize(matrix)(rhs)

    np.testing.assert_allclose(matrix @ solution, rhs, atol=1e-12)
    assert not np.allclose(matrix.T @ solution, rhs)


@pytest.mark.parametrize("solver", SOLVERS)
def test_every_solver_satisfies_the_seam(solver):
    assert isinstance(solver, Solver)


@pytest.mark.parametrize("solver", SOLVERS)
def test_solvers_are_frozen(solver):
    with pytest.raises(dataclasses.FrozenInstanceError):
        solver.rtol = 1e-3


def test_the_iterative_defaults_are_the_documented_ones():
    """`maxiter` is a deliberate choice: scipy's own default is `10 * n`, which never stops."""
    for solver in (CGSolver(), BiCGSTABSolver()):
        assert solver.rtol == 1e-6
        assert solver.atol == 0.0
        assert solver.maxiter == 1000
        assert solver.preconditioner == "none"
