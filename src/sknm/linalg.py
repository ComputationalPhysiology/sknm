"""Solvers for the assembled system, behind one seam: matrix in, solve function out.

A simulation factorizes its operator once and then solves against the same matrix at every
step, because the conductances do not change with time. That two-phase shape -- an expensive
preparation followed by many cheap solves -- is what `factorize` expresses, and it is the only
thing a solver has to provide:

```python
solve = DirectSolver().factorize(operator.matrix)
x = solve(operator.rhs(v_prev))
```

`DirectSolver` is the default. All three systems are symmetric positive definite, so a sparse
LU has no tolerance to tune and returns the same answer every run; the iterative solvers save
at most a small fraction of a step and cost a convergence criterion, so they are the escape
hatch for networks whose factorization does not fit in memory rather than the fast path.

Solvers are objects rather than a string and an options dictionary, so that `maxiter` cannot
appear to mean something for a direct solve and so that a type checker can see the fields. The
string shorthand ``solver="cg"`` is accepted wherever one is taken, and names the
zero-argument object.

Preconditioners are available and default to none. On the systems this package assembles,
Jacobi -- the reference implementation's own choice -- measurably helps neither variant, and an
incomplete LU is a net loss on the simplified system and does not finish on the full one. The
default `maxiter` of 1000 replaces scipy's ``10 * n``, which on a large network amounts to
never giving up.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, get_args, runtime_checkable

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

#: What `factorize` returns: a function taking a right-hand side to a solution.
Factorization = Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]

Preconditioner = Literal["none", "jacobi", "ilu"]

#: Column permutations SuperLU accepts, which decide how much fill a factorization produces.
ColumnPermutation = Literal["COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"]


class ConvergenceError(RuntimeError):
    """An iterative solver stopped before reaching its tolerance.

    Its own type, so that a sweep over many simulations can catch it and carry on knowing
    exactly what failed, and a `RuntimeError` so that code which does not care still sees it.
    """


@runtime_checkable
class Solver(Protocol):
    """Something that can prepare to solve against a matrix repeatedly.

    Satisfied structurally, so a solver need not inherit from anything or import from `sknm`.
    """

    def factorize(self, matrix: sp.csr_array) -> Factorization:
        """Prepare to solve against `matrix`.

        Parameters
        ----------
        matrix : scipy.sparse.csr_array
            The system matrix, square. `CGSolver` additionally requires it to be symmetric
            positive definite, which everything `sknm.assembly` produces is.

        Returns
        -------
        callable
            A function taking a right-hand side of shape ``(n,)`` to the solution of
            ``matrix @ x = rhs``. It may be called any number of times.
        """
        ...


@dataclass(frozen=True)
class DirectSolver:
    """Sparse LU factorization, computed once and reused. The default.

    Exact to roundoff and deterministic, with no tolerance to choose. The factorization is
    denser than the matrix, so on a very large network it is the memory rather than the time
    that eventually makes an iterative solver the better choice.

    Parameters
    ----------
    permc_spec : {"COLAMD", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A"}, optional
        Column permutation passed to SuperLU, which decides how much fill the factorization
        produces. By default ``"COLAMD"``. See `scipy.sparse.linalg.splu`.
    """

    permc_spec: ColumnPermutation = "COLAMD"

    def factorize(self, matrix: sp.csr_array) -> Factorization:
        """Factorize `matrix` with SuperLU.

        Parameters
        ----------
        matrix : scipy.sparse.csr_array
            The system matrix.

        Returns
        -------
        callable
            A function solving ``matrix @ x = rhs`` exactly, by back-substitution.
        """
        # SuperLU wants compressed columns; handing it anything else copies silently.
        factorization = spla.splu(sp.csc_array(matrix), permc_spec=self.permc_spec)
        return lambda rhs: np.asarray(factorization.solve(rhs), dtype=np.float64)


@dataclass(frozen=True)
class CGSolver:
    """Conjugate gradients, which the systems' positive definiteness makes applicable.

    Parameters
    ----------
    rtol : float, optional
        Relative residual to stop at, by default ``1e-6``. This matches the reference
        implementation's own tolerance.
    atol : float, optional
        Absolute residual to stop at, by default ``0.0``, so that `rtol` alone decides.
    maxiter : int, optional
        Iterations to allow before reporting failure, by default ``1000``. Replaces scipy's
        ``10 * n``, which on a large network never stops.
    preconditioner : {"none", "jacobi", "ilu"}, optional
        By default ``"none"``; neither of the others measurably helps on these systems.

    Raises
    ------
    ValueError
        If `preconditioner` names none of the three.
    """

    rtol: float = 1e-6
    atol: float = 0.0
    maxiter: int = 1000
    preconditioner: Preconditioner = "none"

    def factorize(self, matrix: sp.csr_array) -> Factorization:
        """Prepare a conjugate gradient solve against `matrix`.

        Parameters
        ----------
        matrix : scipy.sparse.csr_array
            The system matrix.

        Returns
        -------
        callable
            A function solving ``matrix @ x = rhs`` to `rtol`.

        Raises
        ------
        ConvergenceError
            Raised by the returned function when `maxiter` iterations are not enough.
        """
        return _iterative(spla.cg, "CGSolver", self, matrix)


@dataclass(frozen=True)
class BiCGSTABSolver:
    """Stabilized biconjugate gradients, the reference implementation's own choice.

    Applicable to a nonsymmetric system, which none of these are, so it converges more slowly
    than `CGSolver` here for the same answer. Kept because it is what the reference uses and so
    makes a direct comparison possible.

    Parameters
    ----------
    rtol : float, optional
        Relative residual to stop at, by default ``1e-6``.
    atol : float, optional
        Absolute residual to stop at, by default ``0.0``.
    maxiter : int, optional
        Iterations to allow before reporting failure, by default ``1000``.
    preconditioner : {"none", "jacobi", "ilu"}, optional
        By default ``"none"``.

    Raises
    ------
    ValueError
        If `preconditioner` names none of the three.
    """

    rtol: float = 1e-6
    atol: float = 0.0
    maxiter: int = 1000
    preconditioner: Preconditioner = "none"

    def factorize(self, matrix: sp.csr_array) -> Factorization:
        """Prepare a stabilized biconjugate gradient solve against `matrix`.

        Parameters
        ----------
        matrix : scipy.sparse.csr_array
            The system matrix.

        Returns
        -------
        callable
            A function solving ``matrix @ x = rhs`` to `rtol`.

        Raises
        ------
        ConvergenceError
            Raised by the returned function when `maxiter` iterations are not enough.
        """
        return _iterative(spla.bicgstab, "BiCGSTABSolver", self, matrix)


#: The solvers a string can name, and the object each names.
_BY_NAME: dict[str, Solver] = {
    "direct": DirectSolver(),
    "cg": CGSolver(),
    "bicgstab": BiCGSTABSolver(),
}


def as_solver(solver: Solver | str) -> Solver:
    """Accept a solver or the string naming one.

    Parameters
    ----------
    solver : Solver or str
        A solver object, or one of ``"direct"``, ``"cg"`` and ``"bicgstab"``, each naming the
        corresponding solver with its default settings.

    Returns
    -------
    Solver
        The solver itself, or the object the string names.

    Raises
    ------
    ValueError
        If `solver` is a string naming no solver.
    TypeError
        If `solver` is neither a string nor something with a `factorize` method.

    Examples
    --------
    >>> from sknm.linalg import as_solver
    >>> as_solver("cg")
    CGSolver(rtol=1e-06, atol=0.0, maxiter=1000, preconditioner='none')
    """
    if isinstance(solver, str):
        try:
            return _BY_NAME[solver]
        except KeyError:
            choices = ", ".join(repr(name) for name in _BY_NAME)
            raise ValueError(f"solver must be one of {choices}, got {solver!r}") from None
    if not isinstance(solver, Solver):
        raise TypeError(
            f"solver must have a factorize method or name one of {', '.join(_BY_NAME)}, "
            f"got {type(solver).__name__}"
        )
    return solver


def _iterative(
    method: Callable[..., tuple[npt.NDArray[np.float64], int]],
    name: str,
    solver: CGSolver | BiCGSTABSolver,
    matrix: sp.csr_array,
) -> Factorization:
    """Build the solve function for an iterative solver, shared by both of them.

    The preconditioner is built here rather than inside the returned function, so that an
    incomplete LU is computed once and amortized over every step of a simulation rather than
    recomputed on each one.
    """
    preconditioning = _preconditioner(matrix, solver.preconditioner)

    def solve(rhs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        solution, info = method(
            matrix,
            rhs,
            rtol=solver.rtol,
            atol=solver.atol,
            maxiter=solver.maxiter,
            M=preconditioning,
        )
        if info != 0:
            raise ConvergenceError(_failure(name, solver, matrix, rhs, solution, info))
        return np.asarray(solution, dtype=np.float64)

    return solve


def _failure(
    name: str,
    solver: CGSolver | BiCGSTABSolver,
    matrix: sp.csr_array,
    rhs: npt.NDArray[np.float64],
    solution: npt.NDArray[np.float64],
    info: int,
) -> str:
    """Say how far the solve got, which is what decides whether to relax it or abandon it."""
    scale = float(np.linalg.norm(rhs))
    residual = float(np.linalg.norm(matrix @ solution - rhs))
    reached = residual / scale if scale > 0.0 else residual
    if info < 0:
        return f"{name} broke down after {solver.maxiter} iterations at most (scipy info {info})"
    return (
        f"{name} did not converge: reached a relative residual of {reached:.3g} in "
        f"{solver.maxiter} iterations, short of rtol={solver.rtol:g}. Raise maxiter, relax "
        f"rtol, or use DirectSolver, which needs neither."
    )


def _preconditioner(matrix: sp.csr_array, kind: Preconditioner) -> spla.LinearOperator | None:
    """Build the preconditioner an iterative solver applies, or `None` for no preconditioning.

    Parameters
    ----------
    matrix : scipy.sparse.csr_array
        The system matrix.
    kind : {"none", "jacobi", "ilu"}
        Which preconditioner to build.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator or None
        An approximate inverse of `matrix`, or `None`.

    Raises
    ------
    ValueError
        If `kind` names no preconditioner.
    """
    if kind == "none":
        return None
    if kind == "jacobi":
        diagonal = matrix.diagonal()
        inverse = np.divide(1.0, diagonal, out=np.ones_like(diagonal), where=diagonal != 0.0)
        return spla.LinearOperator(matrix.shape, matvec=lambda x: inverse * x)
    if kind == "ilu":
        incomplete = spla.spilu(sp.csc_array(matrix))
        return spla.LinearOperator(matrix.shape, matvec=incomplete.solve)
    raise ValueError(f"preconditioner must be one of {get_args(Preconditioner)}, got {kind!r}")
