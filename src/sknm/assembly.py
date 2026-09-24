"""The matrix each model solves, assembled from a network, a time step and a variant.

One step of any of the three models is a linear system in the membrane potential, and for KNM
in the extracellular potential as well. This module builds that system and nothing else: no
membrane model, no time loop, no solver. What it produces is an `Operator`, which carries the
matrix together with the bookkeeping needed to build a right-hand side from the previous
membrane potential and to read the potentials back out of a solution vector.

With ``D = diag(Cm * Am / dt)`` the capacitive diagonal, the three systems are

- `Variant.SKNM_UE0`: ``(D + L_i) v = D v_prev``
- `Variant.SKNM`: the same, with ``D`` scaled by ``(1 + lam) / lam``
- `Variant.KNM`: the block system

  ```text
  [ D + L_i      L_i      ] [ v   ]   [ D v_prev ]
  [   L_i     L_i + L_e   ] [ u_e ] = [    0     ]
  ```

where ``L_i`` and ``L_i + L_e`` are the network's two Laplacians.

These are symmetric positive definite, which is what lets a Cholesky-style factorization or a
conjugate gradient solve be used on them. The reference implementation writes the same systems
divided through by the membrane capacitance, which makes them nonsymmetric whenever the cells
differ in area; multiplying that form's first block row by ``Cm * Am / dt`` recovers the form
above, so the two describe the same model and the asymmetry is a spelling.

KNM determines the extracellular potential only up to a constant per connected component, so
one cell per component has its extracellular unknown removed. The row **and** the column are
deleted, which preserves the symmetry; stamping a unit row in its place, as the reference does,
would not. Grounding is a gauge on the extracellular potential and cannot move the membrane
potential.

Nothing here carries a unit except `dt`, which is converted on the way in. Everything else
comes off a `CellNetwork` as bare magnitudes in the package's base units.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from sknm import units
from sknm.network import CellNetwork


class Variant(StrEnum):
    """Which of the three models a system is assembled for.

    One enum rather than two independent switches, which would admit a fourth combination that
    is not a model. Each member's value is the string the member can also be named by.

    Attributes
    ----------
    KNM : str
        The Kirchhoff Network Model. Membrane and extracellular potential solved together.
    SKNM : str
        The Simplified Kirchhoff Network Model. The extracellular potential is eliminated
        through the network's `lam`, leaving the membrane potential as the only unknown.
    SKNM_UE0 : str
        SKNM with the extracellular potential taken as zero rather than eliminated. A third
        model, not an approximation of the other two.
    """

    KNM = "knm"
    SKNM = "sknm"
    SKNM_UE0 = "sknm_ue0"


@dataclass(frozen=True, eq=False)
class Operator:
    """An assembled system: the matrix, and the bookkeeping that goes with it.

    Built by `assemble`, not directly. A step consists of forming a right-hand side with
    `rhs`, solving the system, and reading the potentials back with `membrane_potential` and
    `extracellular_potential` -- so that where the blocks are, and which extracellular unknowns
    were removed by the grounding, is known here and nowhere else.

    Attributes
    ----------
    matrix : scipy.sparse.csr_array
        The system matrix, symmetric and positive definite, of shape ``(n_dofs, n_dofs)``.
        In mS.
    variant : Variant
        The model this system was assembled for.
    n_cells : int
        Number of cells in the network it came from.
    D : numpy.ndarray
        Diagonal of the capacitive matrix ``D``, shape ``(n_cells,)``, read-only. This is
        ``Cm * Am / dt``, times ``(1 + lam) / lam`` for `Variant.SKNM`. In mS.
    ground : numpy.ndarray
        Cells whose extracellular unknown was removed, one per connected component, sorted.
        Empty for the two variants that have no extracellular unknown.
    """

    matrix: sp.csr_array
    variant: Variant
    n_cells: int
    D: npt.NDArray[np.float64]
    ground: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        self.D.flags.writeable = False
        self.ground.flags.writeable = False

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(variant={self.variant.name}, n_cells={self.n_cells}, "
            f"n_dofs={self.n_dofs}, n_ground={self.ground.size})"
        )

    @property
    def n_dofs(self) -> int:
        """int: Number of unknowns in the system, which is the size of `matrix`."""
        return int(self.matrix.shape[0])

    def rhs(self, v_prev: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Build the right-hand side for one step from a previous membrane potential.

        Parameters
        ----------
        v_prev : array_like
            Membrane potential of each cell after the membrane model has been advanced, in mV,
            shape ``(n_cells,)``.

        Returns
        -------
        numpy.ndarray
            Right-hand side of shape ``(n_dofs,)``. Its first `n_cells` entries are
            ``D * v_prev``; the extracellular block, where there is one, is zero.

        Raises
        ------
        ValueError
            If `v_prev` does not hold one value per cell.
        """
        potential = np.asarray(v_prev, dtype=np.float64)
        if potential.shape != (self.n_cells,):
            raise ValueError(
                f"v_prev must hold one value per cell: expected shape ({self.n_cells},), "
                f"got {potential.shape}"
            )
        capacitive = self.D * potential
        if self.variant is not Variant.KNM:
            return capacitive
        return np.concatenate([capacitive, np.zeros(self.n_dofs - self.n_cells)])

    def membrane_potential(self, solution: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Read the membrane potential out of a solution vector.

        Parameters
        ----------
        solution : array_like
            Solution of the system, shape ``(n_dofs,)``.

        Returns
        -------
        numpy.ndarray
            Membrane potential of each cell in mV, shape ``(n_cells,)``.

        Raises
        ------
        ValueError
            If `solution` does not have one entry per degree of freedom.
        """
        return self._as_solution(solution)[: self.n_cells]

    def extracellular_potential(self, solution: npt.ArrayLike) -> npt.NDArray[np.float64] | None:
        """Read the extracellular potential out of a solution vector.

        The grounded cells are put back, at the zero their removal fixed them to. The potential
        is determined only up to a constant per connected component, and that constant is the
        choice of ground.

        Parameters
        ----------
        solution : array_like
            Solution of the system, shape ``(n_dofs,)``.

        Returns
        -------
        numpy.ndarray or None
            Extracellular potential of each cell in mV, shape ``(n_cells,)``; zeros for
            `Variant.SKNM_UE0`, which defines it to be zero. `None` for `Variant.SKNM`, which
            eliminates it rather than solving for it, so no value is available.

        Raises
        ------
        ValueError
            If `solution` does not have one entry per degree of freedom.
        """
        values = self._as_solution(solution)
        if self.variant is Variant.SKNM:
            return None
        potential = np.zeros(self.n_cells)
        if self.variant is Variant.SKNM_UE0:
            return potential
        free = np.setdiff1d(np.arange(self.n_cells, dtype=np.int64), self.ground)
        potential[free] = values[self.n_cells :]
        return potential

    def _as_solution(self, solution: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Check that a vector is the right length to be a solution of this system."""
        values = np.asarray(solution, dtype=np.float64)
        if values.shape != (self.n_dofs,):
            raise ValueError(
                f"solution must have one entry per degree of freedom: expected shape "
                f"({self.n_dofs},), got {values.shape}"
            )
        return values


def assemble(
    network: CellNetwork,
    *,
    dt: Any,
    variant: Variant | str,
    ground: npt.ArrayLike | None = None,
) -> Operator:
    """Assemble the linear system one step of a variant solves.

    Parameters
    ----------
    network : CellNetwork
        The cells, their connections and the conductances between them. The membrane
        capacitance is taken from it.
    dt : pint.Quantity
        Length of a time step, as a time, for example ``0.02 * ms``.
    variant : Variant or str
        Which model to assemble for. A string naming a `Variant` value is accepted.
    ground : array_like or None, optional
        Cells whose extracellular potential is pinned to zero, one per connected component. By
        default `None`, meaning the lowest-index cell of each component. Only `Variant.KNM` has
        an extracellular unknown, so the other two variants ignore this.

    Returns
    -------
    Operator
        The assembled system.

    Raises
    ------
    TypeError
        If `dt` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `dt` does not measure a time.
    ValueError
        If `variant` names no model, if `dt` is not positive and finite, or if `ground` refers
        to a cell that does not exist or does not pin each connected component exactly once.

    Examples
    --------
    >>> from sknm import assemble, chain
    >>> from sknm.units import cm, mS, ms, uS, um
    >>> network = chain(
    ...     4, lx=16 * um, ly=16 * um, lz=19.2 * um, delta_e=0.2,
    ...     sigma_i=4.0 * mS / cm, sigma_e=20.0 * mS / cm, Gg=0.2 * uS,
    ... )
    >>> assemble(network, dt=0.02 * ms, variant="sknm")
    Operator(variant=SKNM, n_cells=4, n_dofs=4, n_ground=0)

    KNM carries an extracellular unknown per cell, one of which the grounding removes:

    >>> assemble(network, dt=0.02 * ms, variant="knm")
    Operator(variant=KNM, n_cells=4, n_dofs=7, n_ground=1)
    """
    model = _as_variant(variant)
    step = float(units.in_base_units(dt, "time", name="dt"))
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError(f"dt must be positive and finite, got {step} ms")

    capacitive = np.asarray(network.Cm * network.membrane_area / step, dtype=np.float64)
    if model is Variant.SKNM:
        # Eliminating the extracellular potential under `L_e = lam * L_i` leaves the
        # intracellular coupling reduced by `lam / (1 + lam)`; dividing the capacitive term by
        # that instead keeps the matrix a plain `D + L_i`.
        capacitive = capacitive * (1.0 + network.lam) / network.lam

    intra = network.laplacian("intra")
    if model is Variant.KNM:
        grounded = _resolve_ground(network, ground)
        matrix = _ground(_knm_blocks(network, capacitive, intra), network.n_cells, grounded)
    else:
        grounded = np.empty(0, dtype=np.int64)
        matrix = sp.csr_array(sp.diags_array(capacitive) + intra)

    return Operator(
        matrix=matrix,
        variant=model,
        n_cells=network.n_cells,
        D=capacitive,
        ground=grounded,
    )


def _as_variant(variant: Variant | str) -> Variant:
    """Accept a `Variant` or the string naming one, and say what the choices are if neither."""
    try:
        return Variant(variant)
    except ValueError:
        choices = ", ".join(repr(member.value) for member in Variant)
        raise ValueError(f"variant must be one of {choices}, got {variant!r}") from None


def _knm_blocks(
    network: CellNetwork, capacitive: npt.NDArray[np.float64], intra: sp.csr_array
) -> sp.csr_array:
    """The 2N block system, before the grounding removes anything from it.

    The lower-right block is the network's ``intra_extra`` Laplacian rather than the sum of two
    separately built ones: weighting by ``Gi + Ge`` in a single pass keeps it exactly symmetric,
    which adding two matrices assembled in different orders does not guarantee.
    """
    intra_extra = network.laplacian("intra_extra")
    membrane = sp.csr_array(sp.diags_array(capacitive) + intra)
    top = sp.hstack([membrane, intra], format="csr")
    bottom = sp.hstack([intra, intra_extra], format="csr")
    return sp.csr_array(sp.vstack([top, bottom], format="csr"))


def _ground(matrix: sp.csr_array, n_cells: int, grounded: npt.NDArray[np.int64]) -> sp.csr_array:
    """Delete the row and the column of each grounded extracellular unknown.

    Deleting both is what keeps the matrix symmetric, and with it the positive definiteness
    that the whole choice of solver rests on. Replacing the row by a unit row would impose the
    same condition and destroy both.
    """
    removed = n_cells + grounded
    keep = np.setdiff1d(np.arange(matrix.shape[0], dtype=np.int64), removed)
    return sp.csr_array(matrix[keep][:, keep])


def _resolve_ground(network: CellNetwork, ground: npt.ArrayLike | None) -> npt.NDArray[np.int64]:
    """Work out which cells have their extracellular unknown removed, and check the choice.

    The extracellular potential of each connected component is determined only up to a
    constant, so each component needs exactly one cell pinned: one fewer leaves the system
    singular, one more overdetermines that component's potential.
    """
    labels = network.component_labels
    n_components = int(labels.max()) + 1 if labels.size else 0

    if ground is None:
        _, lowest = np.unique(labels, return_index=True)
        return np.sort(lowest.astype(np.int64))

    chosen = np.atleast_1d(np.asarray(ground, dtype=np.int64)).reshape(-1)
    outside = chosen[(chosen < 0) | (chosen >= network.n_cells)]
    if outside.size:
        raise ValueError(
            f"ground refers to cell index {int(outside[0])}, but the network has "
            f"{network.n_cells} cells"
        )
    counts = np.bincount(labels[chosen], minlength=n_components)
    if not np.array_equal(counts, np.ones(n_components, dtype=counts.dtype)):
        raise ValueError(
            f"ground must pin exactly one cell per connected component; the network has "
            f"{n_components} of them and this pins {counts.tolist()}"
        )
    return np.sort(chosen)
