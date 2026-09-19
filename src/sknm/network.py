"""The cell network: which cells exist, which are connected, and how well.

A `CellNetwork` is a weighted graph. Cells carry a membrane area and an extracellular volume
fraction; connections carry a length, a cross-sectional area and a gap junction conductance.
Everything else is derived from those, including the two Laplacians the models are built on and
the ratio `lam` by which SKNM summarizes the extracellular space.

Both conductances of a connection mix the volume fractions of *both* of its endpoint cells, so
neither can be computed by a `Connection` on its own; that arithmetic lives here, on the
network, and the connection carries only what is local to it.

Networks are immutable. `frozen=True` stops attributes being rebound but not arrays being
written into, so every array is stamped read-only on construction and the arrays passed in are
copied rather than adopted. Derived quantities are cached, which is only sound because nothing
can change underneath them; to change a conductance, build a new network with
`CellNetwork.with_conductances`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from sknm.units import mm

# Largest cell dimension accepted by the constructors. Cells are tens of micrometres across,
# so anything approaching a millimetre is a dropped unit conversion -- 16 rather than 16 * um.
MAX_CELL_SIZE = 1.0 * mm

LaplacianKind = Literal["intra", "intra_extra"]


@dataclass(frozen=True, eq=False)
class Connection:
    """An adjacency between two cells through which current can pass.

    Carries only what belongs to the connection itself. The conductances that a simulation
    actually uses also depend on the two cells it joins, and are computed by `CellNetwork`.

    Parameters
    ----------
    cells : tuple of int
        Indices of the two connected cells.
    length : float
        Distance between the two cell centres, in cm.
    cross_section : float
        Area of the interface between the two cells, in cm^2.
    Gg : float
        Gap junction conductance, in mS. Zero means the gap junctions are shut, which blocks
        intracellular current without disconnecting the cells extracellularly.

    Raises
    ------
    ValueError
        If the two cell indices are equal or negative, or if `length` or `cross_section` is
        not positive, or if `Gg` is negative.
    """

    cells: tuple[int, int]
    length: float
    cross_section: float
    Gg: float

    def __post_init__(self) -> None:
        first, second = self.cells
        if first == second:
            raise ValueError(f"a cell cannot connect to itself, but cell {first} does")
        if first < 0 or second < 0:
            raise ValueError(f"cell index must be non-negative, got {self.cells}")
        if self.length <= 0:
            raise ValueError(f"length must be positive, got {self.length}")
        if self.cross_section <= 0:
            raise ValueError(f"cross_section must be positive, got {self.cross_section}")
        if self.Gg < 0:
            raise ValueError(f"Gg must be non-negative, got {self.Gg}")


@dataclass(frozen=True, eq=False, kw_only=True)
class CellNetwork:
    """A collection of cells and the connections between them.

    An arbitrary graph, not necessarily a sheet. Prefer the `from_edges`, `chain` and `sheet`
    constructors; this signature is the raw struct-of-arrays form they all build.

    Parameters
    ----------
    membrane_area : numpy.ndarray
        Membrane area of each cell, shape ``(n_cells,)``, in cm^2.
    delta_e : numpy.ndarray
        Extracellular volume fraction of each cell, shape ``(n_cells,)``, strictly between 0
        and 1.
    connections : numpy.ndarray
        Pairs of connected cell indices, shape ``(n_connections, 2)``.
    length : numpy.ndarray
        Length of each connection, shape ``(n_connections,)``, in cm.
    cross_section : numpy.ndarray
        Cross-sectional area of each connection, shape ``(n_connections,)``, in cm^2.
    Gg : numpy.ndarray
        Gap junction conductance of each connection, shape ``(n_connections,)``, in mS.
    sigma_i : float
        Intracellular conductivity, in mS/cm.
    sigma_e : float
        Extracellular conductivity, in mS/cm.
    Cm : float, optional
        Specific membrane capacitance in uF/cm^2, by default 1.0.
    lam_override : float or None, optional
        Value to report as `lam` instead of the one derived from the conductances. By default
        `None`, meaning derive it.

    Raises
    ------
    ValueError
        If the arrays disagree on the number of cells or connections, if a connection refers
        to a cell that does not exist or joins a cell to itself, if a volume fraction is not
        strictly between 0 and 1, if a conductivity, capacitance, length, cross-section or
        membrane area is not positive, if a gap junction conductance is negative, or if any
        dimension exceeds `MAX_CELL_SIZE`.
    """

    membrane_area: npt.NDArray[np.float64]
    delta_e: npt.NDArray[np.float64]
    connections: npt.NDArray[np.int64]
    length: npt.NDArray[np.float64]
    cross_section: npt.NDArray[np.float64]
    Gg: npt.NDArray[np.float64]
    sigma_i: float
    sigma_e: float
    Cm: float = 1.0
    lam_override: float | None = None

    def __post_init__(self) -> None:
        connections = np.array(self.connections, dtype=np.int64)
        if connections.size == 0:
            connections = connections.reshape((0, 2))
        if connections.ndim != 2 or connections.shape[1] != 2:
            raise ValueError(
                f"connections must have shape (n_connections, 2), got {connections.shape}"
            )
        object.__setattr__(self, "connections", connections)
        for name in ("membrane_area", "delta_e", "length", "cross_section", "Gg"):
            # Copy rather than adopt: stamping a caller's array read-only would reach out of
            # this object and change something it does not own.
            value = np.array(getattr(self, name), dtype=np.float64)
            if value.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional, got shape {value.shape}")
            object.__setattr__(self, name, value)

        self._validate()

        for name in ("connections", "membrane_area", "delta_e", "length", "cross_section", "Gg"):
            getattr(self, name).flags.writeable = False

    def _validate(self) -> None:
        n_cells = self.membrane_area.size
        n_connections = self.connections.shape[0]
        for name, unit, expected in [
            ("delta_e", "cell", n_cells),
            ("length", "connection", n_connections),
            ("cross_section", "connection", n_connections),
            ("Gg", "connection", n_connections),
        ]:
            got = getattr(self, name).size
            if got != expected:
                raise ValueError(
                    f"{name} must have one value per {unit}: expected {expected}, got {got}"
                )

        for name in ("membrane_area", "delta_e", "length", "cross_section", "Gg"):
            # Checked up front because an infinity survives the comparisons below -- an
            # infinite Gg passes "non-negative" and then turns Gi into a silent nan.
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name} must be finite")

        if n_connections and (self.connections < 0).any():
            raise ValueError("cell index must be non-negative")
        if n_connections and self.connections.max() >= n_cells:
            worst = int(self.connections.max())
            raise ValueError(
                f"a connection refers to cell index {worst}, but the network has {n_cells} cells"
            )
        if n_connections and (self.connections[:, 0] == self.connections[:, 1]).any():
            raise ValueError("a cell cannot connect to itself")

        if not ((self.delta_e > 0) & (self.delta_e < 1)).all():
            raise ValueError("delta_e must lie strictly between 0 and 1")
        if not (self.membrane_area > 0).all():
            raise ValueError("membrane_area must be positive")
        if not (self.length > 0).all():
            raise ValueError("length must be positive")
        if not (self.cross_section > 0).all():
            raise ValueError("cross_section must be positive")
        if not (self.Gg >= 0).all():
            raise ValueError("Gg must be non-negative")

        for name, value in [("sigma_i", self.sigma_i), ("sigma_e", self.sigma_e), ("Cm", self.Cm)]:
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")

        # A dropped unit conversion is the failure mode worth catching here: it produces a
        # network that solves perfectly well and describes nothing biological.
        if (self.length > MAX_CELL_SIZE).any():
            raise ValueError(
                f"implausible connection length {self.length.max():g} cm, above the "
                f"{MAX_CELL_SIZE:g} cm limit; lengths are in cm, so a cell is 16 * um"
            )
        if (self.cross_section > MAX_CELL_SIZE**2).any():
            raise ValueError(
                f"implausible connection cross_section {self.cross_section.max():g} cm^2, "
                f"above the {MAX_CELL_SIZE**2:g} cm^2 limit"
            )
        if (self.membrane_area > 6 * MAX_CELL_SIZE**2).any():
            raise ValueError(
                f"implausible membrane_area {self.membrane_area.max():g} cm^2, above the "
                f"{6 * MAX_CELL_SIZE**2:g} cm^2 limit"
            )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(n_cells={self.n_cells}, "
            f"n_connections={self.n_connections}, sigma_i={self.sigma_i:g}, "
            f"sigma_e={self.sigma_e:g}, Cm={self.Cm:g})"
        )

    @property
    def n_cells(self) -> int:
        """int: Number of cells."""
        return int(self.membrane_area.size)

    @property
    def n_connections(self) -> int:
        """int: Number of connections."""
        return int(self.connections.shape[0])

    @cached_property
    def Gi(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: Intracellular conductance of each connection, in mS.

        The cytosol along the connection in series with the gap junctions at its interface,
        with the intracellular volume fraction averaged over both endpoint cells. Written as
        ``Gg / (Gg * R + 1)`` rather than the equivalent ``1 / (R + 1 / Gg)`` so that a shut
        gap junction, ``Gg = 0``, gives exactly zero instead of dividing by zero.
        """
        first, second = self.connections[:, 0], self.connections[:, 1]
        delta_i = 1.0 - self.delta_e
        mean_delta_i = 0.5 * (delta_i[first] + delta_i[second])
        cytosol = self.length / (mean_delta_i * self.cross_section * self.sigma_i)
        return np.asarray(self.Gg / (self.Gg * cytosol + 1.0), dtype=np.float64)

    @cached_property
    def Ge(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: Extracellular conductance of each connection, in mS.

        The extracellular volume fraction averaged over both endpoint cells, times the
        conductance of that share of the connection's cross-section.
        """
        first, second = self.connections[:, 0], self.connections[:, 1]
        mean_delta_e = 0.5 * (self.delta_e[first] + self.delta_e[second])
        conductance = mean_delta_e * self.cross_section * self.sigma_e / self.length
        return np.asarray(conductance, dtype=np.float64)

    @cached_property
    def incidence(self) -> sp.csr_array:
        """scipy.sparse.csr_array: Incidence matrix, shape ``(n_connections, n_cells)``.

        Row `c` of a connection from cell `j` to cell `k` holds ``+1`` in column `j` and
        ``-1`` in column `k`. Both Laplacians are ``A.T @ diag(G) @ A`` over this one matrix
        and differ only in `G`, which is what makes them share a sparsity pattern.
        """
        rows = np.repeat(np.arange(self.n_connections, dtype=np.int64), 2)
        columns = self.connections.reshape(-1)
        data = np.tile(np.array([1.0, -1.0]), self.n_connections)
        shape = (self.n_connections, self.n_cells)
        return sp.coo_array((data, (rows, columns)), shape=shape).tocsr()

    def laplacian(self, kind: LaplacianKind) -> sp.csr_array:
        """Build a weighted graph Laplacian over the connections.

        Assembled as ``A.T @ diag(G) @ A``, which reproduces the reference implementation's
        hand-indexed stencil: a diagonal holding the sum of a cell's incident conductances and
        an off-diagonal holding the negated conductance of each connection.

        Parameters
        ----------
        kind : {"intra", "intra_extra"}
            Which conductances to weight by: `Gi` alone, or `Gi + Ge`.

        Returns
        -------
        scipy.sparse.csr_array
            Symmetric positive semidefinite matrix of shape ``(n_cells, n_cells)`` with zero
            row sums.

        Raises
        ------
        ValueError
            If `kind` is not one of the two accepted values.
        """
        if kind == "intra":
            weights = self.Gi
        elif kind == "intra_extra":
            weights = self.Gi + self.Ge
        else:
            raise ValueError(f"kind must be one of {get_args(LaplacianKind)}, got {kind!r}")
        incidence = self.incidence
        return sp.csr_array(incidence.T @ sp.diags_array(weights) @ incidence)

    @cached_property
    def lam(self) -> float:
        """float: The ratio by which SKNM summarizes the extracellular space.

        A least-squares fit of ``Ge ~ lam * Gi`` over every connection, weighted by the square
        of the connection's shape factor ``length / cross_section`` (paper eq. 30). Uniform
        conductances make the fit exact, so `lam` is then simply ``Ge / Gi``.

        Returns the constructor's `lam_override` instead, when one was given.

        Raises
        ------
        ValueError
            If no connection conducts intracellularly, which leaves the fit undetermined.
        """
        if self.lam_override is not None:
            return float(self.lam_override)
        shape_factor = (self.length / self.cross_section) ** 2
        denominator = float(np.sum(self.Gi**2 * shape_factor))
        if denominator == 0.0:
            raise ValueError(
                "lam is undefined for a network with no intracellular coupling; pass "
                "lam_override to set it explicitly"
            )
        return float(np.sum(self.Ge * self.Gi * shape_factor) / denominator)

    @property
    def conductance_ratio(self) -> float:
        """float: Alias for `lam`, spelled out."""
        return self.lam

    @cached_property
    def component_labels(self) -> npt.NDArray[np.int64]:
        """numpy.ndarray: Connected-component label of each cell, shape ``(n_cells,)``.

        Purely topological: a connection whose gap junctions are shut still joins its two
        cells, because it still passes extracellular current.
        """
        adjacency = sp.coo_array(
            (
                np.ones(self.n_connections),
                (self.connections[:, 0], self.connections[:, 1]),
            ),
            shape=(self.n_cells, self.n_cells),
        ).tocsr()
        _, labels = connected_components(adjacency, directed=False)
        return np.asarray(labels, dtype=np.int64)

    def with_conductances(self, Gg: npt.ArrayLike) -> CellNetwork:
        """Build a copy of this network with different gap junction conductances.

        The replacement for mutating the conductances in place, which the cached derived
        quantities rule out. A sweep over gap junction variation is then a sequence of
        networks built from one conductance array each, rather than one network reconfigured
        between runs.

        Everything else carries over unchanged, `lam_override` included; drop the override by
        building the network afresh.

        Parameters
        ----------
        Gg : array_like
            New gap junction conductance of each connection, shape ``(n_connections,)``.

        Returns
        -------
        CellNetwork
            A new network. This one is left untouched.

        Raises
        ------
        ValueError
            If `Gg` does not have one value per connection, or holds a negative conductance.
        """
        return dataclasses.replace(self, Gg=np.asarray(Gg, dtype=np.float64))


def _spread(value: npt.ArrayLike, size: int) -> npt.NDArray[np.float64]:
    """Broadcast a scalar over `size` entries, leaving an array alone.

    An array of the wrong length is passed through untouched so that `CellNetwork` reports the
    mismatch against the field it belongs to, rather than raising a shape error from here.
    """
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        return np.full(size, float(array))
    return array


def _check_dimension(name: str, value: float) -> None:
    """Reject a cell dimension that is not positive or is too large to be a cell."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    if value > MAX_CELL_SIZE:
        raise ValueError(
            f"implausible cell dimension {name} = {value:g} cm, above the {MAX_CELL_SIZE:g} cm "
            f"limit; dimensions are in cm, so a 16 um cell is 16 * um"
        )


def _box_surface_area(lx: float, ly: float, lz: float) -> float:
    """Surface area of a cuboid cell, the default when no membrane area is given."""
    return 2 * (lx * ly + lx * lz + ly * lz)


def from_edges(
    connections: Sequence[Connection],
    *,
    membrane_area: npt.ArrayLike,
    delta_e: npt.ArrayLike,
    sigma_i: float,
    sigma_e: float,
    Cm: float = 1.0,
    n_cells: int | None = None,
    lam_override: float | None = None,
) -> CellNetwork:
    """Build a network from an explicit list of connections.

    The general constructor: any topology, per-cell geometry, per-connection conductance. The
    paper's own geometries are not built here.

    Parameters
    ----------
    connections : sequence of Connection
        The connections. May be empty, in which case `n_cells` is required.
    membrane_area : array_like
        Membrane area per cell in cm^2, as a scalar or one value per cell.
    delta_e : array_like
        Extracellular volume fraction per cell, as a scalar or one value per cell.
    sigma_i : float
        Intracellular conductivity, in mS/cm.
    sigma_e : float
        Extracellular conductivity, in mS/cm.
    Cm : float, optional
        Specific membrane capacitance in uF/cm^2, by default 1.0.
    n_cells : int or None, optional
        Number of cells. By default `None`, meaning take it from the length of whichever of
        `membrane_area` and `delta_e` is an array, or failing that from the highest cell index
        a connection refers to. Give it explicitly to include cells no connection reaches.
    lam_override : float or None, optional
        Value to report as `lam` instead of deriving it, by default `None`.

    Returns
    -------
    CellNetwork
        The network.

    Raises
    ------
    ValueError
        If `n_cells` cannot be determined, or if the geometry is invalid; see `CellNetwork`.

    Examples
    --------
    >>> from sknm import Connection, from_edges
    >>> from sknm.units import um
    >>> network = from_edges(
    ...     [Connection((0, 1), length=16 * um, cross_section=16 * um * 19.2 * um, Gg=2e-4)],
    ...     membrane_area=1.8e-5,
    ...     delta_e=0.2,
    ...     sigma_i=4.0,
    ...     sigma_e=20.0,
    ... )
    >>> network.n_cells, network.n_connections
    (2, 1)
    """
    if n_cells is None:
        n_cells = _infer_n_cells(connections, membrane_area, delta_e)
    pairs = np.array([connection.cells for connection in connections], dtype=np.int64)
    return CellNetwork(
        membrane_area=_spread(membrane_area, n_cells),
        delta_e=_spread(delta_e, n_cells),
        connections=pairs.reshape((-1, 2)),
        length=np.array([connection.length for connection in connections], dtype=np.float64),
        cross_section=np.array(
            [connection.cross_section for connection in connections], dtype=np.float64
        ),
        Gg=np.array([connection.Gg for connection in connections], dtype=np.float64),
        sigma_i=sigma_i,
        sigma_e=sigma_e,
        Cm=Cm,
        lam_override=lam_override,
    )


def _infer_n_cells(
    connections: Sequence[Connection], membrane_area: npt.ArrayLike, delta_e: npt.ArrayLike
) -> int:
    """Work out how many cells a `from_edges` call describes."""
    for value in (membrane_area, delta_e):
        array = np.asarray(value)
        if array.ndim == 1:
            return int(array.size)
    if connections:
        return max(max(connection.cells) for connection in connections) + 1
    raise ValueError(
        "n_cells cannot be inferred from scalar geometry and no connections; pass it explicitly"
    )


def chain(
    n_cells: int,
    *,
    lx: float,
    ly: float,
    lz: float,
    delta_e: npt.ArrayLike,
    sigma_i: float,
    sigma_e: float,
    Gg: npt.ArrayLike,
    membrane_area: npt.ArrayLike | None = None,
    Cm: float = 1.0,
    lam_override: float | None = None,
) -> CellNetwork:
    """Build a one-dimensional strand of cells, each connected to the next.

    Parameters
    ----------
    n_cells : int
        Number of cells, at least one.
    lx, ly, lz : float
        Cell dimensions in cm. Connections run along x, so they have length `lx` and
        cross-section ``ly * lz``.
    delta_e : array_like
        Extracellular volume fraction per cell, as a scalar or one value per cell.
    sigma_i : float
        Intracellular conductivity, in mS/cm.
    sigma_e : float
        Extracellular conductivity, in mS/cm.
    Gg : array_like
        Gap junction conductance in mS, as a scalar or one value per connection.
    membrane_area : array_like or None, optional
        Membrane area per cell in cm^2. By default `None`, meaning the surface area of a
        cuboid cell, ``2 * (lx*ly + lx*lz + ly*lz)``.
    Cm : float, optional
        Specific membrane capacitance in uF/cm^2, by default 1.0.
    lam_override : float or None, optional
        Value to report as `lam` instead of deriving it, by default `None`.

    Returns
    -------
    CellNetwork
        The network, with connection `i` joining cells `i` and ``i + 1``.

    Raises
    ------
    ValueError
        If `n_cells` is less than one, or if the geometry is invalid; see `CellNetwork`.
    """
    if n_cells < 1:
        raise ValueError(f"n_cells must be at least 1, got {n_cells}")
    for name, value in [("lx", lx), ("ly", ly), ("lz", lz)]:
        _check_dimension(name, value)
    n_connections = n_cells - 1
    index = np.arange(n_connections, dtype=np.int64)
    return CellNetwork(
        membrane_area=_spread(
            _box_surface_area(lx, ly, lz) if membrane_area is None else membrane_area, n_cells
        ),
        delta_e=_spread(delta_e, n_cells),
        connections=np.column_stack([index, index + 1]),
        length=np.full(n_connections, lx),
        cross_section=np.full(n_connections, ly * lz),
        Gg=_spread(Gg, n_connections),
        sigma_i=sigma_i,
        sigma_e=sigma_e,
        Cm=Cm,
        lam_override=lam_override,
    )


def sheet(
    nx: int,
    ny: int,
    *,
    lx: float,
    ly: float,
    lz: float,
    delta_e: npt.ArrayLike,
    sigma_i: float,
    sigma_e: float,
    Gg: npt.ArrayLike,
    membrane_area: npt.ArrayLike | None = None,
    Cm: float = 1.0,
    lam_override: float | None = None,
) -> CellNetwork:
    """Build a rectangular sheet of cells, each connected to its four neighbours.

    Cells are numbered row-major, ``index = j * nx + i``, and connections are ordered
    x-direction first and then y-direction, both row-major. That is the reference
    implementation's own layout, and it is what an array-valued `Gg` is indexed by.

    Parameters
    ----------
    nx, ny : int
        Number of cells along x and along y, each at least one.
    lx, ly, lz : float
        Cell dimensions in cm. An x-direction connection has length `lx` and cross-section
        ``ly * lz``; a y-direction connection has length `ly` and cross-section ``lx * lz``.
    delta_e : array_like
        Extracellular volume fraction per cell, as a scalar or one value per cell.
    sigma_i : float
        Intracellular conductivity, in mS/cm.
    sigma_e : float
        Extracellular conductivity, in mS/cm.
    Gg : array_like
        Gap junction conductance in mS, as a scalar or one value per connection.
    membrane_area : array_like or None, optional
        Membrane area per cell in cm^2. By default `None`, meaning the surface area of a
        cuboid cell, ``2 * (lx*ly + lx*lz + ly*lz)``.
    Cm : float, optional
        Specific membrane capacitance in uF/cm^2, by default 1.0.
    lam_override : float or None, optional
        Value to report as `lam` instead of deriving it, by default `None`.

    Returns
    -------
    CellNetwork
        The network.

    Raises
    ------
    ValueError
        If `nx` or `ny` is less than one, or if the geometry is invalid; see `CellNetwork`.

    Examples
    --------
    >>> from sknm import sheet
    >>> from sknm.units import um
    >>> network = sheet(
    ...     40, 40, lx=16 * um, ly=16 * um, lz=19.2 * um,
    ...     delta_e=0.2, sigma_i=4.0, sigma_e=20.0, Gg=1 / 5e3,
    ... )
    >>> network.n_connections
    3120
    >>> round(network.lam, 2)
    39.65
    """
    if nx < 1 or ny < 1:
        raise ValueError(f"nx and ny must be at least 1, got {(nx, ny)}")
    for name, value in [("lx", lx), ("ly", ly), ("lz", lz)]:
        _check_dimension(name, value)

    index = np.arange(nx * ny, dtype=np.int64).reshape((ny, nx))
    along_x = np.column_stack([index[:, :-1].reshape(-1), index[:, 1:].reshape(-1)])
    along_y = np.column_stack([index[:-1, :].reshape(-1), index[1:, :].reshape(-1)])
    n_x, n_y = along_x.shape[0], along_y.shape[0]

    return CellNetwork(
        membrane_area=_spread(
            _box_surface_area(lx, ly, lz) if membrane_area is None else membrane_area, nx * ny
        ),
        delta_e=_spread(delta_e, nx * ny),
        connections=np.concatenate([along_x, along_y]).reshape((-1, 2)),
        length=np.concatenate([np.full(n_x, lx), np.full(n_y, ly)]),
        cross_section=np.concatenate([np.full(n_x, ly * lz), np.full(n_y, lx * lz)]),
        Gg=_spread(Gg, n_x + n_y),
        sigma_i=sigma_i,
        sigma_e=sigma_e,
        Cm=Cm,
        lam_override=lam_override,
    )
