"""The bidomain and monodomain counterparts of the network models, for Figures 6 to 9.

The paper's second half solves the same tissue with the two homogenized continuum models, on
a mesh finer than a cell rather than on a network of cells. That is a genuinely different
discretization -- the reference meshes a 16 um cell with 10 um elements -- so agreement
between it and `sknm` is evidence about the models rather than a restatement of one scheme in
the notation of another.

**This module needs `dolfinx` and `fenicsx-beat`, which `sknm` does not depend on.** The
package's runtime set is numpy, scipy and pint, and nothing here changes that: the scripts
that import this one check `available()` and return with a message when the two are missing.
Install them with, for example, a `dolfinx` container image plus ``pip install fenicsx-beat``.

What does *not* need either is the conductivity field itself, which is a per-cell quantity
computed from the cell geometry and the gap junction resistances. It is plain numpy, it is
what the conductance ratio and Figure 9's left panel are built from, and it lives above the
import guard so that both can be had -- and tested -- without a mesh.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

import common
from sknm import presets, units
from sknm.membrane import base_model_IM, from_gotranx
from sknm.units import cm, kohm, mS

#: What a caller has to install for anything below the import guard to work, and the message
#: the scripts print when it is missing.
REQUIREMENT = (
    "the continuum models need dolfinx and fenicsx-beat >= 0.6, which sknm does not depend "
    "on: run this in a dolfinx container with `pip install fenicsx-beat`"
)

try:
    import basix.ufl
    import beat
    import dolfinx
    import dolfinx.fem.petsc
    import dolfinx.geometry
    from mpi4py import MPI

    # The attribute, not just the import: fenicsx-beat below 0.6 installs cleanly and has no
    # bidomain model, and an AttributeError from inside a run says far less than this does.
    _AVAILABLE = hasattr(beat, "BidomainModel")
except ImportError:
    _AVAILABLE = False


def available() -> bool:
    """Whether the continuum models can be solved here.

    Returns
    -------
    bool
        True when `dolfinx` and a `fenicsx-beat` carrying `BidomainModel` are both importable.
        The figure scripts print `REQUIREMENT` and return when it is False, rather than
        exiting non-zero: a machine without `dolfinx` is behaving correctly, and the example
        job that runs every script must not fail on it.
    """
    return _AVAILABLE


def require() -> None:
    """Stop with a message when the continuum machinery is missing.

    The figure scripts are linear, so there is no function body to return early from;
    declining to run means leaving the interpreter.

    Raises
    ------
    SystemExit
        With code zero, when `available` is false. Zero because a machine without `dolfinx`
        is not a broken machine, and because a non-zero exit would fail any job running the
        script rather than letting it report what is absent.
    """
    if not available():
        print(REQUIREMENT)
        raise SystemExit(0)


#: Gap junction resistance between two cells with no variation applied, in kOhm. The
#: reference's `Rg_x`, and the resistance `sknm.presets.GAP_JUNCTION_CONDUCTANCE` inverts.
GAP_JUNCTION_RESISTANCE = float((1.0 / presets.GAP_JUNCTION_CONDUCTANCE).m_as(kohm))

#: Conductivities and the membrane area of one cell, in the cardiac CGS base set the package
#: stores: mS/cm and cm^2.
SIGMA_I = float(presets.SIGMA_I.m_as(mS / cm))
SIGMA_E = float(presets.SIGMA_E.m_as(mS / cm))
MEMBRANE_AREA = float(presets.MEMBRANE_AREA.m_as(cm**2))
CM = float(presets.CM.m_as(units.uF / cm**2))

#: The stimulated strip, in cells, as the reference writes it: ``x < 2*lx`` and
#: ``14*ly < y < 25*ly``. Two columns wide and eleven rows tall, centred on the row the
#: conduction velocity is measured along.
STIMULUS_COLUMNS = 2
STIMULUS_ROWS = (14, 25)

#: The two cell columns a conduction velocity is measured between, probed at their centres
#: and halfway up the sheet, and the membrane potential the wave counts as having reached them
#: at. The paper's protocol, shared by all four models.
CONDUCTION_COLUMNS = (9, 34)

#: The longest ODE step taken, in ms. The membrane model is substepped whenever the time step
#: is longer than this, and stepped once with the time step itself when it is shorter. The
#: reference fixes this at 0.01 and computes the number of substeps as ``round(dt/dt_ode)``,
#: which is **zero** at every time step below 0.005 -- the membrane model is then never
#: advanced at all and the resting state overwrites the solve every step. Taking the smaller
#: of the two, and at least one substep, agrees with the reference wherever its own arithmetic
#: gives a positive count.
MAX_ODE_STEP = 0.01

#: How close to a boundary a node has to be to count as on it, as a fraction of the smaller
#: cell dimension. Comparisons against the edge of a region are made with this tolerance
#: because a structured mesh puts nodes exactly on those edges, and a strict comparison then
#: turns on the last bit of a floating point coordinate.
_ON_THE_EDGE = 1e-6


@dataclasses.dataclass(frozen=True)
class BidomainSetup:
    """One point of a sweep: everything that decides what a continuum simulation computes.

    The sibling of `common.Setup`, which describes the same tissue as a network of cells.
    Every field is part of the cache key, so adding one here invalidates cached results rather
    than silently serving numbers computed without it.

    Parameters
    ----------
    alpha : float, optional
        Anisotropy factor, by default 1.0.
    delta_e : float, optional
        Extracellular volume fraction, by default the paper's 0.2.
    gamma : float, optional
        Gap junction variation, by default 0.0.
    nx, ny : int, optional
        Number of cells along each axis, by default the paper's 40 by 40.
    dt : float, optional
        Time step in ms, by default the reference's 0.01.
    t_end : float, optional
        How long to run for, in ms, by default 50.
    threshold : float, optional
        Membrane potential at which the wave counts as having arrived, in mV, by default -20.
    element_size : float, optional
        Target element size in cm, by default 8e-4. The mesh divides each cell into a whole
        number of elements, so the size actually used is this one rounded to the nearest
        divisor of the cell; see `elements_per_cell`.
    """

    alpha: float = 1.0
    delta_e: float = presets.DELTA_E
    gamma: float = 0.0
    nx: int = 40
    ny: int = 40
    dt: float = 0.01
    t_end: float = common.T_END
    threshold: float = common.THRESHOLD
    element_size: float = 8e-4

    @property
    def n_cells(self) -> int:
        """int: How many cells the sheet holds."""
        return self.nx * self.ny

    def cell_size(self) -> tuple[float, float, float]:
        """Length, width and depth of one cell, in cm.

        Returns
        -------
        tuple of float
            ``(lx, ly, lz)``. The first two are `sknm.presets.hipsc_cell_size`; the third is
            the cardiac rule ``lz = (1 + delta_e) * ly``, which is what makes the cell's
            extracellular sheath carry the volume fraction.
        """
        lx, ly = presets.hipsc_cell_size(self.alpha)
        return float(lx.m_as(cm)), float(ly.m_as(cm)), (1.0 + self.delta_e) * float(ly.m_as(cm))

    def gap_junction_resistance(self) -> npt.NDArray[np.float64]:
        """The gap junction resistance of each cell, in kOhm.

        The variation is the reference's: a draw ``a`` in [0, 1] scales the resistance by
        ``1 / (a(1 - gamma) + (1 - a)(1 + gamma))``. At ``gamma = 0`` the two terms sum to
        one whatever the draw is, so every cell keeps the unvaried resistance.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_cells,)``, in the row-major cell order the reference indexes by.

        Raises
        ------
        ValueError
            If the sheet is not the 40 by 40 one the committed draws describe.
        """
        draws = gap_junction_draws()
        if draws.shape != (self.n_cells,):
            raise ValueError(
                f"the committed draws are for a 40 by 40 sheet with {draws.size} cells, but "
                f"this one is {self.nx} by {self.ny} with {self.n_cells}"
            )
        scale = draws * (1.0 - self.gamma) + (1.0 - draws) * (1.0 + self.gamma)
        return GAP_JUNCTION_RESISTANCE / scale

    def intracellular_conductivity(
        self,
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """The two diagonal components of the intracellular conductivity, per cell, in mS/cm.

        A cell's gap junction resistance is in series with its cytosol, which is what puts the
        resistance in the denominator: a cell that conducts perfectly well internally still
        passes current no faster than its junctions allow. The geometric factor is the one
        that makes the tensor anisotropic when the cell is.

        Returns
        -------
        tuple of numpy.ndarray
            ``(M_i_x, M_i_y)``, each of shape ``(n_cells,)``.
        """
        lx, ly, lz = self.cell_size()
        delta_i = 1.0 - self.delta_e
        resistance = self.gap_junction_resistance()
        conductivity = delta_i * SIGMA_I

        def component(geometry: float) -> npt.NDArray[np.float64]:
            return conductivity / (1.0 + SIGMA_I * resistance * delta_i * geometry)

        return component(ly * lz / lx), component(lx * lz / ly)

    def elements_per_cell(self) -> tuple[int, int]:
        """How many elements each cell is divided into, along each axis.

        Whole numbers, so that no element straddles a cell boundary and the piecewise
        constant conductivity field is represented exactly. That matters wherever `M_i`
        genuinely varies from cell to cell, which is at any positive gap junction variation.

        Taken per axis rather than once for both: a cell at alpha = 4 is 40 by 10 um, and one
        count for both axes would either stretch the elements to the cell's aspect ratio or
        refine far past the target in the short direction.

        Returns
        -------
        tuple of int
            Elements along x and along y, each at least one.
        """
        lx, ly, _ = self.cell_size()
        return (
            max(1, round(lx / self.element_size)),
            max(1, round(ly / self.element_size)),
        )

    def elements(self) -> tuple[int, int]:
        """How many elements the whole sheet is divided into, along each axis.

        Returns
        -------
        tuple of int
            Along x and along y.
        """
        per_x, per_y = self.elements_per_cell()
        return self.nx * per_x, self.ny * per_y

    def domain_size(self) -> tuple[float, float]:
        """Width and height of the sheet, in cm.

        Returns
        -------
        tuple of float
            Along x and along y.
        """
        lx, ly, _ = self.cell_size()
        return self.nx * lx, self.ny * ly

    def extracellular_conductivity(self) -> float:
        """The extracellular conductivity, in mS/cm.

        Isotropic and the same in every cell: the extracellular space has no gap junctions,
        so nothing about a cell's coupling reaches it.

        Returns
        -------
        float
            ``delta_e * sigma_e``.
        """
        return self.delta_e * SIGMA_E

    def conductance_ratio(self) -> float:
        """The single ratio the monodomain model replaces the two conductivities with.

        The value of ``lambda`` that minimizes `conductivity_misfit`, which is the paper's
        equation (32). Setting the derivative of the misfit to zero and solving gives a
        quotient of two area-weighted sums, and the cell areas cancel out of it -- they do
        not cancel out of the misfit itself.

        Returns
        -------
        float
            Dimensionless.
        """
        Mi_x, Mi_y = self.intracellular_conductivity()
        M_e = self.extracellular_conductivity()
        return float(M_e * (Mi_x + Mi_y).sum() / (Mi_x**2 + Mi_y**2).sum())

    def conductivity_misfit(self, ratio: float | None = None) -> float:
        """How far the two conductivities are from being related by one ratio, in mS^2.

        The paper's equation (31): the area integral of the squared difference between the
        extracellular conductivity and `ratio` times the intracellular one, component by
        component. It is zero exactly when a single number relates them everywhere, which is
        the assumption the monodomain model is derived from, and it grows as the gap junction
        conductances spread or the cell becomes anisotropic.

        Each cell contributes its own area, and `M_i` is constant within a cell and `M_e`
        constant everywhere, so the integral is an exact finite sum and needs no mesh.

        Parameters
        ----------
        ratio : float, optional
            The ratio to measure against, by default the minimizing one from
            `conductance_ratio`.

        Returns
        -------
        float
            In mS^2: a conductivity squared times an area.
        """
        if ratio is None:
            ratio = self.conductance_ratio()
        lx, ly, _ = self.cell_size()
        Mi_x, Mi_y = self.intracellular_conductivity()
        M_e = self.extracellular_conductivity()
        residual = (M_e - ratio * Mi_x) ** 2 + (M_e - ratio * Mi_y) ** 2
        return float(0.5 * lx * ly * residual.sum())

    def surface_to_volume_ratio(self) -> float:
        """The membrane area per unit tissue volume, in 1/cm.

        The reference's ``chi``: one cell's membrane area over the volume of the box it
        occupies, extracellular sheath included. It is what turns a per-cell conductance into
        a conductivity, and `fenicsx-beat` expects the conductivity with it already divided
        out, because the capacitance and the time step are its own.

        Returns
        -------
        float
            ``Am / (lx*ly*lz)``.
        """
        lx, ly, lz = self.cell_size()
        return MEMBRANE_AREA / (lx * ly * lz)

    def label(self, **extra: Any) -> str:
        """A cache key naming every parameter that decides the result.

        Parameters
        ----------
        **extra
            Anything beyond the setup that the result depends on, such as which model.

        Returns
        -------
        str
            Every field and every extra, sorted by name.
        """
        return common.cache_label(self, **extra)


def gap_junction_draws() -> npt.NDArray[np.float64]:
    """The reference's own gap junction draws for its 40 by 40 sheet, one per cell.

    Committed under `data/`, for the same reason the beta cell draws are: the published
    figures are drawn with these numbers and any other set moves the curves.

    There are 1600 of them, one per **cell**, where the network model's 1560 are one per
    **connection**. The two describe the same tissue and are not interchangeable.

    Only the x file is committed. The reference loads a y file as well and then reads it
    nowhere -- every one of its conductivity functions, the yy components included, indexes
    the x draws -- so shipping the y file would suggest a second input that decides nothing.

    Returns
    -------
    numpy.ndarray
        Shape ``(1600,)``, in the row-major cell order the reference indexes by.
    """
    return np.loadtxt(common.DATA_DIR / "gj_scale_x_BD.txt")


def stimulus_weight(setup: BidomainSetup, points: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """How much of the stimulus each of `points` receives: one inside the strip, zero outside.

    A node **on** the strip's own edge gets a half, and that is the whole subtlety here.

    The reference tests its strip with strict inequalities, which is unambiguous on its own
    unstructured mesh because no node lands exactly on the edge. On a mesh whose elements
    divide the cell, nodes land there exactly, and neither answer is right: each node carries
    a share of the membrane equal to the area its basis function covers, so a node on the edge
    has half of that share inside the strip. Counting it whole injects 17.6% more stimulus
    current than the strip holds at the size these figures are drawn at, which launches the
    wave 5.7 ms early and steepens the upstroke by 2.2%; dropping it takes out a whole row and
    a whole column, which leaves too little to propagate at all -- measured on four such
    meshes the sheet depolarized a little and nothing travelled.

    Weighting the edge by a half makes the stimulated membrane exactly the strip's own area on
    any mesh, and on a mesh where no node lands on the edge -- the reference's own -- it
    reduces to the reference's test.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet the points belong to.
    points : array_like
        Coordinates, shape ``(n, 2)`` or ``(n, 3)``, in cm.

    Returns
    -------
    numpy.ndarray
        Shape ``(n,)``, each entry 0, 0.5 or 1.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    lx, ly, _ = setup.cell_size()
    tolerance = _ON_THE_EDGE * min(lx, ly)
    first, last = STIMULUS_ROWS

    def share(values: npt.NDArray[np.float64], edge: float, below: bool) -> Any:
        """One on the inner side of `edge`, a half on it, zero beyond."""
        inside = values <= edge - tolerance if below else values >= edge + tolerance
        on_the_edge = np.abs(values - edge) <= tolerance
        return np.where(on_the_edge, 0.5, inside.astype(np.float64))

    # The strip runs to the left edge of the sheet, where the domain ends rather than the
    # strip, so there is no share to take there.
    return share(coordinates[:, 0], STIMULUS_COLUMNS * lx, below=True) * np.minimum(
        share(coordinates[:, 1], first * ly, below=False),
        share(coordinates[:, 1], last * ly, below=True),
    )


def is_grounded(setup: BidomainSetup, points: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    """Which of `points` lie on the corner the extracellular potential is grounded at.

    An L of two segments one cell long meeting at the origin, which is what the reference's
    mesh marks as its second physical line. The extracellular potential is otherwise
    determined only up to a constant; grounding it at the same lower-left corner the network
    model grounds at is what makes the two comparable.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet the points belong to.
    points : array_like
        Coordinates, shape ``(n, 2)`` or ``(n, 3)``, in cm.

    Returns
    -------
    numpy.ndarray
        Boolean, shape ``(n,)``.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    lx, ly, _ = setup.cell_size()
    tolerance = _ON_THE_EDGE * min(lx, ly)
    x, y = coordinates[:, 0], coordinates[:, 1]
    along_x = (y <= tolerance) & (x <= lx + tolerance)
    along_y = (x <= tolerance) & (y <= ly + tolerance)
    return along_x | along_y


def build_mesh(setup: BidomainSetup) -> Any:
    """Build the sheet's mesh.

    Structured rather than read from a file, because the reference's own meshes are not
    distributed with this package and a figure script has to run from a clean clone. The
    element size is chosen to divide the cell, so the conductivity field is represented
    exactly; against the reference's unstructured mesh at its own 10 um, four fifths of whose
    elements straddle a cell boundary, this setup differs by 0.5% in conduction velocity.

    Each quadrilateral is cut by a single diagonal. The crossed alternative adds a node per
    quadrilateral and costs 2.1 times as much; on this benchmark the two give the same
    conduction velocity, because the wave runs along an axis and the right-diagonal pattern
    is unchanged by swapping the two axes.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet to mesh.

    Returns
    -------
    dolfinx.mesh.Mesh
        Triangles over ``[0, nx*lx] x [0, ny*ly]``.
    """
    width, height = setup.domain_size()
    nx, ny = setup.elements()
    return dolfinx.mesh.create_rectangle(
        MPI.COMM_WORLD,
        [np.array([0.0, 0.0]), np.array([width, height])],
        [nx, ny],
        cell_type=dolfinx.mesh.CellType.triangle,
        diagonal=dolfinx.mesh.DiagonalType.right,
    )


def cell_index(setup: BidomainSetup, points: npt.ArrayLike) -> npt.NDArray[np.intp]:
    """Which cell each of `points` falls in, in the reference's row-major order.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet the points belong to.
    points : array_like
        Coordinates, shape ``(n, 2)`` or ``(n, 3)``, in cm.

    Returns
    -------
    numpy.ndarray
        Indices into the per-cell arrays, shape ``(n,)``. A point on the far edge of the
        sheet belongs to the last cell rather than to one past it.
    """
    coordinates = np.asarray(points, dtype=np.float64)
    lx, ly, _ = setup.cell_size()
    columns = np.clip(np.floor(coordinates[:, 0] / lx).astype(np.intp), 0, setup.nx - 1)
    rows = np.clip(np.floor(coordinates[:, 1] / ly).astype(np.intp), 0, setup.ny - 1)
    return rows * setup.nx + columns


def intracellular_conductivity_field(setup: BidomainSetup, mesh: Any) -> Any:
    """The intracellular conductivity as a piecewise constant tensor over the mesh.

    Diagonal, and constant within a cell: an element takes the conductivity of the cell its
    midpoint lies in, and since every element lies inside one cell the field is exact rather
    than an average across a boundary.

    Divided by the surface-to-volume ratio, which is the form `fenicsx-beat` takes: the
    reference folds ``dt/(chi*Cm)`` into its own coefficient, where `beat` supplies the time
    step and the capacitance itself.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet.
    mesh : dolfinx.mesh.Mesh
        From `build_mesh`.

    Returns
    -------
    dolfinx.fem.Function
        In a discontinuous Lagrange tensor space of degree zero.
    """
    dimension = mesh.topology.dim
    element = basix.ufl.element("DG", mesh.basix_cell(), 0, shape=(dimension, dimension))
    space = dolfinx.fem.functionspace(mesh, element)

    index_map = mesh.topology.index_map(dimension)
    cells = np.arange(index_map.size_local + index_map.num_ghosts, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, dimension, cells)

    Mi_x, Mi_y = setup.intracellular_conductivity()
    index = cell_index(setup, midpoints)
    chi = setup.surface_to_volume_ratio()

    values = np.zeros((len(cells), dimension, dimension))
    values[:, 0, 0] = Mi_x[index] / chi
    values[:, 1, 1] = Mi_y[index] / chi

    field = dolfinx.fem.Function(space, name="M_i")
    field.x.array[:] = values.reshape(-1)
    return field


def monodomain_conductivity_field(setup: BidomainSetup, mesh: Any) -> Any:
    """The monodomain conductivity: the intracellular one scaled by ``lambda/(1 + lambda)``.

    The monodomain model replaces the two conductivities by one, on the assumption that a
    single ratio relates them. That collapses the elliptic equation into the parabolic one and
    leaves this harmonic-mean-like coefficient, which is the reference's own.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet.
    mesh : dolfinx.mesh.Mesh
        From `build_mesh`.

    Returns
    -------
    dolfinx.fem.Function
        In the same tensor space as `intracellular_conductivity_field`.
    """
    intracellular = intracellular_conductivity_field(setup, mesh)
    ratio = setup.conductance_ratio()
    field = dolfinx.fem.Function(intracellular.function_space, name="M")
    field.x.array[:] = ratio / (1.0 + ratio) * intracellular.x.array
    return field


def _ground_the_extracellular_potential(setup: BidomainSetup) -> Any:
    """Build the factory `beat` calls to place the corner condition on its own space."""

    def bcs(model: Any) -> list[Any]:
        dofs = dolfinx.fem.locate_dofs_geometrical(model.V_ue, lambda x: is_grounded(setup, x.T))
        value = dolfinx.default_scalar_type(0.0)
        return [dolfinx.fem.dirichletbc(value, dofs, model.V_ue)]

    return bcs


def build_model(setup: BidomainSetup, model: str, mesh: Any = None) -> Any:
    """Build one of the two continuum models over the sheet.

    Fully implicit, which is what the reference does, and solved by a direct factorization:
    reused across steps it costs less than the reference's preconditioned Krylov solver, and
    it leaves the run bound by the membrane model rather than by the linear algebra.

    Both conductivities are handed over with the surface-to-volume ratio divided out, since
    `fenicsx-beat` supplies the time step and the capacitance itself.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet.
    model : {"bidomain", "monodomain"}
        Which model to build.
    mesh : dolfinx.mesh.Mesh, optional
        The mesh to build on, by default a new one from `build_mesh`.

    Returns
    -------
    beat.BidomainModel or beat.MonodomainModel
        Ready to step.

    Raises
    ------
    ValueError
        If `model` names neither.
    """
    mesh = build_mesh(setup) if mesh is None else mesh
    time = dolfinx.fem.Constant(mesh, 0.0)
    parameters = {"theta": 1.0, "default_timestep": setup.dt}

    if model == "monodomain":
        return beat.MonodomainModel(
            time=time,
            mesh=mesh,
            M=monodomain_conductivity_field(setup, mesh),
            C_m=CM,
            params=parameters,
        )
    if model == "bidomain":
        extracellular = setup.extracellular_conductivity() / setup.surface_to_volume_ratio()
        return beat.BidomainModel(
            time=time,
            mesh=mesh,
            M_i=intracellular_conductivity_field(setup, mesh),
            # Isotropic and uniform, and the diagonal of the elliptic block is this plus
            # `M_i` -- which is what the reference assembles there.
            M_e=dolfinx.fem.Constant(mesh, np.diag([extracellular, extracellular])),
            C_m=CM,
            params=parameters,
            bcs=_ground_the_extracellular_potential(setup),
        )
    raise ValueError(f"model must be 'bidomain' or 'monodomain', got {model!r}")


@dataclasses.dataclass(frozen=True)
class Measurement:
    """What one run of a continuum model is read for.

    Parameters
    ----------
    conduction_velocity : float
        Along the measurement row, in cm/s.
    upstroke_rate : float
        The steepest rise of the membrane potential at the centre of the sheet, in V/s, which
        is the same number as mV/ms.
    start_time, end_time : float
        When the wave reached each of the two probes, in ms.
    peak_potential : float
        The highest membrane potential seen at the centre, in mV. A run where nothing
        propagates still depolarizes a little, and this is what tells the two apart.
    """

    conduction_velocity: float
    upstroke_rate: float
    start_time: float
    end_time: float
    peak_potential: float


class _PointProbe:
    """Evaluate a finite element function at a fixed set of points.

    The probes sit at cell centres, which are nodes only when a cell is divided into an even
    number of elements, so the value is interpolated within whichever element contains the
    point rather than read off a degree of freedom.
    """

    def __init__(self, mesh: Any, points: npt.ArrayLike) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        tree = dolfinx.geometry.bb_tree(mesh, mesh.topology.dim)
        candidates = dolfinx.geometry.compute_collisions_points(tree, self.points)
        colliding = dolfinx.geometry.compute_colliding_cells(mesh, candidates, self.points)
        self.cells = np.array(
            [colliding.links(index)[0] for index in range(len(self.points))], dtype=np.int32
        )

    def __call__(self, function: Any) -> npt.NDArray[np.float64]:
        return np.asarray(function.eval(self.points, self.cells)).reshape(-1)


def membrane_substeps(dt: float) -> tuple[float, int]:
    """The membrane model's own step, and how many of them make up one step of the PDE.

    The reference fixes its ODE step at 0.01 ms and takes ``round(dt/dt_ode)`` of them, which
    is **zero** for every time step below 0.005: the membrane model is then never advanced at
    all, its clock never moves, and the resting state overwrites the spatial solve on every
    step. Taking the ODE step as the smaller of the two is what prevents that, and it also
    keeps the two clocks together, since the substeps then add up to exactly one time step --
    which the reference's arithmetic does only when the time step is a multiple of 0.01.

    Parameters
    ----------
    dt : float
        The time step of the spatial solve, in ms.

    Returns
    -------
    tuple of (float, int)
        The ODE step in ms, and how many to take. Their product is `dt`.
    """
    step = min(dt, MAX_ODE_STEP)
    return step, round(dt / step)


def probe_points(setup: BidomainSetup) -> npt.NDArray[np.float64]:
    """The three points a run is read at: the two velocity probes and the centre.

    The paper's protocol, shared by all four of its models: the wave is timed between the
    centres of cell columns 9 and 34, on the row halfway up the sheet, and the upstroke is
    recorded at the centre of the domain.

    Which row the velocity is read along matters. The stimulated strip spans rows 14 to 24, so
    the wave reaches a row near the bottom edge later than one level with the strip: moving the
    far probe to a quarter of the way up costs 1.0% of the velocity, and to a fortieth, 2.5%.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet.

    Returns
    -------
    numpy.ndarray
        Shape ``(3, 3)``: the near probe, the far probe and the centre, in cm.
    """
    lx, _, _ = setup.cell_size()
    width, height = setup.domain_size()
    first, last = CONDUCTION_COLUMNS
    return np.array(
        [
            [(first + 0.5) * lx, 0.5 * height, 0.0],
            [(last + 0.5) * lx, 0.5 * height, 0.0],
            [0.5 * width, 0.5 * height, 0.0],
        ]
    )


def _membrane_parameters(setup: BidomainSetup, model: Any, space: Any) -> Any:
    """The membrane model's parameters, with the stimulus written into the strip.

    The stimulus is the cell model's own amplitude parameter rather than a source term in the
    equation, which is what the reference does and what `sknm` does. That makes the comparison
    between them one of models rather than one of stimulus protocols.
    """
    coordinates = space.tabulate_dof_coordinates()
    parameters = model.initial_parameters(len(coordinates))
    amplitude = presets.STIMULUS_AMPLITUDE * stimulus_weight(setup, coordinates)
    parameters[model.parameter_index("stim_amplitude")] = amplitude
    return parameters


def run(
    setup: BidomainSetup,
    model: str,
    *,
    callback: Any = None,
    minimum_time: float = 0.0,
) -> Measurement:
    """Solve one continuum model over the sheet and measure the wave it carries.

    Operator splitting, as the reference does it: the membrane model advances every node's
    state over the step, the potential it produced becomes the right-hand side of the linear
    system, and the solution is written back into the states. The run stops as soon as the
    wave reaches the far probe, which on the paper's sheet is at about 37 of the 50 ms a full
    action potential takes.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet.
    model : {"bidomain", "monodomain"}
        Which model to solve.
    callback : callable, optional
        Called as ``callback(t, pde)`` with the time reached and the model holding the
        solution: once at ``t = 0`` before any step, and then after each one. This is how the
        snapshots and the extracellular range are read off a run without the loop knowing
        about either.
    minimum_time : float, optional
        How long to keep running whatever the wave does, in ms, by default 0. The run
        otherwise stops as soon as the wave reaches the far probe, which is before the times
        the snapshots are taken at.

    Returns
    -------
    Measurement
        The velocity, the upstroke and when the wave passed each probe.

    Raises
    ------
    ValueError
        If the wave never reached the far probe, or reached both in the same step, so there is
        no velocity to report.
    """
    mesh = build_mesh(setup)
    pde = build_model(setup, model, mesh)
    membrane = from_gotranx(base_model_IM)

    states = membrane.initial_states(len(pde.V.tabulate_dof_coordinates()))
    parameters = _membrane_parameters(setup, membrane, pde.V)
    v_index = membrane.v_index

    probe = _PointProbe(mesh, probe_points(setup))
    ode_step, substeps = membrane_substeps(setup.dt)

    # A fresh finite element function is zero, which is +77 mV away from where a cell at rest
    # sits. The first step overwrites it, so no measurement would notice, but the snapshots
    # are read off this field and its first frame would be a sheet at zero.
    pde.v.x.array[:] = states[v_index]

    start_time = end_time = None
    peak = -np.inf
    fastest = 0.0
    previous = probe(pde.v)
    if callback is not None:
        callback(0.0, pde)

    for step in range(round(setup.t_end / setup.dt)):
        t0 = step * setup.dt
        for substep in range(substeps):
            states = membrane.step(states, t0 + substep * ode_step, ode_step, parameters)

        pde.assign_previous()
        pde.v_.x.array[:] = states[v_index]
        pde.step((t0, t0 + setup.dt))
        states[v_index] = pde.v.x.array

        t1 = (step + 1) * setup.dt
        values = probe(pde.v)
        fastest = max(fastest, (values[2] - previous[2]) / setup.dt)
        peak = max(peak, values[2])
        if callback is not None:
            callback(t1, pde)
        if start_time is None and values[0] >= setup.threshold:
            start_time = t1
        if end_time is None and values[1] >= setup.threshold:
            end_time = t1
        if end_time is not None and t1 >= minimum_time:
            break
        previous = values

    if start_time is None or end_time is None:
        raise ValueError(
            f"the wave did not reach both probes within {setup.t_end} ms; the highest "
            f"membrane potential at the centre was {peak:.4g} mV"
        )
    if end_time == start_time:
        # Both crossings are rounded to a whole time step, so a step long enough to cover the
        # whole transit reports them as one moment and there is no time difference to divide
        # by. A velocity of infinity is not the answer.
        raise ValueError(
            f"the wave did not reach the two probes in different steps at dt = {setup.dt} ms, "
            f"so its velocity cannot be resolved; use a shorter step"
        )

    first, last = CONDUCTION_COLUMNS
    lx, _, _ = setup.cell_size()
    # A length in cm over a time in ms, reported in cm/s.
    distance = (last - first) * lx * 1e3
    return Measurement(
        conduction_velocity=distance / (end_time - start_time),
        upstroke_rate=fastest,
        start_time=start_time,
        end_time=end_time,
        peak_potential=float(peak),
    )


def to_grid(setup: BidomainSetup, space: Any, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Lay a field out as the grid of nodes it was solved on.

    A finite element function's degrees of freedom come in the order the mesh numbers its
    vertices, which is not row by row, so reshaping the array draws a scrambled sheet. The
    mesh here is structured, so each node's row and column follow from its coordinates.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet.
    space : dolfinx.fem.FunctionSpace
        The space the values live in, of degree one.
    values : array_like
        One value per degree of freedom.

    Returns
    -------
    numpy.ndarray
        Shape ``(ny + 1, nx + 1)`` for the mesh's `nx` by `ny` elements, with row zero at the
        bottom of the sheet.
    """
    nx, ny = setup.elements()
    width, height = setup.domain_size()
    coordinates = space.tabulate_dof_coordinates()
    columns = np.rint(coordinates[:, 0] / (width / nx)).astype(np.intp)
    rows = np.rint(coordinates[:, 1] / (height / ny)).astype(np.intp)

    grid = np.full((ny + 1, nx + 1), np.nan)
    grid[rows, columns] = np.asarray(values, dtype=np.float64)
    return grid


def snapshots(setup: BidomainSetup, model: str, times: Sequence[float]) -> npt.NDArray[np.float64]:
    """Record the membrane potential across the whole sheet at each of a series of times.

    A run stops as soon as the wave reaches the far probe, which is earlier than the last
    time a snapshot figure asks for, so the run is held open to the last of them.

    Parameters
    ----------
    setup : BidomainSetup
        The sheet to run.
    model : {"bidomain", "monodomain"}
        Which model to solve.
    times : sequence of float
        When to read the sheet, in ms, in increasing order. A time is caught on the step
        whose end is within half a step of it.

    Returns
    -------
    numpy.ndarray
        Shape ``(len(times), ny + 1, nx + 1)``, in mV, laid out as the grid.

    Raises
    ------
    ValueError
        If the run ended before a time was reached. Returning the frames that were captured
        would draw the figure with a panel missing or a moment repeated.
    """
    remaining = list(times)
    frames = []

    def capture(t: float, pde: Any) -> None:
        if remaining and t >= remaining[0] - 0.5 * setup.dt:
            remaining.pop(0)
            frames.append(to_grid(setup, pde.V, pde.v.x.array.copy()))

    run(setup, model, callback=capture, minimum_time=max(times))
    if remaining:
        raise ValueError(f"the run ended before {remaining} ms")
    return np.stack(frames)
