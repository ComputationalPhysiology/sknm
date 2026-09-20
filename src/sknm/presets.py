"""The paper's own setup for human induced pluripotent stem cell derived cardiomyocytes.

Everything here is data from Jaeger & Tveito (2023) and its reference implementation, put
behind names rather than left to be retyped: the five measured cell sizes, the material
constants, the sheet they are arranged in, where the stimulus goes and where the two published
numbers are measured. The general constructors in `sknm.network` take explicit geometry
instead; this module is what to reach for when the intent is "the paper's simulation"::

    network = presets.hipsc_sheet(40, 40)
    sim = Simulation(network, from_gotranx(base_model_IM), dt=0.02 * ms)
    sim.set_parameter("stim_amplitude", presets.hipsc_stimulus_amplitude(40, 40))

Two of these constants are easy to get wrong from the paper alone and are worth naming:

- `MEMBRANE_AREA` is a **constant**, the same for every anisotropy factor. It is not the
  surface area of the cuboid the cell dimensions describe, which `sheet` would compute by
  default and which is 3.4% smaller at ``alpha=1``.
- `hipsc_stimulus_amplitude` returns an amplitude for **every** cell, zero outside the
  stimulated region. A membrane model carries a stimulus amplitude of its own, and setting one
  only on the stimulated cells leaves every other cell stimulating itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from sknm import units
from sknm.analysis import ConductionPath
from sknm.network import CellNetwork, sheet

#: Cell length and width at the five anisotropy factors the paper's Figure 3 is drawn at.
#: `hipsc_cell_size` reproduces every one of them exactly; these are the values the reference
#: implementation ships meshes for, and so the sample a reproduction of that figure uses.
CELL_DIMENSIONS: Mapping[float, tuple[Any, Any]] = MappingProxyType(
    {
        1.0: (16 * units.um, 16 * units.um),
        1.5: (21 * units.um, 14 * units.um),
        2.0: (25 * units.um, 12.5 * units.um),
        3.0: (33 * units.um, 11 * units.um),
        4.0: (40 * units.um, 10 * units.um),
    }
)

#: Intracellular conductivity.
SIGMA_I = 4.0 * units.mS / units.cm
#: Extracellular conductivity.
SIGMA_E = 20.0 * units.mS / units.cm
#: Specific membrane capacitance.
CM = 1.0 * units.uF / units.cm**2
#: Membrane area of one cell. Constant across anisotropy factors, and not the cuboid's surface.
MEMBRANE_AREA = 1.8e-5 * units.cm**2
#: Gap junction conductance, the reference's gap junction resistance of 5e3 kOhm inverted.
GAP_JUNCTION_CONDUCTANCE = (1 / (5e3 * units.kohm)).to(units.mS)
#: Extracellular volume fraction.
DELTA_E = 0.2
#: Stimulus current amplitude, in the membrane model's own convention, so bare.
STIMULUS_AMPLITUDE = 20.0

#: Intracellular volume of one cell in um^3, which `hipsc_cell_size` holds fixed.
_CELL_VOLUME = 4000.0

#: Columns of the two cells a conduction velocity is measured between, and the number of cells
#: between them, which sets the distance.
_CONDUCTION_COLUMNS = (9, 34)
#: Depth and height of the stimulated region, in cells. The reference stimulates
#: ``x < 2*lx`` and ``14*ly < y < 25*ly`` with a node at each cell centre, which is the two
#: leftmost columns over eleven rows centred on the row the velocity is measured along.
_STIMULUS_COLUMNS = 2
_STIMULUS_ROWS = 11


def hipsc_sheet(
    nx: int = 40,
    ny: int = 40,
    *,
    alpha: Any = 1.0,
    delta_e: Any = DELTA_E,
    Gg: Any = GAP_JUNCTION_CONDUCTANCE,
) -> CellNetwork:
    """Build the paper's sheet of hiPSC-CMs.

    Parameters
    ----------
    nx, ny : int, optional
        Number of cells along x and along y, by default 40 each, the paper's sheet.
    alpha : float, optional
        Anisotropy factor, by default 1.0, setting the cell size through `hipsc_cell_size`.
        Dimensionless, so a bare number.
    delta_e : float, optional
        Extracellular volume fraction, by default 0.2. Sets the cell's third dimension,
        ``lz = (1 + delta_e) * ly``, as well as the extracellular conductance. Dimensionless.
    Gg : pint.Quantity, optional
        Gap junction conductance of each connection, by default
        `GAP_JUNCTION_CONDUCTANCE`. The paper tunes this to bring the conduction velocity to
        roughly 4 cm/s.

    Returns
    -------
    CellNetwork
        A sheet of ``nx * ny`` cells, each connected to its four neighbours, numbered row-major.

    Raises
    ------
    TypeError
        If `Gg` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `alpha` or `delta_e` carries a unit, or `Gg` is not a conductance.
    ValueError
        If `alpha` is not positive, if `delta_e` is not a single number, or if `nx` or `ny` is
        less than one.

    Examples
    --------
    >>> from sknm import presets
    >>> network = presets.hipsc_sheet(40, 40)
    >>> network.n_cells
    1600
    >>> round(network.lam, 2)
    39.65
    """
    lx, ly = hipsc_cell_size(alpha)
    fraction = units.as_number(delta_e, name="delta_e")
    if np.ndim(fraction) != 0:
        raise ValueError(
            f"delta_e must be a single number here, because the cell's third dimension is "
            f"derived from it; got an array of shape {np.shape(fraction)}. Build the sheet with "
            f"`sknm.sheet` to give each cell its own volume fraction."
        )
    return sheet(
        nx,
        ny,
        lx=lx,
        ly=ly,
        lz=(1.0 + float(fraction)) * ly,
        delta_e=fraction,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
        Gg=Gg,
        membrane_area=MEMBRANE_AREA,
        Cm=CM,
    )


def hipsc_stimulus_amplitude(nx: int = 40, ny: int = 40) -> npt.NDArray[np.float64]:
    """The stimulus amplitude of every cell of the sheet, for `Simulation.set_parameter`.

    `STIMULUS_AMPLITUDE` over the stimulated region and **zero everywhere else**. The zeros are
    the point: a membrane model has a stimulus amplitude of its own, and raising the amplitude
    only where the stimulus belongs leaves every other cell firing on its own schedule, which
    looks like a wave and travels at the wrong speed.

    The region is the reference's: the two leftmost columns, over eleven rows centred on the row
    a conduction velocity is measured along. On a sheet too small to hold it, it is clipped.

    Parameters
    ----------
    nx, ny : int, optional
        Shape of the sheet, by default 40 by 40.

    Returns
    -------
    numpy.ndarray
        Shape ``(nx * ny,)``, in the membrane model's own convention, so carrying no unit.

    Raises
    ------
    ValueError
        If the region would not cover a single cell.

    Examples
    --------
    >>> from sknm import presets
    >>> amplitude = presets.hipsc_stimulus_amplitude(40, 40)
    >>> int((amplitude > 0).sum())
    22
    """
    amplitude = np.zeros((ny, nx), dtype=np.float64)
    centre = _measurement_row(ny)
    first = max(0, centre - _STIMULUS_ROWS // 2)
    last = min(ny, first + _STIMULUS_ROWS)
    amplitude[first:last, :_STIMULUS_COLUMNS] = STIMULUS_AMPLITUDE
    if not amplitude.any():
        raise ValueError(
            f"a {nx} by {ny} sheet leaves the stimulated region without at least one cell in it"
        )
    return amplitude.reshape(-1)


def hipsc_conduction_path(nx: int = 40, ny: int = 40, *, alpha: Any = 1.0) -> ConductionPath:
    """The two cells the paper measures a conduction velocity between, and their separation.

    Cells at columns 9 and 34 of the row halfway up the sheet, twenty-five cell lengths apart.
    Far enough from the stimulus that the wave is travelling freely rather than still spreading
    out of the cells that were driven.

    Parameters
    ----------
    nx, ny : int, optional
        Shape of the sheet, by default 40 by 40.
    alpha : float, optional
        Anisotropy factor the sheet was built with, by default 1.0. It sets the cell length and
        so the distance; passing one that does not match the network understates or overstates
        the velocity with nothing else to show for it.

    Returns
    -------
    ConductionPath
        Ready for `sknm.analysis.conduction_velocity`.

    Raises
    ------
    ValueError
        If `alpha` is not positive, or if the sheet is too narrow to hold both cells.

    Examples
    --------
    >>> from sknm import presets
    >>> path = presets.hipsc_conduction_path(40, 40)
    >>> path.start, path.end
    (769, 794)
    """
    lx, _ = hipsc_cell_size(alpha)
    first, last = _CONDUCTION_COLUMNS
    if nx <= last:
        raise ValueError(
            f"measuring between columns {first} and {last} needs a sheet at least {last + 1} "
            f"cells wide, but this one is {nx}. Build a `ConductionPath` directly for a sheet "
            f"of another size."
        )
    row = _measurement_row(ny) * nx
    return ConductionPath(start=row + first, end=row + last, distance=(last - first) * lx)


def hipsc_centre_cell(nx: int = 40, ny: int = 40) -> int:
    """The cell at the centre of the sheet, whose trace the reference saves every step.

    Where the maximal upstroke velocity is read off: the wave is fully developed there, and it
    is far from both the stimulus and the far boundary.

    Parameters
    ----------
    nx, ny : int, optional
        Shape of the sheet, by default 40 by 40.

    Returns
    -------
    int
        Index of the cell.

    Examples
    --------
    >>> from sknm import presets
    >>> presets.hipsc_centre_cell(40, 40)
    780
    """
    return _measurement_row(ny) * nx + nx // 2


def vary_conductances(network: CellNetwork, gamma: Any, draws: npt.ArrayLike) -> Any:
    """Spread a network's gap junction conductances around their nominal value.

    The reference varies the gap junction *resistance*,
    ``Rg = Rg0 / (a*(1 - gamma) + (1 - a)*(1 + gamma))``, from one draw `a` per connection. In
    conductance that is a multiplier uniform over ``[1 - gamma, 1 + gamma]``, so ``gamma = 0``
    returns the conductances unchanged whatever the draws are.

    The draws are an argument rather than something this function makes, because the paper's
    sweep reuses **one** set across every value of `gamma` and every variant: that is what makes
    the resulting curves comparable point for point rather than each a different network. Make
    them once, with a seed::

        draws = numpy.random.default_rng(0).random(network.n_connections)

    The reference's own draws, if you have them, are in `random_picks/gj_scale_x.txt` and
    `gj_scale_y.txt`, concatenated in that order: `sheet` numbers its connections x-direction
    first and then y-direction, both row-major, which is the layout those files are written in.

    Parameters
    ----------
    network : CellNetwork
        The network whose conductances to spread. It is not modified; pass the result to
        `CellNetwork.with_conductances`.
    gamma : float
        Gap junction variation, between 0 and 1. Dimensionless.
    draws : array_like
        One draw per connection, each between 0 and 1.

    Returns
    -------
    pint.Quantity
        The varied conductances, one per connection, in mS.

    Raises
    ------
    pint.DimensionalityError
        If `gamma` carries a unit.
    ValueError
        If `gamma` is outside ``[0, 1]``, if there is not exactly one draw per connection, or if
        a draw is outside ``[0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from sknm import presets
    >>> network = presets.hipsc_sheet(4, 4)
    >>> draws = np.random.default_rng(0).random(network.n_connections)
    >>> varied = presets.vary_conductances(network, 0.5, draws)
    >>> bool(np.all(varied.magnitude >= 0.5 * network.Gg))
    True
    """
    variation = float(units.as_number(gamma, name="gamma"))
    if not 0.0 <= variation <= 1.0:
        raise ValueError(
            f"gamma must be between 0 and 1, got {variation}. Beyond 1 the formula turns some "
            f"gap junction conductances negative."
        )
    values = np.asarray(draws, dtype=np.float64)
    if values.shape != (network.n_connections,):
        raise ValueError(
            f"there must be one draw per connection, {network.n_connections} of them, but "
            f"draws has shape {values.shape}"
        )
    if values.size and (values.min() < 0.0 or values.max() > 1.0):
        raise ValueError(
            f"draws must be between 0 and 1, got a range of {values.min()} to {values.max()}"
        )
    return units.with_base_units(
        network.Gg * (1.0 + variation * (1.0 - 2.0 * values)), "conductance"
    )


def hipsc_cell_size(alpha: Any = 1.0) -> tuple[Any, Any]:
    """Cell length and width at an anisotropy factor, holding the cell's volume fixed.

    Making a cell longer makes it correspondingly narrower, so that its intracellular volume
    stays at the roughly 4 pL a hiPSC-CM has::

        ly = round(cbrt(4000 um^3 / alpha) * 2) / 2   um, to the nearest half micrometre
        lx = alpha * ly

    The rounding is why the volume comes out between 3.9 and 4.1 pL rather than exactly 4, and
    it is what makes the five sizes in `CELL_DIMENSIONS` come out at the round numbers they do.
    The third dimension is not set here: it follows the extracellular volume fraction,
    ``lz = (1 + delta_e) * ly``, which `hipsc_sheet` applies.

    Parameters
    ----------
    alpha : float, optional
        Anisotropy factor, the cell's length-to-width ratio, by default 1.0. Dimensionless, so
        a bare number.

    Returns
    -------
    tuple of pint.Quantity
        Length along x and width along y, as lengths.

    Raises
    ------
    pint.DimensionalityError
        If `alpha` carries a unit.
    ValueError
        If `alpha` is not positive and finite.

    Examples
    --------
    >>> from sknm import presets
    >>> lx, ly = presets.hipsc_cell_size(4.0)
    >>> float(lx.m_as("um")), float(ly.m_as("um"))
    (40.0, 10.0)
    """
    factor = float(units.as_number(alpha, name="alpha"))
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"alpha must be positive and finite, got {factor}")
    ly = round(np.cbrt(_CELL_VOLUME / factor) * 2.0) / 2.0
    return factor * ly * units.um, ly * units.um


def _measurement_row(ny: int) -> int:
    """The row of the sheet the reference measures along, just below the middle."""
    return round(ny / 2) - 1
