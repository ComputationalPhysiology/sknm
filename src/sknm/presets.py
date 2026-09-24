"""The paper's own two setups: hiPSC-derived cardiomyocytes, and pancreatic beta cells.

Everything here is data from Jaeger & Tveito (2023) and its reference implementation, put
behind names rather than left to be retyped: the cell sizes, the material constants, the sheet
they are arranged in, where the stimulus goes and where the published numbers are measured. The
general constructors in `sknm.network` take explicit geometry instead; this module is what to
reach for when the intent is "the paper's simulation":

```python
network = presets.hipsc_sheet(40, 40)
sim = Simulation(network, from_gotranx(base_model_IM), dt=0.02 * ms)
sim.set_parameter("stim_amplitude", presets.hipsc_stimulus_amplitude(40, 40))

network = presets.beta_sheet(15, 15)
sim = Simulation(network, presets.beta_membrane_model(), dt=0.02 * ms)
sim.set_parameter("gkatpbar", presets.beta_stimulus_conductance(15, 15))
```

The two reference drivers are structurally identical and differ only in constants, so the two
families here mirror one another. Four of those constants are easy to get wrong from the paper
alone and are worth naming:

- `MEMBRANE_AREA` is a constant, the same for every anisotropy factor. It is not the surface
  area of the cuboid the cell dimensions describe, which `sheet` would compute by default and
  which is 3.4% smaller at ``alpha=1``.
- `hipsc_stimulus_amplitude` returns an amplitude for every cell, zero outside the stimulated
  region. A membrane model carries a stimulus amplitude of its own, and setting one only on the
  stimulated cells leaves every other cell stimulating itself.
- A beta cell's third dimension is fixed at its cell size. The hiPSC sheet derives its third
  dimension from the extracellular volume fraction instead, and at ``delta_e=0.5`` that rule
  would give 19.5 um where the reference uses 13.
- `BETA_CM` is derived, not the round 1.0 uF/cm2. A beta cell's membrane area is the surface of
  a sphere and its membrane model carries an absolute capacitance of 5300 fF; the two disagree
  by 0.18% if the specific capacitance is assumed. See `BETA_CM`.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np
import numpy.typing as npt

from sknm import units
from sknm.analysis import ConductionPath
from sknm.membrane import PBM, MembraneModel, from_gotranx
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

#: Membrane potential at which a hiPSC-CM counts as activated. `sknm.analysis` has no default
#: threshold, because this one applied to a beta cell would leave every cell unactivated.
HIPSC_THRESHOLD = -20.0 * units.mV

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

# --- the pancreatic beta cell setup ---------------------------------------------------------

#: Diameter of a beta cell, and every one of its three dimensions. Unlike a hiPSC-CM, whose
#: third dimension follows the extracellular volume fraction, this is fixed.
BETA_CELL_SIZE = 13.0 * units.um
#: Membrane area of one beta cell: the surface of a sphere of diameter `BETA_CELL_SIZE`, which
#: is 5.3093e-6 cm2 and is the reference's stated constant to every digit it quotes. A beta cell
#: is modelled as a sphere; the cuboid its cell size describes would have nearly twice the area.
BETA_MEMBRANE_AREA = (np.pi * BETA_CELL_SIZE**2).to(units.cm**2)
#: Absolute membrane capacitance of one beta cell, the value `sknm.membrane.PBM`'s voltage
#: equation divides by. Published by Bertram & Sherman (2004) with the rest of the model.
BETA_CAPACITANCE = 5300.0 * units.fF
#: Specific membrane capacitance of a beta cell, derived so that ``BETA_CM *
#: BETA_MEMBRANE_AREA`` is exactly `BETA_CAPACITANCE`, instead of assumed to be the round
#: 1.0 uF/cm2 the reference writes.
#:
#: The three numbers cannot all be round at once: 5300 fF is a published measurement, the
#: membrane area is pi*d^2 at a diameter rounded to 13 um, and their quotient is 0.998248
#: uF/cm2. The reference carries 5300 fF in the membrane model and 1.0 uF/cm2 in the network,
#: which disagree by 0.18%. That is enough for `sknm.Simulation` to reject the pairing, since
#: the two capacitances have to be the same number for the split to conserve charge. Of the
#: three, the specific capacitance is the only generic constant and not a measurement, so it is
#: the one that gives way. The effect on a conduction velocity is about 0.1%.
BETA_CM = (BETA_CAPACITANCE / BETA_MEMBRANE_AREA).to(units.uF / units.cm**2)
#: Gap junction conductance between beta cells, the reference's 5e6 kOhm inverted. A thousand
#: times weaker than the cardiac one, which is most of why a beta wave is 150 times slower.
BETA_GAP_JUNCTION_CONDUCTANCE = (1 / (5e6 * units.kohm)).to(units.mS)
#: Extracellular volume fraction of the beta cell setup.
BETA_DELTA_E = 0.5
#: The beta cell membrane model's own K-ATP conductance, in its own convention, so bare.
BETA_KATP_CONDUCTANCE = 500.0
#: The stimulated value of that conductance. The beta stimulus halves a conductance instead of
#: injecting a current, so the unstimulated cells keep the model's own default.
BETA_STIMULUS_KATP_CONDUCTANCE = 250.0
#: Membrane potential at which a beta cell counts as activated. A beta action potential peaks at
#: about -19.5 mV, so `HIPSC_THRESHOLD` applied here would activate almost nothing.
BETA_THRESHOLD = -50.0 * units.mV

#: The reference's beta cell measurement columns and stimulated region, in the same form as the
#: cardiac ones above: ``x < 2*lx`` and ``5*ly < y < 10*ly`` with a node at each cell centre.
_BETA_CONDUCTION_COLUMNS = (4, 12)
_BETA_STIMULUS_COLUMNS = 2
_BETA_STIMULUS_ROWS = 5


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

    `STIMULUS_AMPLITUDE` over the stimulated region and zero everywhere else. The zeros matter:
    a membrane model has a stimulus amplitude of its own, so raising the amplitude only where
    the stimulus belongs leaves every other cell firing on its own schedule, which looks like a
    wave and travels at the wrong speed.

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
    return _stimulated_region(
        nx,
        ny,
        columns=_STIMULUS_COLUMNS,
        rows=_STIMULUS_ROWS,
        stimulated=STIMULUS_AMPLITUDE,
        elsewhere=0.0,
    )


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
        so the distance, so one that does not match the network scales the velocity by the
        ratio between them.

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
    return _conduction_path(nx, ny, columns=_CONDUCTION_COLUMNS, lx=lx)


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
    return _centre_cell(nx, ny)


def vary_conductances(network: CellNetwork, gamma: Any, draws: npt.ArrayLike) -> Any:
    """Spread a network's gap junction conductances around their nominal value.

    The reference varies the gap junction resistance,
    ``Rg = Rg0 / (a*(1 - gamma) + (1 - a)*(1 + gamma))``, from one draw `a` per connection. In
    conductance that is a multiplier uniform over ``[1 - gamma, 1 + gamma]``, so ``gamma = 0``
    returns the conductances unchanged whatever the draws are.

    The draws are an argument rather than something this function makes, because the paper's
    sweep reuses a single set across every value of `gamma` and every variant. That is what
    makes the resulting curves comparable point for point, instead of each one being a
    different network. Make them once, with a seed:

    ```python
    draws = numpy.random.default_rng(0).random(network.n_connections)
    ```

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
    stays at the roughly 4 pL a hiPSC-CM has:

    ```text
    ly = round(cbrt(4000 um^3 / alpha) * 2) / 2   um, to the nearest half micrometre
    lx = alpha * ly
    ```

    The rounding is why the volume comes out between 3.9 and 4.1 pL rather than exactly 4, and
    it is why the five sizes in `CELL_DIMENSIONS` come out at the round numbers they do.
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
    """The row of the sheet the reference measures along, just below the middle.

    The C writes `round(num_cells_y/2) - 1`, where both operands are ints, so the division
    truncates before `round` ever sees it: at ``ny = 15`` it is 6, not 7. Reading that
    expression as floating point agrees at every even `ny` and is one row out at every odd one.
    """
    return ny // 2 - 1


def _centre_cell(nx: int, ny: int) -> int:
    """The cell whose trace the reference saves every step. One formula for both setups."""
    return _measurement_row(ny) * nx + nx // 2


def _stimulated_region(
    nx: int, ny: int, *, columns: int, rows: int, stimulated: float, elsewhere: float
) -> npt.NDArray[np.float64]:
    """A per-cell parameter array: `stimulated` over the region, `elsewhere` outside it.

    The region is the leftmost `columns` columns over a block of `rows` rows centred in the
    sheet, with an uneven remainder going to the lower rows. That reproduces both of the
    reference's windows, which are written in cell widths with a node at each cell centre:
    ``14*ly < y < 25*ly`` is rows 14 to 24 of 40, and ``5*ly < y < 10*ly`` is rows 5 to 9 of 15.

    Centred in the sheet is not the same as centred on the row a velocity is measured along.
    They coincide on the cardiac sheet and differ by one row on the beta sheet, so deriving
    this from `_measurement_row` would move the beta stimulus off the reference's cells.

    Written as ``first + rows`` rather than ``centre +/- rows // 2``, which silently rounds an
    even row count up to the next odd one.
    """
    values = np.full((ny, nx), float(elsewhere), dtype=np.float64)
    first = max(0, (ny - rows) // 2)
    last = min(ny, first + rows)
    values[first:last, :columns] = stimulated
    if not (values == stimulated).any():
        raise ValueError(
            f"a {nx} by {ny} sheet leaves the stimulated region without at least one cell in it"
        )
    return values.reshape(-1)


def _conduction_path(nx: int, ny: int, *, columns: tuple[int, int], lx: Any) -> ConductionPath:
    """The two cells a velocity is measured between, and the distance along the row."""
    first, last = columns
    if nx <= last:
        raise ValueError(
            f"measuring between columns {first} and {last} needs a sheet at least {last + 1} "
            f"cells wide, but this one is {nx}. Build a `ConductionPath` directly for a sheet "
            f"of another size."
        )
    row = _measurement_row(ny) * nx
    return ConductionPath(start=row + first, end=row + last, distance=(last - first) * lx)


def beta_sheet(
    nx: int = 15,
    ny: int = 15,
    *,
    delta_e: Any = BETA_DELTA_E,
    Gg: Any = BETA_GAP_JUNCTION_CONDUCTANCE,
) -> CellNetwork:
    """Build the paper's sheet of pancreatic beta cells.

    Differs from `hipsc_sheet` in more than its constants. There is no anisotropy factor, since
    a beta cell is a cube of side `BETA_CELL_SIZE`, and the third dimension is fixed at that
    side rather than derived from `delta_e`. So raising the extracellular volume fraction
    changes the extracellular conductance without changing the cell.

    Parameters
    ----------
    nx, ny : int, optional
        Number of cells along x and along y, by default 15 each, the paper's sheet.
    delta_e : float, optional
        Extracellular volume fraction, by default 0.5. Dimensionless, so a bare number. It sets
        the extracellular conductance only; the cell's dimensions do not depend on it.
    Gg : pint.Quantity, optional
        Gap junction conductance of each connection, by default
        `BETA_GAP_JUNCTION_CONDUCTANCE`.

    Returns
    -------
    CellNetwork
        A sheet of ``nx * ny`` cells, each connected to its four neighbours, numbered row-major.
        Its specific capacitance is `BETA_CM`, so that it pairs with `beta_membrane_model`.

    Raises
    ------
    TypeError
        If `Gg` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `delta_e` carries a unit, or `Gg` is not a conductance.
    ValueError
        If `delta_e` is not a single number, or if `nx` or `ny` is less than one.

    Examples
    --------
    >>> from sknm import presets
    >>> network = presets.beta_sheet(15, 15)
    >>> network.n_cells
    225
    >>> round(network.lam)
    65005
    """
    fraction = units.as_number(delta_e, name="delta_e")
    if np.ndim(fraction) != 0:
        raise ValueError(
            f"delta_e must be a single number here; got an array of shape "
            f"{np.shape(fraction)}. Build the sheet with `sknm.sheet` to give each cell its "
            f"own volume fraction."
        )
    return sheet(
        nx,
        ny,
        lx=BETA_CELL_SIZE,
        ly=BETA_CELL_SIZE,
        lz=BETA_CELL_SIZE,
        delta_e=fraction,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
        Gg=Gg,
        membrane_area=BETA_MEMBRANE_AREA,
        Cm=BETA_CM,
    )


def beta_membrane_model() -> MembraneModel:
    """The phantom bursting beta cell model, with the capacitance its voltage equation uses.

    `sknm.membrane.PBM` names its membrane potential ``v`` rather than ``V_m`` and integrates
    ``dv/dt = -I / Cm``, so both have to be bound when it is wrapped. Declaring the capacitance
    lets `sknm.Simulation` check the network against it, and is why `beta_sheet` derives
    `BETA_CM` instead of rounding it.

    Returns
    -------
    MembraneModel
        Ready to pair with `beta_sheet`.

    Examples
    --------
    >>> from sknm import presets
    >>> model = presets.beta_membrane_model()
    >>> model.num_states, model.v_name
    (5, 'v')
    """
    return from_gotranx(PBM, v_name="v", capacitance=BETA_CAPACITANCE)


def beta_stimulus_conductance(nx: int = 15, ny: int = 15) -> npt.NDArray[np.float64]:
    """The K-ATP conductance of every cell of the sheet, for `Simulation.set_parameter`.

    The beta stimulus halves the K-ATP conductance on a region of cells, which depolarizes them
    enough to start a wave. So unlike `hipsc_stimulus_amplitude`, the value outside the region
    is the membrane model's own default and not zero. This still returns the whole array,
    because `set_parameter` writes what it is given.

    The region is the reference's: the two leftmost columns over five rows centred on the row a
    conduction velocity is measured along. On a sheet too small to hold it, it is clipped.

    Parameters
    ----------
    nx, ny : int, optional
        Shape of the sheet, by default 15 by 15.

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
    >>> conductance = presets.beta_stimulus_conductance(15, 15)
    >>> int((conductance == presets.BETA_STIMULUS_KATP_CONDUCTANCE).sum())
    10
    """
    return _stimulated_region(
        nx,
        ny,
        columns=_BETA_STIMULUS_COLUMNS,
        rows=_BETA_STIMULUS_ROWS,
        stimulated=BETA_STIMULUS_KATP_CONDUCTANCE,
        elsewhere=BETA_KATP_CONDUCTANCE,
    )


def beta_conduction_path(nx: int = 15, ny: int = 15) -> ConductionPath:
    """The two cells the paper measures a beta conduction velocity between, and their gap.

    Cells at columns 4 and 12 of the row halfway up the sheet, eight cell lengths apart. There
    is no anisotropy factor to pass: a beta cell has one size.

    Parameters
    ----------
    nx, ny : int, optional
        Shape of the sheet, by default 15 by 15.

    Returns
    -------
    ConductionPath
        Ready for `sknm.analysis.conduction_velocity`.

    Raises
    ------
    ValueError
        If the sheet is too narrow to hold both cells.

    Examples
    --------
    >>> from sknm import presets
    >>> path = presets.beta_conduction_path(15, 15)
    >>> path.start, path.end
    (94, 102)
    """
    return _conduction_path(nx, ny, columns=_BETA_CONDUCTION_COLUMNS, lx=BETA_CELL_SIZE)


def beta_centre_cell(nx: int = 15, ny: int = 15) -> int:
    """The beta cell at the centre of the sheet, whose trace the reference saves every step.

    The same cell as `hipsc_centre_cell` would give for a sheet of the same shape. Both
    reference drivers compute it with the same expression, so the two names share one formula.

    Parameters
    ----------
    nx, ny : int, optional
        Shape of the sheet, by default 15 by 15.

    Returns
    -------
    int
        Index of the cell.

    Examples
    --------
    >>> from sknm import presets
    >>> presets.beta_centre_cell(15, 15)
    97
    """
    return _centre_cell(nx, ny)
