"""Measuring a travelling wave: when each cell activated, and how fast it rose.

The two numbers the paper reports for a network of cells -- the conduction velocity and the
maximal upstroke velocity -- are both read off the membrane potential at *every* time step.
Sampling a `Result` cannot supply them: a trace recorded every millisecond resolves neither a
threshold crossing to better than a millisecond nor an upstroke that lasts about one. So they
are gathered by a callback instead, `ActivationRecorder`, which `Simulation.run` calls after
each step::

    path = presets.hipsc_conduction_path(40, 40)
    recorder = ActivationRecorder(sim, threshold=-20 * mV, stop_when_activated=path.end)
    sim.run(50 * ms, record=(), callback=recorder)
    velocity = conduction_velocity(recorder, path)

The recorder also ends the run as soon as the wave has arrived, which is most of the saving: a
conduction velocity is known long before the 50 ms a full action potential takes.

Its arrays are bare magnitudes in the package's base units, like `Result.t` and `Result.v`.
`conduction_velocity` is the exception that returns a quantity: it is a single number handed
back to a reader, who may not be thinking in centimetres per second.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from sknm import units

if TYPE_CHECKING:  # pragma: no cover - imported for typing only, and would be circular
    from sknm.simulation import Simulation


@dataclass(frozen=True)
class ConductionPath:
    """Two cells and the distance between them, along which a velocity is measured.

    The three travel together because they are only meaningful together. A cell index pairs
    with a distance that depends on the cell size, so a path built for one anisotropy factor
    and used with a network built for another gives a conduction velocity wrong by as much as a
    factor of two, with nothing else to show for it. `sknm.presets` hands out all three at once.

    Parameters
    ----------
    start, end : int
        Indices of the cell the wave leaves and the cell it arrives at. The path is oriented:
        `end` is expected to activate after `start`.
    distance : pint.Quantity
        Distance between the two cells, as a length. Along the path the wave actually travels,
        which for a straight run of cells is the number of cells between them times their size.

    Attributes
    ----------
    start, end : int
        As given.
    distance : float
        `distance` in cm.

    Raises
    ------
    TypeError
        If `distance` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `distance` does not measure a length.
    ValueError
        If `distance` is not positive and finite, or if `start` and `end` are the same cell.

    Examples
    --------
    >>> from sknm.analysis import ConductionPath
    >>> from sknm.units import um
    >>> round(ConductionPath(start=769, end=794, distance=25 * 16 * um).distance, 6)
    0.04
    """

    start: int
    end: int
    distance: Any

    def __post_init__(self) -> None:
        distance = float(units.in_base_units(self.distance, "length", name="distance"))
        if not np.isfinite(distance) or distance <= 0.0:
            raise ValueError(f"distance must be positive and finite, got {distance} cm")
        object.__setattr__(self, "distance", distance)

        start, end = int(self.start), int(self.end)
        if start == end:
            raise ValueError(f"start and end must be different cells, both are {start}")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


class ActivationRecorder:
    """Watch a simulation step by step, recording when each cell activated and how fast it rose.

    A `Simulation.run` callback. It is built from the simulation it will watch rather than from
    a cell count, because the centred difference of the supplementary's equation (S8) needs the
    membrane potential from *before* the first step, and only the simulation has it.

    Every cell is watched. The arrays are one float per cell, and the two quantities are read
    off different cells anyway -- the reference measures its velocity between two cells far
    apart and its upstroke at the centre of the network.

    Parameters
    ----------
    simulation : Simulation
        The simulation to watch. Its current membrane potential is taken as the starting point,
        so build the recorder after setting the initial condition.
    threshold : pint.Quantity
        Membrane potential at which a cell counts as activated, as a potential. Required, and
        deliberately without a default: the value belongs to the membrane model rather than to
        the measurement, and one model's is another's catastrophe. `sknm.presets` carries
        `HIPSC_THRESHOLD` and `BETA_THRESHOLD`, which are -20 mV and -50 mV -- and a beta cell
        action potential peaks at about -19.5 mV, so the cardiac threshold applied to it leaves
        every measurement cell unactivated.
    stop_when_activated : int, sequence of int or None, optional
        Cells whose activation ends the run. By default `None`, meaning run to `t_end`. Naming
        the far end of a conduction path stops the run as soon as the wave has arrived, which
        is well before a full action potential has finished.

    Attributes
    ----------
    threshold : float
        `threshold` in mV.

    Raises
    ------
    TypeError
        If `threshold` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `threshold` does not measure a potential.
    IndexError
        If `stop_when_activated` names a cell that does not exist.
    """

    def __init__(
        self,
        simulation: Simulation,
        *,
        threshold: Any,
        stop_when_activated: npt.ArrayLike | None = None,
    ) -> None:
        self._threshold = float(units.in_base_units(threshold, "potential", name="threshold"))
        n_cells = int(simulation.v.size)

        if stop_when_activated is None:
            self._stop_cells: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
        else:
            self._stop_cells = np.atleast_1d(np.asarray(stop_when_activated, dtype=np.int64))
            out_of_range = self._stop_cells[
                (self._stop_cells < -n_cells) | (self._stop_cells >= n_cells)
            ]
            if out_of_range.size:
                raise IndexError(
                    f"stop_when_activated names cell {out_of_range[0]}, but the network has "
                    f"{n_cells} cells"
                )

        self._activation_time = np.full(n_cells, np.nan)
        self._max_upstroke_velocity = np.full(n_cells, np.nan)
        # Two steps of history, because (S8) differences across the step either side of a
        # sample rather than across the step just taken.
        self._previous = simulation.v.copy()
        self._before_that = simulation.v.copy()

    def __repr__(self) -> str:
        activated = int(np.count_nonzero(~np.isnan(self._activation_time)))
        return (
            f"{type(self).__name__}(threshold={self._threshold} mV, "
            f"activated={activated}/{self._activation_time.size})"
        )

    @property
    def threshold(self) -> float:
        """float: Membrane potential at which a cell counts as activated, in mV."""
        return self._threshold

    @property
    def activation_time(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: When each cell first reached the threshold, in ms.

        Shape ``(n_cells,)``, and `nan` for a cell that has not reached it. A read-only view of
        the live array, so it keeps up with a running simulation.
        """
        return _read_only(self._activation_time)

    @property
    def max_upstroke_velocity(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: Greatest rate of rise seen at each cell, in mV/ms.

        Shape ``(n_cells,)``, and `nan` until the simulation has taken a step. Millivolts per
        millisecond is volts per second, the unit the paper reports.

        The rate is the centred difference of the supplementary's equation (S8),
        ``(v[n+1] - v[n-1]) / (2*dt)``, and it is taken on the membrane potential as it stands
        at the end of a step, after the spatial solve.
        """
        return _read_only(self._max_upstroke_velocity)

    def __call__(self, simulation: Simulation) -> bool:
        """Record one step, and say whether the run should stop.

        Parameters
        ----------
        simulation : Simulation
            The simulation, just after a step.

        Returns
        -------
        bool
            True once every cell named by `stop_when_activated` has activated.
        """
        potential = simulation.v

        newly_activated = np.isnan(self._activation_time) & (potential >= self._threshold)
        self._activation_time[newly_activated] = simulation.t

        rate = (potential - self._before_that) / (2.0 * simulation.dt)
        np.fmax(self._max_upstroke_velocity, rate, out=self._max_upstroke_velocity)

        self._before_that = self._previous
        self._previous = potential.copy()

        if self._stop_cells.size == 0:
            return False
        return not np.isnan(self._activation_time[self._stop_cells]).any()


def _read_only(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """A view that cannot be written through, so that a caller cannot rewrite what was measured."""
    view = values.view()
    view.flags.writeable = False
    return view


def conduction_velocity(activation: ActivationRecorder, path: ConductionPath) -> Any:
    """How fast the wave travelled along a path, from the two cells' activation times.

    Parameters
    ----------
    activation : ActivationRecorder
        A recorder that has watched a run.
    path : ConductionPath
        The two cells and the distance between them.

    Returns
    -------
    pint.Quantity
        The velocity, in cm/s. A quantity rather than a bare number: it is a result to be read
        rather than state to be stepped, and the unit it is reported in varies by author.

    Raises
    ------
    IndexError
        If `path` names a cell the recorder has not been watching.
    ValueError
        If either cell never reached the threshold, or if `path.end` activated no later than
        `path.start`.

    Examples
    --------
    >>> import numpy as np
    >>> from sknm.analysis import ConductionPath, conduction_velocity
    >>> from sknm.units import um
    >>> class Measured:  # stands in for a recorder that has watched a run
    ...     activation_time = np.array([26.2, 36.9])
    ...     threshold = -20.0
    >>> velocity = conduction_velocity(Measured(), ConductionPath(0, 1, distance=400 * um))
    >>> float(round(velocity.m_as("cm / s"), 3))
    3.738
    """
    times = activation.activation_time
    start, end = times[path.start], times[path.end]
    for cell, time in ((path.start, start), (path.end, end)):
        if np.isnan(time):
            raise ValueError(
                f"cell {cell} did not reach the threshold of {activation.threshold} mV during "
                f"the run, so there is no activation time to measure a velocity from"
            )
    if end <= start:
        raise ValueError(
            f"cell {path.end} activated before cell {path.start} did, at {end} ms against "
            f"{start} ms. A conduction path runs from start to end; check that the path is the "
            f"right way round and that the stimulus is at the start of it"
        )
    # Times are in ms and the distance in cm, so the ratio is cm/ms; the base velocity unit is
    # cm/s, which is what the paper and the reference implementation both report.
    return units.with_base_units(path.distance / (end - start) * 1e3, "velocity")
