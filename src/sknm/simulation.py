"""The time stepper: a network, a membrane model and a variant, advanced through time.

`Simulation` owns everything that changes as a run proceeds -- the state array, the parameter
array and the clock -- and holds the two things that do not: the assembled `Operator` and the
factorization of its matrix. The network is frozen and the conductances do not vary with time,
so the matrix is factorized once, on the first step, and reused for the rest of the run.

One step is Godunov splitting, which is what the reference implementation does and what the
paper's convergence study justifies its time step against:

1. advance every cell's membrane model by `dt`, independently of its neighbours;
2. solve the variant's linear system for the membrane potential, coupling them.

There is no separate array for the membrane potential. It is a row of the state array, so the
spatial solve writes back into the states that the next membrane step reads, and the two cannot
fall out of step with one another.

A stimulus is not a concept here. The paper's protocols are a membrane parameter raised on some
cells -- an injected current for the cardiac model, a halved conductance for the beta cell --
with all of the timing inside the membrane model's own equations, so `set_parameter` is the
whole mechanism and the numerical core never learns that a stimulus exists.

**Times carry units**: ``dt=0.02 * ms``, ``run(50 * ms, record_every=1 * ms)``. They are
converted once, here, and everything downstream is in milliseconds. What comes back out --
`Simulation.t`, `Result.t` -- is a bare magnitude in milliseconds, as everywhere else in the
package.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, get_args

import numpy as np
import numpy.typing as npt

from sknm import units
from sknm.assembly import Operator, Variant, assemble
from sknm.linalg import Factorization, Solver, as_solver
from sknm.membrane import MembraneModel
from sknm.network import CellNetwork

Splitting = Literal["godunov", "strang"]

#: Recording name for the membrane potential, whatever the membrane model calls its own state.
#: `record=("v",)` is the default, so it has to mean the same thing for every model.
MEMBRANE_POTENTIAL = "v"

#: Recording name for the extracellular potential, which is not a membrane state.
EXTRACELLULAR_POTENTIAL = "u_e"

#: Marks a recorded quantity that comes from the solution vector rather than from a state row.
#: Negative, so it cannot collide with a row index.
_EXTRACELLULAR_ROW = -1

#: Largest relative disagreement between a model's declared capacitance and the network's that
#: is taken for the same number rather than a mismatch. Wide enough for the rounding in a
#: printed value, far too narrow for the factor-of-ten mistakes the check exists to catch.
CAPACITANCE_TOLERANCE = 1e-9


@dataclass(frozen=True, eq=False)
class Result:
    """The traces a run recorded.

    Built by `Simulation.run`, not directly.

    Attributes
    ----------
    t : numpy.ndarray
        Time of each sample in ms, shape ``(n_samples,)``, read-only.
    traces : mapping of str to numpy.ndarray
        Each recorded quantity, of shape ``(n_cells, n_samples)`` and read-only. Keyed by the
        name it was asked for: ``"v"`` for the membrane potential, ``"u_e"`` for the
        extracellular potential, or the membrane model's own name for any other state.
    variant : Variant
        The model that was run.
    """

    t: npt.NDArray[np.float64]
    traces: Mapping[str, npt.NDArray[np.float64]]
    variant: Variant

    def __post_init__(self) -> None:
        self.t.flags.writeable = False
        for trace in self.traces.values():
            trace.flags.writeable = False
        # A frozen dataclass stops the mapping being rebound but not its contents being
        # rewritten, and a proxy is the only read-only mapping the standard library has.
        object.__setattr__(self, "traces", MappingProxyType(dict(self.traces)))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(variant={self.variant.name}, n_cells={self.n_cells}, "
            f"n_samples={self.n_samples}, recorded={sorted(self.traces)})"
        )

    @property
    def n_samples(self) -> int:
        """int: Number of samples recorded."""
        return int(self.t.size)

    @property
    def n_cells(self) -> int:
        """int: Number of cells each trace covers, or zero if nothing was recorded."""
        for trace in self.traces.values():
            return int(trace.shape[0])
        return 0

    @property
    def names(self) -> tuple[str, ...]:
        """tuple of str: What was recorded, in the order it was asked for."""
        return tuple(self.traces)

    @property
    def v(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: The membrane potential, shape ``(n_cells, n_samples)``.

        Raises
        ------
        KeyError
            If the membrane potential was not among the recorded quantities.
        """
        return self[MEMBRANE_POTENTIAL]

    @property
    def u_e(self) -> npt.NDArray[np.float64] | None:
        """numpy.ndarray or None: The extracellular potential, shape ``(n_cells, n_samples)``.

        `None` if it was not recorded, and `None` for `Variant.SKNM`, which eliminates the
        extracellular potential rather than solving for it and so has none to report.

        The potential is produced by the spatial solve, so the sample taken before the first
        step of a run has none and is `nan`.
        """
        return self.traces.get(EXTRACELLULAR_POTENTIAL)

    def __getitem__(self, name: str) -> npt.NDArray[np.float64]:
        """Read one trace by the name it was recorded under.

        Parameters
        ----------
        name : str
            A recorded name.

        Returns
        -------
        numpy.ndarray
            The trace, shape ``(n_cells, n_samples)``, read-only.

        Raises
        ------
        KeyError
            If `name` was not recorded.
        """
        try:
            return self.traces[name]
        except KeyError:
            recorded = ", ".join(repr(known) for known in self.traces) or "nothing"
            raise KeyError(f"{name!r} was not recorded; this result holds {recorded}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.traces

    def __iter__(self) -> Iterator[str]:
        return iter(self.traces)


class Simulation:
    """A network and a membrane model, advanced through time by one of the three variants.

    Parameters
    ----------
    network : CellNetwork
        The cells, their connections and the conductances between them.
    model : MembraneModel
        The membrane model every cell runs. One model per simulation; heterogeneity between
        cells is expressed through per-cell parameters, not through a second model.
    variant : Variant or str, optional
        Which of the three models to solve, by default `Variant.SKNM`. A string naming a
        `Variant` value is accepted.
    dt : pint.Quantity, optional
        Length of a time step, as a time. By default ``0.02 * ms``, the reference
        implementation's own step and the one the paper's convergence study settles on. The
        membrane model takes one step of the same length, which that study also assumes.
    solver : Solver or str, optional
        How the linear system is solved, by default `DirectSolver`. A string naming one is
        accepted; see `sknm.linalg.as_solver`.
    ground : array_like or None, optional
        Cells whose extracellular potential is pinned to zero, one per connected component. By
        default `None`, meaning the lowest-index cell of each. Only `Variant.KNM` has an
        extracellular unknown, so the other two ignore it.
    splitting : {"godunov"}, optional
        Operator splitting scheme, by default ``"godunov"``.
    check_capacitance : bool, optional
        Whether to reject a membrane model whose declared capacitance disagrees with the
        network's, by default `True`.

    Attributes
    ----------
    network : CellNetwork
        The network, read-only.
    model : MembraneModel
        The membrane model, read-only.

    Raises
    ------
    TypeError
        If `dt` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `dt` does not measure a time.
    ValueError
        If `variant`, `solver` or `splitting` names nothing known, if `dt` is not positive and
        finite, if `ground` is not one cell per connected component, or if the model's declared
        capacitance disagrees with the network's.
    NotImplementedError
        If `splitting` is ``"strang"``.

    Examples
    --------
    >>> from sknm import Simulation, chain
    >>> from sknm.membrane import fitzhugh_nagumo
    >>> from sknm.units import cm, mS, ms, uS, um
    >>> network = chain(
    ...     4, lx=16 * um, ly=16 * um, lz=19.2 * um, delta_e=0.2,
    ...     sigma_i=4.0 * mS / cm, sigma_e=20.0 * mS / cm, Gg=0.2 * uS,
    ... )
    >>> simulation = Simulation(network, fitzhugh_nagumo(), dt=0.01 * ms)
    >>> simulation.set_parameter("stim_amplitude", 1.0, cells=[0])
    >>> result = simulation.run(5 * ms, record_every=1 * ms)
    >>> result.v.shape
    (4, 6)

    The far end of the strand is depolarized by the end of the run:

    >>> bool(result.v[-1, -1] > result.v[-1, 0])
    True
    """

    def __init__(
        self,
        network: CellNetwork,
        model: MembraneModel,
        *,
        variant: Variant | str = Variant.SKNM,
        dt: Any = 0.02 * units.ms,
        solver: Solver | str = "direct",
        ground: npt.ArrayLike | None = None,
        splitting: Splitting = "godunov",
        check_capacitance: bool = True,
    ) -> None:
        _check_splitting(splitting)
        if check_capacitance:
            _check_capacitance(network, model)

        self._network = network
        self._model = model
        self._solver = as_solver(solver)
        self._dt = float(units.in_base_units(dt, "time", name="dt"))
        self._operator = assemble(network, dt=dt, variant=variant, ground=ground)

        self._v_index = int(model.v_index)
        self._states: npt.NDArray[np.float64] = np.array(
            model.initial_states(network.n_cells), dtype=np.float64
        )
        self._parameters: npt.NDArray[np.float64] = np.array(
            model.initial_parameters(network.n_cells), dtype=np.float64
        )
        self._n_steps = 0
        # Filled in on the first step. The factorization is the expensive part of a solve and
        # is deferred so that building a sweep of simulations costs nothing until one is run.
        self._solve: Factorization | None = None
        self._solution: npt.NDArray[np.float64] | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(variant={self.variant.name}, "
            f"n_cells={self._network.n_cells}, dt={self._dt:g} ms, t={self.t:g} ms)"
        )

    # --- What it is ------------------------------------------------------------------------

    @property
    def network(self) -> CellNetwork:
        """CellNetwork: The network being simulated."""
        return self._network

    @property
    def model(self) -> MembraneModel:
        """MembraneModel: The membrane model every cell runs."""
        return self._model

    @property
    def operator(self) -> Operator:
        """Operator: The assembled system one step solves."""
        return self._operator

    @property
    def variant(self) -> Variant:
        """Variant: Which of the three models this simulation solves."""
        return self._operator.variant

    @property
    def solver(self) -> Solver:
        """Solver: How the linear system is solved."""
        return self._solver

    @property
    def dt(self) -> float:
        """float: Length of a time step in ms.

        Read-only: the operator was assembled around this step and would not be reassembled.
        """
        return self._dt

    # --- What changes ----------------------------------------------------------------------

    @property
    def t(self) -> float:
        """float: Current time in ms.

        Counted in whole steps rather than accumulated, so that a long run's clock does not
        drift away from the step it is on.
        """
        return self._n_steps * self._dt

    @property
    def states(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: The state of every cell, shape ``(num_states, n_cells)``.

        Writable, and rebound on every step, so hold the `Simulation` rather than the array.
        """
        return self._states

    @property
    def parameters(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: Membrane parameters, shape ``(num_parameters, n_cells)``.

        Writable. `set_parameter` is the same thing by name; this is the escape hatch for
        anything it does not express.
        """
        return self._parameters

    @property
    def v(self) -> npt.NDArray[np.float64]:
        """numpy.ndarray: Membrane potential of every cell, shape ``(n_cells,)``.

        A view into `states`, not a copy, so there is only one of it.
        """
        return self._states[self._v_index]

    @property
    def extracellular_potential(self) -> npt.NDArray[np.float64] | None:
        """numpy.ndarray or None: Extracellular potential, shape ``(n_cells,)``.

        `None` before the first step, which is what produces it, and `None` for `Variant.SKNM`,
        which eliminates it rather than solving for it. Zeros for `Variant.SKNM_UE0`, which
        defines it to be zero.
        """
        if self._solution is None:
            return None
        return self._operator.extracellular_potential(self._solution)

    def set_parameter(
        self, name: str, value: npt.ArrayLike, cells: npt.ArrayLike | None = None
    ) -> None:
        """Set a membrane parameter, on every cell or on some of them.

        This is how the paper's stimulus protocols are expressed: an amplitude raised on the
        cells being stimulated, with all of the timing inside the membrane model's equations.
        Membrane parameters do not enter the linear system, so setting one never costs a
        refactorization.

        `value` carries no unit. It is in whatever convention the membrane model uses for that
        parameter, which the model knows and this package does not.

        Parameters
        ----------
        name : str
            Name of a parameter of the membrane model.
        value : array_like
            The new value: a scalar for every selected cell, or one value per selected cell.
        cells : array_like or None, optional
            Which cells to set it on, as indices or as a boolean mask over all cells. By
            default `None`, meaning every cell.

        Raises
        ------
        KeyError
            If `name` is not a parameter of the membrane model.
        IndexError
            If `cells` selects a cell that does not exist.
        ValueError
            If `value` has neither one entry nor one entry per selected cell.
        """
        row = self._model.parameter_index(name)
        if cells is None:
            self._parameters[row] = value
        else:
            self._parameters[row, np.asarray(cells)] = value

    # --- Running ---------------------------------------------------------------------------

    def step(self) -> None:
        """Advance the simulation by one time step.

        Godunov splitting: the membrane model advances every cell independently over the step,
        and the resulting membrane potential is then the right-hand side of the variant's
        linear system, whose solution replaces it.

        Raises
        ------
        sknm.linalg.ConvergenceError
            If an iterative solver cannot reach its tolerance.
        """
        if self._solve is None:
            self._solve = self._solver.factorize(self._operator.matrix)

        states = self._model.step(self._states, self.t, self._dt, self._parameters)
        solution = self._solve(self._operator.rhs(states[self._v_index]))
        states[self._v_index] = self._operator.membrane_potential(solution)

        self._states = states
        self._solution = solution
        self._n_steps += 1

    def run(
        self,
        t_end: Any,
        *,
        record_every: Any = 1.0 * units.ms,
        record: Sequence[str] = (MEMBRANE_POTENTIAL,),
        callback: Callable[[Simulation], bool | None] | None = None,
    ) -> Result:
        """Advance the simulation to `t_end`, recording as it goes.

        The first sample is the state before any step is taken, so a trace starts at the
        initial condition. Sampling is then every whole number of steps nearest `record_every`.

        Parameters
        ----------
        t_end : pint.Quantity
            Time to run to, as a time. Not a duration: it is measured from the start of the
            simulation, so a second `run` continues from where the first stopped.
        record_every : pint.Quantity, optional
            Interval between samples, as a time. By default ``1 * ms``. Rounded to the nearest
            whole number of steps, and at least one.
        record : sequence of str, optional
            What to record, by default ``("v",)``. ``"v"`` is the membrane potential whatever
            the membrane model calls it, ``"u_e"`` the extracellular potential, and any other
            name must be a state of the membrane model.
        callback : callable or None, optional
            Called with this `Simulation` after every step. Returning a true value stops the
            run there. By default `None`. This is how a run ends as soon as it has measured
            what it was started for, rather than at `t_end`.

        Returns
        -------
        Result
            The recorded traces.

        Raises
        ------
        TypeError
            If `t_end` or `record_every` is a bare number rather than a quantity.
        pint.DimensionalityError
            If either does not measure a time.
        ValueError
            If `t_end` is before the current time, if `record_every` is not positive, or if
            `record` names something that cannot be recorded.
        sknm.linalg.ConvergenceError
            If an iterative solver cannot reach its tolerance.
        """
        end = float(units.in_base_units(t_end, "time", name="t_end"))
        interval = float(units.in_base_units(record_every, "time", name="record_every"))
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError(f"record_every must be positive and finite, got {interval} ms")
        n_steps = round((end - self.t) / self._dt)
        if n_steps < 0:
            raise ValueError(
                f"t_end is {end} ms, which is before the simulation's current time of "
                f"{self.t} ms; t_end is measured from the start of the simulation"
            )
        every = max(1, round(interval / self._dt))

        rows = self._recording_rows(record)
        times: list[float] = []
        samples: dict[str, list[npt.NDArray[np.float64]]] = {name: [] for name in rows}
        self._sample(times, samples, rows)

        for index in range(1, n_steps + 1):
            self.step()
            if index % every == 0:
                self._sample(times, samples, rows)
            if callback is not None and callback(self):
                break

        traces = {
            name: np.stack(columns, axis=1)
            if columns
            else np.empty((self._network.n_cells, 0), dtype=np.float64)
            for name, columns in samples.items()
        }
        return Result(t=np.array(times, dtype=np.float64), traces=traces, variant=self.variant)

    def _recording_rows(self, record: Sequence[str]) -> dict[str, int]:
        """Resolve each recording name to a row of the state array, or to the extracellular one.

        `Variant.SKNM` has no extracellular potential to give, so asking for one records
        nothing rather than raising: it is a property of the model that was chosen, not a
        mistake in the request, and the same run makes sense under the other two variants.
        """
        rows: dict[str, int] = {}
        for name in record:
            if name == MEMBRANE_POTENTIAL:
                rows[name] = self._v_index
            elif name == EXTRACELLULAR_POTENTIAL:
                if self.variant is not Variant.SKNM:
                    rows[name] = _EXTRACELLULAR_ROW
            else:
                try:
                    rows[name] = int(self._model.state_index(name))
                except KeyError:
                    raise ValueError(
                        f"{name!r} cannot be recorded: it is neither "
                        f"{MEMBRANE_POTENTIAL!r}, nor {EXTRACELLULAR_POTENTIAL!r}, nor a "
                        f"state of {type(self._model).__name__}"
                    ) from None
        return rows

    def _sample(
        self,
        times: list[float],
        samples: dict[str, list[npt.NDArray[np.float64]]],
        rows: Mapping[str, int],
    ) -> None:
        """Take one sample of every recorded quantity at the current time."""
        times.append(self.t)
        for name, row in rows.items():
            if row == _EXTRACELLULAR_ROW:
                extracellular = self.extracellular_potential
                samples[name].append(
                    np.full(self._network.n_cells, np.nan)
                    if extracellular is None
                    else extracellular
                )
            else:
                samples[name].append(self._states[row].copy())


def _check_splitting(splitting: Splitting) -> None:
    """Accept Godunov splitting and explain why the alternative is not offered."""
    if splitting == "godunov":
        return
    if splitting == "strang":
        raise NotImplementedError(
            "only Godunov splitting is implemented. Strang splitting would halve the membrane "
            "step and take two of them, doubling the dominant cost, and it would invalidate "
            "the convergence study of Tables S1 and S4, which measures the accuracy of a time "
            "step under Godunov splitting with a Rush-Larsen membrane step."
        )
    raise ValueError(f"splitting must be one of {get_args(Splitting)}, got {splitting!r}")


def _check_capacitance(network: CellNetwork, model: MembraneModel) -> None:
    """Reject a membrane model whose capacitance is not the network's.

    Raised rather than warned. A mismatch does not break a run: it produces a perfectly
    plausible wave travelling at the wrong speed, and a warning about it would scroll past
    unseen in a sweep of dozens of simulations.
    """
    declared = model.capacitance
    if declared is None:
        # The model's voltage equation assumes a capacitance it does not expose, so there is
        # nothing to compare against.
        return
    if abs(declared - network.Cm) > CAPACITANCE_TOLERANCE * abs(network.Cm):
        raise ValueError(
            f"the membrane model's capacitance of {declared:g} uF/cm^2 is not the network's "
            f"{network.Cm:g} uF/cm^2. A mismatch shows up as a plausible wave at the wrong "
            f"speed rather than as a failure. Build the network with `Cm={declared:g} * uF / "
            f"cm ** 2`, or pass `check_capacitance=False` if the difference is intended."
        )
