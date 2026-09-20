"""The membrane seam: what a simulation requires of a membrane model.

`MembraneModel` is a `Protocol`, so a class satisfies it structurally -- without importing
anything from `sknm` -- and it is where the membrane potential's state name, the state layout
and the model's capacitance are bound to the model they belong to.

The contract in one line: **advance `(v, s)` by `dt`**. A simulation never sees `I_ion`; under
operator splitting the membrane model and the spatial operator exchange only `v`.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from sknm import units


@runtime_checkable
class MembraneModel(Protocol):
    """A system of ODEs governing one cell's membrane potential and internal state.

    Implementations are vectorised over cells and hold no per-simulation state: the caller owns
    the state and parameter arrays and passes them in on every step.
    """

    @property
    def v_name(self) -> str:
        """str: Name of the membrane potential state, for example ``"V_m"``."""
        ...

    @property
    def v_index(self) -> int:
        """int: Row of the membrane potential in a state array.

        ``states[v_index]`` is a contiguous, zero-copy row, which lets a spatial solve write
        straight back into the state array with no separate copy of `v` to keep in sync.
        """
        ...

    @property
    def num_states(self) -> int:
        """int: Number of state variables per cell."""
        ...

    @property
    def num_parameters(self) -> int:
        """int: Number of parameters per cell."""
        ...

    @property
    def capacitance(self) -> float | None:
        """float or None: Absolute membrane capacitance of one cell, in uF.

        Absolute rather than specific, because that is what a voltage equation of the form
        ``dv/dt = -I / Cm`` divides by. A network holds a *specific* capacitance and a membrane
        area per cell, and the two conventions meet at ``Cm * membrane_area``; comparing them
        is `Simulation`'s job.

        `None` when the model's voltage equation assumes a capacitance that it does not expose
        as a parameter -- ``dV/dt = -I_tot`` has one implicitly -- in which case a caller has
        nothing to check. A mismatch shows up as a plausible wave travelling at the wrong speed
        rather than as an obvious failure, which is why it is worth declaring where it can be.
        """
        ...

    def state_index(self, name: str) -> int:
        """Look up the row of a state by name.

        Parameters
        ----------
        name : str
            Name of the state variable.

        Returns
        -------
        int
            Row of `name` in a state array.

        Raises
        ------
        KeyError
            If `name` is not a state of this model.
        """
        ...

    def parameter_index(self, name: str) -> int:
        """Look up the row of a parameter by name.

        Parameters
        ----------
        name : str
            Name of the parameter.

        Returns
        -------
        int
            Row of `name` in a parameter array.

        Raises
        ------
        KeyError
            If `name` is not a parameter of this model.
        """
        ...

    def initial_states(self, n_cells: int) -> npt.NDArray[np.float64]:
        """Build an initial state array for `n_cells` identical cells.

        Parameters
        ----------
        n_cells : int
            Number of cells.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_states, n_cells)`` and dtype float64.
        """
        ...

    def initial_parameters(self, n_cells: int) -> npt.NDArray[np.float64]:
        """Build an initial parameter array for `n_cells` identical cells.

        Per-cell heterogeneity is expressed by writing into the returned array, which is how a
        stimulus applied to a subset of cells is set up.

        Parameters
        ----------
        n_cells : int
            Number of cells.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_parameters, n_cells)`` and dtype float64.
        """
        ...

    def step(
        self,
        states: npt.NDArray[np.float64],
        t: float,
        dt: float,
        parameters: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Advance `states` by `dt`.

        Functional rather than in-place: the returned array is new and `states` is left
        untouched, so the caller rebinds.

        Parameters
        ----------
        states : numpy.ndarray
            Current states, shape ``(num_states, n_cells)``.
        t : float
            Current time.
        dt : float
            Time step.
        parameters : numpy.ndarray
            Parameters, shape ``(num_parameters, n_cells)``.

        Returns
        -------
        numpy.ndarray
            New array of shape ``(num_states, n_cells)`` holding the advanced states.
        """
        ...


class GotranxModel:
    """Adapt a gotranx-generated numpy module to `MembraneModel`.

    The generated module is a collection of free functions with no notion of which state is the
    membrane potential or what the model's capacitance is; this binds those to it.

    Parameters
    ----------
    module : module
        An imported gotranx-generated numpy module.
    v_name : str, optional
        Name of the membrane potential state, by default ``"V_m"``.
    capacitance : pint.Quantity or None, optional
        Absolute membrane capacitance of one cell, such as ``5300 * units.fF``, or `None` when
        the model's voltage equation assumes a capacitance it does not expose. By default
        `None`.
    scheme : str, optional
        Name of the integration scheme function in `module`, by default
        ``"generalized_rush_larsen"``.

    Raises
    ------
    ValueError
        If `module` has no function named `scheme`.
    KeyError
        If `module` has no state named `v_name`.
    TypeError
        If `capacitance` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `capacitance` does not measure a capacitance. A specific capacitance is the
        plausible wrong answer, and it is refused rather than silently rescaled.
    """

    def __init__(
        self,
        module: ModuleType,
        *,
        v_name: str = "V_m",
        capacitance: Any = None,
        scheme: str = "generalized_rush_larsen",
    ) -> None:
        self._module = module
        self._v_name = v_name
        self._capacitance: float | None = (
            None
            if capacitance is None
            else float(units.in_base_units(capacitance, "capacitance", name="capacitance"))
        )
        try:
            self._step = getattr(module, scheme)
        except AttributeError as exc:
            raise ValueError(
                f"{module.__name__} has no scheme {scheme!r}. Regenerate it with "
                f"`python3 tools/generate_membrane_models.py`."
            ) from exc
        # Resolve here rather than at the first step, so a misnamed state fails at construction.
        self._v_index = int(module.state_index(v_name))
        self._initial_states: npt.NDArray[np.float64] = np.asarray(
            module.init_state_values(), dtype=np.float64
        )
        self._initial_parameters: npt.NDArray[np.float64] = np.asarray(
            module.init_parameter_values(), dtype=np.float64
        )

    @property
    def v_name(self) -> str:
        """str: Name of the membrane potential state."""
        return self._v_name

    @property
    def v_index(self) -> int:
        """int: Row of the membrane potential in a state array."""
        return self._v_index

    @property
    def num_states(self) -> int:
        """int: Number of state variables per cell."""
        return int(self._initial_states.size)

    @property
    def num_parameters(self) -> int:
        """int: Number of parameters per cell."""
        return int(self._initial_parameters.size)

    @property
    def capacitance(self) -> float | None:
        """float or None: Absolute capacitance in uF, or `None` when the model exposes none."""
        return self._capacitance

    def state_index(self, name: str) -> int:
        """Look up the row of a state by name.

        Parameters
        ----------
        name : str
            Name of the state variable.

        Returns
        -------
        int
            Row of `name` in a state array.
        """
        return int(self._module.state_index(name))

    def parameter_index(self, name: str) -> int:
        """Look up the row of a parameter by name.

        Parameters
        ----------
        name : str
            Name of the parameter.

        Returns
        -------
        int
            Row of `name` in a parameter array.
        """
        return int(self._module.parameter_index(name))

    def initial_states(self, n_cells: int) -> npt.NDArray[np.float64]:
        """Build an initial state array for `n_cells` identical cells.

        Parameters
        ----------
        n_cells : int
            Number of cells.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_states, n_cells)``.
        """
        return np.tile(self._initial_states[:, None], (1, n_cells))

    def initial_parameters(self, n_cells: int) -> npt.NDArray[np.float64]:
        """Build an initial parameter array for `n_cells` identical cells.

        Parameters
        ----------
        n_cells : int
            Number of cells.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(num_parameters, n_cells)``.
        """
        return np.tile(self._initial_parameters[:, None], (1, n_cells))

    def step(
        self,
        states: npt.NDArray[np.float64],
        t: float,
        dt: float,
        parameters: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Advance `states` by `dt` using the generated scheme.

        Parameters
        ----------
        states : numpy.ndarray
            Current states, shape ``(num_states, n_cells)``.
        t : float
            Current time.
        dt : float
            Time step.
        parameters : numpy.ndarray
            Parameters, shape ``(num_parameters, n_cells)``.

        Returns
        -------
        numpy.ndarray
            New array of shape ``(num_states, n_cells)``.
        """
        # gotranx's `rhs` takes (t, states, parameters) but its schemes take
        # (states, t, dt, parameters); the seam presents the scheme order.
        return np.asarray(self._step(states, t, dt, parameters), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self._module.__name__}, v_name={self._v_name!r}, "
            f"num_states={self.num_states}, num_parameters={self.num_parameters})"
        )


def from_gotranx(
    module: ModuleType,
    *,
    v_name: str = "V_m",
    capacitance: Any = None,
    scheme: str = "generalized_rush_larsen",
) -> MembraneModel:
    """Wrap a gotranx-generated numpy module as a `MembraneModel`.

    Takes an imported module rather than a path to an ``.ode`` file, so that gotranx is not
    needed at run time. To use your own ``.ode``, run ``tools/generate_membrane_models.py`` over
    it, import the result, and pass it here.

    Parameters
    ----------
    module : module
        An imported gotranx-generated numpy module.
    v_name : str, optional
        Name of the membrane potential state, by default ``"V_m"``.
    capacitance : pint.Quantity or None, optional
        Absolute membrane capacitance of one cell, such as ``5300 * units.fF``, by default
        `None`. Declare it whenever the model's voltage equation divides by one: it is what
        lets `sknm.Simulation` reject a network whose capacitance is not the model's.
    scheme : str, optional
        Name of the integration scheme function in `module`, by default
        ``"generalized_rush_larsen"``.

    Returns
    -------
    MembraneModel
        A model delegating to `module`.

    Raises
    ------
    ValueError
        If `module` has no function named `scheme`.
    KeyError
        If `module` has no state named `v_name`.
    TypeError
        If `capacitance` is a bare number rather than a quantity.
    pint.DimensionalityError
        If `capacitance` does not measure a capacitance.

    Examples
    --------
    >>> from sknm.membrane import base_model_IM, from_gotranx
    >>> model = from_gotranx(base_model_IM)
    >>> model.num_states
    25
    """
    return GotranxModel(module, v_name=v_name, capacitance=capacitance, scheme=scheme)
