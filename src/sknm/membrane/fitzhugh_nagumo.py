"""FitzHugh-Nagumo, a two-state excitable membrane model.

Cheap enough to use wherever a membrane model is needed but its physiological detail is not:
exercising the seam, network tests with analytic answers, and demonstrations. It carries no
units and is not a cardiac model; use `sknm.membrane.base_model_IM` for anything physiological.

.. math::

    \\frac{dv}{dt} = v - \\frac{v^3}{3} - w + I

    \\frac{dw}{dt} = \\varepsilon (v + a - b w)

where :math:`I` is the ``stim_amplitude`` parameter.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from sknm.membrane.protocol import MembraneModel

_STATES = ("v", "w")
_PARAMETERS = ("a", "b", "eps", "stim_amplitude")

_DEFAULT_PARAMETERS = (0.7, 0.8, 0.08, 0.0)

# Fixed point of the parameter defaults above: the root of v - v**3/3 - (v + a)/b, with
# w = (v + a)/b. Both derivatives evaluate to exactly 0.0 in float64 here, so an unstimulated
# cell holds its resting state instead of drifting.
_RESTING_V = -1.199408035244035
_RESTING_W = -0.6242600440550438


class FitzHughNagumo:
    """A two-state excitable membrane model satisfying `MembraneModel`.

    Attributes
    ----------
    v_name : str
        ``"v"``.
    v_index : int
        ``0``.
    num_states : int
        ``2``, for ``v`` and ``w``.
    num_parameters : int
        ``4``, for ``a``, ``b``, ``eps`` and ``stim_amplitude``.
    capacitance : None
        The model is dimensionless and has no capacitance to declare.

    Examples
    --------
    >>> from sknm.membrane import fitzhugh_nagumo
    >>> model = fitzhugh_nagumo()
    >>> model.initial_states(4).shape
    (2, 4)
    """

    def __init__(self) -> None:
        self._states = np.array([_RESTING_V, _RESTING_W], dtype=np.float64)
        self._parameters = np.array(_DEFAULT_PARAMETERS, dtype=np.float64)

    @property
    def v_name(self) -> str:
        """str: Name of the membrane potential state."""
        return "v"

    @property
    def v_index(self) -> int:
        """int: Row of the membrane potential in a state array."""
        return 0

    @property
    def num_states(self) -> int:
        """int: Number of state variables per cell."""
        return len(_STATES)

    @property
    def num_parameters(self) -> int:
        """int: Number of parameters per cell."""
        return len(_PARAMETERS)

    @property
    def capacitance(self) -> float | None:
        """None: the model is dimensionless and has no capacitance to declare."""
        return None

    def state_index(self, name: str) -> int:
        """Look up the row of a state by name.

        Parameters
        ----------
        name : str
            One of ``"v"`` or ``"w"``.

        Returns
        -------
        int
            Row of `name` in a state array.

        Raises
        ------
        KeyError
            If `name` is not a state of this model.
        """
        try:
            return _STATES.index(name)
        except ValueError:
            raise KeyError(f"unknown state {name!r}; known states are {_STATES}") from None

    def parameter_index(self, name: str) -> int:
        """Look up the row of a parameter by name.

        Parameters
        ----------
        name : str
            One of ``"a"``, ``"b"``, ``"eps"`` or ``"stim_amplitude"``.

        Returns
        -------
        int
            Row of `name` in a parameter array.

        Raises
        ------
        KeyError
            If `name` is not a parameter of this model.
        """
        try:
            return _PARAMETERS.index(name)
        except ValueError:
            raise KeyError(
                f"unknown parameter {name!r}; known parameters are {_PARAMETERS}"
            ) from None

    def initial_states(self, n_cells: int) -> npt.NDArray[np.float64]:
        """Build an initial state array of `n_cells` cells at rest.

        Parameters
        ----------
        n_cells : int
            Number of cells.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(2, n_cells)`` holding the resting fixed point.
        """
        return np.tile(self._states[:, None], (1, n_cells))

    def initial_parameters(self, n_cells: int) -> npt.NDArray[np.float64]:
        """Build an initial parameter array for `n_cells` cells.

        Parameters
        ----------
        n_cells : int
            Number of cells.

        Returns
        -------
        numpy.ndarray
            Array of shape ``(4, n_cells)`` holding ``a = 0.7``, ``b = 0.8``, ``eps = 0.08``
            and ``stim_amplitude = 0.0`` for every cell.
        """
        return np.tile(self._parameters[:, None], (1, n_cells))

    def step(
        self,
        states: npt.NDArray[np.float64],
        t: float,
        dt: float,
        parameters: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Advance `states` by `dt` with forward Euler.

        Forward Euler suffices because FitzHugh-Nagumo is not stiff.

        Parameters
        ----------
        states : numpy.ndarray
            Current states, shape ``(2, n_cells)``.
        t : float
            Current time. Unused; the model is autonomous.
        dt : float
            Time step.
        parameters : numpy.ndarray
            Parameters, shape ``(4, n_cells)``.

        Returns
        -------
        numpy.ndarray
            New array of shape ``(2, n_cells)``.
        """
        v, w = states[0], states[1]
        a, b, eps, stim = parameters[0], parameters[1], parameters[2], parameters[3]
        dv = v - v**3 / 3.0 - w + stim
        dw = eps * (v + a - b * w)
        return np.stack((v + dt * dv, w + dt * dw))

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


def fitzhugh_nagumo() -> MembraneModel:
    """Construct a FitzHugh-Nagumo membrane model.

    Returns
    -------
    MembraneModel
        A `FitzHughNagumo` instance with default parameters, at rest.
    """
    return FitzHughNagumo()
