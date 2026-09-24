"""Units, carried on the values themselves rather than left to a convention.

Every dimensional quantity entering `sknm` must be a `pint` quantity: ``16 * um``, not ``16``.
A bare number is rejected, and a quantity whose dimension is wrong -- a time where a length
belongs -- is rejected too. The mistake this forecloses is the silent one: geometry entered in
the wrong unit produces a network that solves perfectly well and describes nothing biological.

Internally the package stores bare magnitudes in one fixed base set, the cardiac CGS set the
reference implementation uses. Units are parsed once, at the boundary; nothing in the numerical
core carries a unit:

========================  ==============  =====================
Dimension                 Base unit       Written as
========================  ==============  =====================
``length``                centimetre      ``cm``
``area``                  square cm       ``cm ** 2``
``time``                  millisecond     ``ms``
``potential``             millivolt       ``mV``
``conductance``           millisiemens    ``mS``
``conductivity``          mS per cm       ``mS / cm``
``capacitance``           microfarad      ``uF``
``specific_capacitance``  uF per cm^2     ``uF / cm ** 2``
``current``               microampere     ``uA``
``current_density``       uA per cm^2     ``uA / cm ** 2``
``velocity``              cm per second   ``cm / s``
========================  ==============  =====================

The one oddity is `velocity`, whose base is centimetres per *second* while every other time in
the package is a millisecond. A conduction velocity is reported in cm/s throughout the
literature, and it is the only quantity here that leaves as a number a reader compares against a
published figure rather than as state the numerical core consumes.

Read an attribute off a `CellNetwork` and you get a bare float in the base unit for its
dimension, documented on the attribute. Conversion happens at the constructors and nowhere
else.

Quantities are built from pint's *application registry*, which is the registry pint hands to
libraries so that quantities made elsewhere in a program interoperate with these:

```python
from sknm.units import um, mS, cm
lx = 16 * um
conductivity = 4 * mS / cm
```

The names are ASCII transliterations -- `um` for the micrometre, `uF` for the microfarad --
because mixing Greek mu into identifiers invites two spellings of the same name.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pint

#: The registry every `sknm` quantity belongs to. Pint's application registry rather than a
#: private one, so that quantities built by other libraries in the same program convert here.
ureg: pint.registry.ApplicationRegistry = pint.get_application_registry()

# Length.
cm = ureg.cm
mm = ureg.mm
um = ureg.um
nm = ureg.nm

# Time.
ms = ureg.ms
s = ureg.s
us = ureg.us

# Potential.
mV = ureg.mV
V = ureg.V
uV = ureg.uV

# Conductance. A conductivity is one of these over a length, for example ``4 * mS / cm``.
mS = ureg.mS
S = ureg.S
uS = ureg.uS
nS = ureg.nS

# Resistance.
ohm = ureg.ohm
kohm = ureg.kohm
Mohm = ureg.Mohm
Gohm = ureg.Gohm

# Capacitance. A specific capacitance is one of these over an area, ``1 * uF / cm ** 2``.
uF = ureg.uF
nF = ureg.nF
pF = ureg.pF
fF = ureg.fF

# Current. A current density is one of these over an area, ``20 * uA / cm ** 2``.
uA = ureg.uA
mA = ureg.mA
nA = ureg.nA
pA = ureg.pA

#: The unit each dimension is stored in once it has crossed into the package.
BASE_UNITS: dict[str, str] = {
    "length": "cm",
    "area": "cm ** 2",
    "time": "ms",
    "potential": "mV",
    "conductance": "mS",
    "conductivity": "mS / cm",
    "capacitance": "uF",
    "specific_capacitance": "uF / cm ** 2",
    "current": "uA",
    "current_density": "uA / cm ** 2",
    "velocity": "cm / s",
}


def in_base_units(value: Any, dimension: str, *, name: str) -> Any:
    """Convert a quantity to the base unit for `dimension` and strip the unit off.

    The one place a unit is removed. It is done through `pint.Quantity.m_as`, never through
    `numpy.asarray`: asking numpy for an array of a quantity yields its magnitude in whatever
    unit it happens to be carrying, so ``16 * um`` would arrive as the number 16 -- a factor of
    10,000 wrong, and silent.

    Parameters
    ----------
    value : pint.Quantity
        The quantity to convert. A scalar or an array quantity.
    dimension : str
        Key into `BASE_UNITS` naming what `value` is meant to measure.
    name : str
        Name of the parameter `value` came from, used in the error messages.

    Returns
    -------
    float or numpy.ndarray
        The magnitude of `value` in the base unit, with no unit attached. A float for a scalar
        quantity, an array for an array quantity.

    Raises
    ------
    KeyError
        If `dimension` is not a key of `BASE_UNITS`.
    TypeError
        If `value` is not a pint quantity.
    pint.DimensionalityError
        If `value` does not measure `dimension`.
    """
    if dimension not in BASE_UNITS:
        raise KeyError(f"unknown dimension {dimension!r}; expected one of {sorted(BASE_UNITS)}")
    base = BASE_UNITS[dimension]
    if not isinstance(value, ureg.Quantity):
        raise TypeError(
            f"{name} must be a quantity with units of {dimension}, for example "
            f"`16 * um`, not a bare {type(value).__name__}. Units are required on every "
            f"dimensional argument so that a dropped conversion cannot pass unnoticed."
        )
    try:
        return value.m_as(base)
    except pint.DimensionalityError as exc:
        raise pint.DimensionalityError(
            exc.units1,
            exc.units2,
            exc.dim1,
            exc.dim2,
            extra_msg=f" -- {name} must be a {dimension}",
        ) from None


def with_base_units(value: Any, dimension: str) -> Any:
    """Re-attach the base unit to a bare magnitude. The inverse of `in_base_units`.

    Used where an object holding stored magnitudes has to be rebuilt through the same
    constructor that parsed them, so that there is only ever one unit-checking path.

    Parameters
    ----------
    value : float or numpy.ndarray
        A magnitude already expressed in the base unit for `dimension`.
    dimension : str
        Key into `BASE_UNITS`.

    Returns
    -------
    pint.Quantity
        `value` carrying the base unit.

    Raises
    ------
    KeyError
        If `dimension` is not a key of `BASE_UNITS`.
    """
    if dimension not in BASE_UNITS:
        raise KeyError(f"unknown dimension {dimension!r}; expected one of {sorted(BASE_UNITS)}")
    return ureg.Quantity(value, BASE_UNITS[dimension])


def as_number(value: Any, *, name: str) -> float | npt.NDArray[np.float64]:
    """Accept a dimensionless value, as a bare number or as a dimensionless quantity.

    Volume fractions and ratios have no dimension, so requiring a unit on them would be
    ceremony. A quantity is still accepted, and is still checked for being dimensionless.

    Parameters
    ----------
    value : float, numpy.ndarray or pint.Quantity
        The value.
    name : str
        Name of the parameter `value` came from, used in the error message.

    Returns
    -------
    float or numpy.ndarray
        `value` as a bare number.

    Raises
    ------
    pint.DimensionalityError
        If `value` is a quantity that is not dimensionless.
    """
    if isinstance(value, ureg.Quantity):
        try:
            converted = value.m_as("")
        except pint.DimensionalityError as exc:
            raise pint.DimensionalityError(
                exc.units1,
                exc.units2,
                exc.dim1,
                exc.dim2,
                extra_msg=f" -- {name} is a ratio and must be dimensionless",
            ) from None
        return converted
    if np.ndim(value) == 0:
        return float(value)
    return np.asarray(value, dtype=np.float64)
