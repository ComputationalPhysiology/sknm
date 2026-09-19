"""Unit constants, so that call sites can read ``16 * um`` instead of ``16e-4``.

These are plain floats, not a unit system: nothing in `sknm` checks or converts units at run
time. Multiplying by a constant from here converts a quantity *into* the package's base units,
which are the cardiac CGS set the reference implementation uses:

===========  ============  =====================
Quantity     Base unit     Constant equal to one
===========  ============  =====================
length       centimetre    ``cm``
time         millisecond   ``ms``
potential    millivolt     ``mV``
conductance  millisiemens  ``mS``
resistance   kilohm        ``kohm``
capacitance  microfarad    ``uF``
current      microampere   ``uA``
===========  ============  =====================

Derived units follow from products and quotients and so need no constants of their own: an
area is ``16 * um * 16 * um``, a conductivity is ``4 * mS / cm``, and a specific capacitance is
``1 * uF / (cm * cm)``.

The names are ASCII transliterations -- `um` for the micrometre, `uF` for the microfarad --
because mixing Greek mu into identifiers invites two spellings of the same name.

Import the module, or the handful of constants a script needs::

    from sknm import units
    from sknm.units import um
"""

# Length.
cm = 1.0
mm = 1e-1
um = 1e-4
nm = 1e-7

# Time.
ms = 1.0
s = 1e3
us = 1e-3

# Potential.
mV = 1.0
V = 1e3
uV = 1e-3

# Conductance. Conductivities are these over a length, for example ``4 * mS / cm``.
mS = 1.0
S = 1e3
uS = 1e-3
nS = 1e-6

# Resistance. The base unit is the reciprocal of the base conductance, which is a kilohm and
# not an ohm -- gap junction resistances in this model run to megohms, so the scale is apt.
kohm = 1.0
ohm = 1e-3
Mohm = 1e3
Gohm = 1e6

# Capacitance. Specific capacitances are these over an area, for example ``1 * uF / (cm * cm)``.
uF = 1.0
nF = 1e-3
pF = 1e-6
fF = 1e-9

# Current. Current densities are these over an area, for example ``20 * uA / (cm * cm)``.
uA = 1.0
mA = 1e3
nA = 1e-3
pA = 1e-6
