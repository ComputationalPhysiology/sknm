"""The units boundary: what crosses it, what is refused, and what comes out the other side.

`sknm` takes units on every dimensional argument and stores bare magnitudes in one base set.
These tests cover the conversion itself; `test_network.py` covers the constructors enforcing it.
"""

import numpy as np
import pint
import pytest

from sknm import units
from sknm.units import (
    BASE_UNITS,
    as_number,
    cm,
    fF,
    in_base_units,
    mS,
    uF,
    um,
    ureg,
    with_base_units,
)


def test_every_base_unit_parses_and_is_distinct_by_dimension():
    """Two dimensions sharing a dimensionality would make the checking vacuous between them."""
    dimensionalities = {
        dimension: ureg.Quantity(1.0, unit).dimensionality for dimension, unit in BASE_UNITS.items()
    }
    assert len(set(map(str, dimensionalities.values()))) == len(BASE_UNITS)


@pytest.mark.parametrize(
    ("written", "dimension", "expected"),
    [
        (16 * units.um, "length", 16e-4),
        (1 * units.mm, "length", 0.1),
        (1 * units.cm, "length", 1.0),
        (16 * units.um * 19.2 * units.um, "area", 16e-4 * 19.2e-4),
        (1 * units.s, "time", 1000.0),
        (1 * units.V, "potential", 1000.0),
        (1 * units.uS, "conductance", 1e-3),
        (1 / (5e3 * units.kohm), "conductance", 2e-4),
        (4 * units.mS / units.cm, "conductivity", 4.0),
        (5300 * units.fF, "capacitance", 5.3e-6),
        (1 * units.uF / (units.cm * units.cm), "specific_capacitance", 1.0),
        (20 * units.uA / (units.cm * units.cm), "current_density", 20.0),
    ],
)
def test_in_base_units_converts_to_the_documented_base(written, dimension, expected):
    assert in_base_units(written, dimension, name="value") == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("value", [16, 16.0, np.float64(16), np.full(3, 16.0)])
def test_in_base_units_refuses_a_bare_number(value):
    """The whole point: a number with no unit cannot be assumed to be in the base unit."""
    with pytest.raises(TypeError, match="lx must be a quantity with units of length"):
        in_base_units(value, "length", name="lx")


def test_in_base_units_refuses_the_wrong_dimension_and_names_the_parameter():
    with pytest.raises(pint.DimensionalityError, match="lx must be a length"):
        in_base_units(16 * units.ms, "length", name="lx")


def test_in_base_units_refuses_an_unknown_dimension():
    with pytest.raises(KeyError, match="unknown dimension"):
        in_base_units(16 * um, "luminosity", name="value")


def test_in_base_units_returns_a_bare_float_for_a_scalar():
    result = in_base_units(16 * um, "length", name="lx")
    assert isinstance(result, float)
    assert not isinstance(result, ureg.Quantity)


def test_in_base_units_returns_a_bare_float64_array_for_an_array():
    result = in_base_units(np.full(4, 16.0) * um, "length", name="length")
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float64
    np.testing.assert_allclose(result, 16e-4)


def test_numpy_would_have_stripped_the_unit_without_converting():
    """Why `in_base_units` exists rather than a plain `np.asarray`.

    Asking numpy for an array of a quantity yields the magnitude in whatever unit the quantity
    happens to carry. For ``16 * um`` that is the number 16 -- which, read as the base unit, is
    16 cm. Four orders of magnitude, silently. Nothing in `sknm` may take that route.
    """
    lx = 16 * um
    assert np.asarray(lx) == 16.0
    assert in_base_units(lx, "length", name="lx") == pytest.approx(16e-4)


@pytest.mark.parametrize("dimension", sorted(BASE_UNITS))
def test_with_base_units_round_trips_through_in_base_units(dimension):
    magnitudes = np.array([1.0, 2.5, 1e-4])
    attached = with_base_units(magnitudes, dimension)
    assert isinstance(attached, ureg.Quantity)
    np.testing.assert_allclose(
        in_base_units(attached, dimension, name="value"), magnitudes, rtol=0, atol=0
    )


def test_with_base_units_refuses_an_unknown_dimension():
    with pytest.raises(KeyError, match="unknown dimension"):
        with_base_units(1.0, "luminosity")


@pytest.mark.parametrize("value", [0.2, np.float64(0.2), ureg.Quantity(0.2)])
def test_as_number_accepts_anything_dimensionless(value):
    assert as_number(value, name="delta_e") == pytest.approx(0.2)


def test_as_number_accepts_a_bare_array():
    result = as_number(np.array([0.1, 0.3]), name="delta_e")
    np.testing.assert_allclose(result, [0.1, 0.3])
    assert result.dtype == np.float64


def test_as_number_refuses_a_dimensional_quantity():
    with pytest.raises(pint.DimensionalityError, match="delta_e is a ratio"):
        as_number(0.2 * um, name="delta_e")


def test_the_registry_is_pints_shared_application_registry():
    """A library that makes its own registry cannot meet quantities made anywhere else."""
    assert ureg is pint.get_application_registry()


@pytest.mark.parametrize(
    ("name", "dimension"),
    [
        ("cm", "length"),
        ("mm", "length"),
        ("um", "length"),
        ("nm", "length"),
        ("ms", "time"),
        ("s", "time"),
        ("us", "time"),
        ("mV", "potential"),
        ("V", "potential"),
        ("uV", "potential"),
        ("mS", "conductance"),
        ("S", "conductance"),
        ("uS", "conductance"),
        ("nS", "conductance"),
        ("uF", "capacitance"),
        ("nF", "capacitance"),
        ("pF", "capacitance"),
        ("fF", "capacitance"),
        ("uA", "current"),
        ("mA", "current"),
        ("nA", "current"),
        ("pA", "current"),
    ],
)
def test_every_exported_unit_measures_what_its_name_says(name, dimension):
    quantity = 1 * getattr(units, name)
    assert quantity.dimensionality == ureg.Quantity(1.0, BASE_UNITS[dimension]).dimensionality


@pytest.mark.parametrize("name", ["ohm", "kohm", "Mohm", "Gohm"])
def test_the_resistance_units_invert_the_conductance_units(name):
    conductance = 1 / (1 * getattr(units, name))
    assert in_base_units(conductance, "conductance", name="Gg") > 0


def test_the_two_membrane_capacitance_conventions_agree():
    """A specific capacitance times a membrane area must equal an absolute capacitance.

    Membrane models differ on which of the two they state, and a mismatch shows up as a
    plausible wave at the wrong speed rather than as an obvious failure.
    """
    specific = 1.0 * uF / (cm * cm)
    membrane_area = 1800 * um * um
    assert (specific * membrane_area).m_as("pF") == pytest.approx(18.0)
    assert in_base_units(specific * membrane_area, "capacitance", name="Cm") == pytest.approx(18e-6)


def test_a_femtofarad_is_what_the_beta_cell_model_means_by_one():
    assert in_base_units(5300 * fF, "capacitance", name="Cm") == pytest.approx(5.3e-6)


def test_the_paper_gap_junction_resistance_inverts_to_its_conductance():
    """Three spellings of the same conductance, all landing on 2e-4 mS."""
    for written in [1 / (5e3 * units.kohm), 0.2 * units.uS, 2e-4 * mS]:
        assert in_base_units(written, "conductance", name="Gg") == pytest.approx(2e-4)
