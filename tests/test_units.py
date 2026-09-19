"""The unit constants.

These are plain floats in one fixed base set, so the only thing worth asserting is that they
are mutually consistent: a quantity written two ways must come out the same number.
"""

import pytest

from sknm import units


def test_the_base_units_are_one():
    assert units.cm == 1.0
    assert units.ms == 1.0
    assert units.mV == 1.0
    assert units.mS == 1.0
    assert units.uF == 1.0
    assert units.uA == 1.0


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        (16 * units.um, 16e-4),
        (1000 * units.um, units.mm),
        (10 * units.mm, units.cm),
        (1000 * units.nm, units.um),
        (1000 * units.ms, units.s),
        (1000 * units.us, units.ms),
        (1000 * units.mV, units.V),
        (1000 * units.uV, units.mV),
        (1000 * units.mS, units.S),
        (1000 * units.uS, units.mS),
        (1000 * units.nS, units.uS),
        (1000 * units.nF, units.uF),
        (1000 * units.pF, units.nF),
        (1000 * units.fF, units.pF),
        (1000 * units.nA, units.uA),
        (1000 * units.uA, units.mA),
    ],
)
def test_the_prefixes_step_by_a_thousand(written, expected):
    assert written == pytest.approx(expected)


def test_resistance_is_the_reciprocal_of_conductance():
    """The base resistance unit is whatever inverts the base conductance unit.

    1/mS is a kilohm, which is why `kohm` and not `ohm` is the unit that equals one.
    """
    assert units.kohm == 1.0
    assert units.kohm * units.mS == pytest.approx(1.0)
    assert units.Mohm * units.uS == pytest.approx(1.0)
    assert units.ohm * units.S == pytest.approx(1.0)


def test_an_area_is_the_product_of_two_lengths():
    """Areas need no constants of their own: `um * um` is already a square centimetre count."""
    assert (16 * units.um) * (16 * units.um) == pytest.approx(256e-8)


def test_capacitance_converts_between_the_two_membrane_model_conventions():
    """A specific capacitance times a membrane area must equal an absolute capacitance.

    Membrane models differ on which of the two they state, and a mismatch shows up as a
    plausible wave at the wrong speed rather than as an obvious failure, so the conversion
    has to be exact.
    """
    specific = 1.0 * units.uF / (units.cm * units.cm)
    membrane_area = 1800 * units.um * units.um
    assert specific * membrane_area == pytest.approx(18 * units.pF)
