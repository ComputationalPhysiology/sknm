"""Measuring a wave: when each cell activated, how fast it rose, and how fast the wave moved.

The recorder is driven two ways here. Most tests hand it a `Ramp` membrane model, whose
potential rises at a rate the test chose, so that the activation time and the upstroke velocity
both have closed forms written down without reference to this code. The rest drive it through a
real `Simulation` on a chain, to pin that it sees the potential *after* the spatial solve and
that it can stop a run.

`tests/test_validation.py` is where the same objects are pointed at the published numbers.
"""

import numpy as np
import pint
import pytest

from sknm import Simulation, Variant, chain
from sknm.analysis import ActivationRecorder, ConductionPath, conduction_velocity
from sknm.membrane import fitzhugh_nagumo
from sknm.units import cm, mS, ms, mV, ohm, s, um

DELTA_E = 0.2
MEMBRANE_AREA = 1.8e-5 * cm**2
DT = 0.02 * ms
#: A step the ramp tests can land on exactly: 0.25 and the rates below are binary-exact,
#: so an activation time is the closed form and not the closed form plus a rounding drift.
RAMP_DT = 0.25 * ms


def geometry():
    ly = 14 * um
    return {
        "lx": 21 * um,
        "ly": ly,
        "lz": (1 + DELTA_E) * ly,
        "delta_e": DELTA_E,
        "sigma_i": 4.0 * mS / cm,
        "sigma_e": 20.0 * mS / cm,
        "Gg": 2e-4 * mS,
    }


class Ramp:
    """A membrane model whose potential rises linearly at a per-cell rate, in mV/ms.

    No coupling and no dynamics beyond the ramp, so every cell's activation time and upstroke
    velocity follow from arithmetic the test does itself. The resting potential is an odd
    number of millivolts so that no crossing of a round threshold lands exactly on a step
    boundary, where the last bit of the linear solve would decide which step it belongs to.
    """

    v_name = "v"
    v_index = 0
    num_states = 1
    num_parameters = 1
    capacitance = None

    def __init__(self, rates, start=-81.0):
        self._rates = np.asarray(rates, dtype=np.float64)
        self._start = start

    def state_index(self, name):
        if name != "v":
            raise KeyError(name)
        return 0

    def parameter_index(self, name):
        if name != "rate":
            raise KeyError(name)
        return 0

    def initial_states(self, n_cells):
        return np.full((1, n_cells), self._start)

    def initial_parameters(self, n_cells):
        return self._rates.reshape((1, n_cells)).copy()

    def step(self, states, t, dt, parameters):
        return states + dt * parameters


@pytest.fixture
def strand():
    return chain(3, membrane_area=MEMBRANE_AREA, **geometry())


def uncoupled(network):
    """The same network with every gap junction shut, so each cell evolves on its own.

    `Variant.SKNM_UE0` because it is the one variant that does not divide by `lam`, which a
    network with no intracellular coupling has no value for.
    """
    return network.with_conductances(np.zeros(network.n_connections) * mS)


def ramp_simulation(network, rates, **kwargs):
    """A simulation whose only dynamics are the ramp, with the coupling switched off."""
    return Simulation(
        uncoupled(network), Ramp(rates), dt=RAMP_DT, variant=Variant.SKNM_UE0, **kwargs
    )


# --- ConductionPath -------------------------------------------------------------------------


def test_a_conduction_path_stores_its_distance_in_centimetres():
    path = ConductionPath(start=3, end=17, distance=400 * um)

    assert (path.start, path.end) == (3, 17)
    assert path.distance == pytest.approx(0.04)


def test_a_conduction_path_refuses_a_distance_without_units():
    with pytest.raises(TypeError, match="distance must be a quantity with units of length"):
        ConductionPath(start=0, end=1, distance=0.04)


def test_a_conduction_path_refuses_a_distance_that_is_not_a_length():
    with pytest.raises(pint.DimensionalityError, match="distance must be a length"):
        ConductionPath(start=0, end=1, distance=5 * ohm)


@pytest.mark.parametrize("distance", [0 * um, -400 * um])
def test_a_conduction_path_refuses_a_distance_that_is_not_positive(distance):
    with pytest.raises(ValueError, match="distance must be positive"):
        ConductionPath(start=0, end=1, distance=distance)


def test_a_conduction_path_refuses_a_cell_joined_to_itself():
    with pytest.raises(ValueError, match="start and end must be different cells"):
        ConductionPath(start=4, end=4, distance=400 * um)


def test_a_conduction_path_cannot_be_rebound():
    path = ConductionPath(start=0, end=1, distance=400 * um)

    with pytest.raises(AttributeError):
        path.distance = 1.0


# --- What the recorder records --------------------------------------------------------------


def test_a_cell_activates_at_the_step_its_potential_first_reaches_the_threshold(strand):
    # Rising 10 mV/ms from -81 mV, a cell passes -20 mV during the 25th step of 0.25 ms.
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)

    sim.run(10 * ms, record=(), callback=recorder)

    np.testing.assert_allclose(recorder.activation_time, 6.25)


def test_cells_rising_at_different_rates_activate_at_different_times(strand):
    sim = ramp_simulation(strand, rates=[10.0, 20.0, 30.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)

    sim.run(10 * ms, record=(), callback=recorder)

    np.testing.assert_allclose(recorder.activation_time, [6.25, 3.25, 2.25])


def test_a_cell_that_never_reaches_the_threshold_has_no_activation_time(strand):
    sim = ramp_simulation(strand, rates=[10.0, 0.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)

    sim.run(10 * ms, record=(), callback=recorder)

    assert np.isnan(recorder.activation_time[1])
    assert not np.isnan(recorder.activation_time[[0, 2]]).any()


def test_the_threshold_is_the_one_the_recorder_was_given(strand):
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-30 * mV)

    sim.run(10 * ms, record=(), callback=recorder)

    assert recorder.threshold == pytest.approx(-30.0)
    np.testing.assert_allclose(recorder.activation_time, 5.25)


def test_the_threshold_defaults_to_the_reference_value(strand):
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])

    assert ActivationRecorder(sim).threshold == pytest.approx(-20.0)


def test_the_threshold_must_be_a_potential(strand):
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])

    with pytest.raises(TypeError, match="threshold must be a quantity with units of potential"):
        ActivationRecorder(sim, threshold=-20.0)


def test_a_cell_already_above_the_threshold_activates_at_the_first_step(strand):
    """A resting potential above the threshold is a setup mistake, not an activation at t=0."""
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])
    sim.states[0, :] = 0.0
    recorder = ActivationRecorder(sim, threshold=-20 * mV)

    sim.run(1 * ms, record=(), callback=recorder)

    np.testing.assert_allclose(recorder.activation_time, sim.dt)


def test_a_cell_exactly_at_the_threshold_has_reached_it(strand):
    """The threshold is a level reached, not a level passed, which is the reference's test."""
    sim = ramp_simulation(strand, rates=[0.0, 0.0, 0.0])
    sim.states[0, :] = -20.0
    recorder = ActivationRecorder(sim, threshold=-20 * mV)

    sim.run(1 * ms, record=(), callback=recorder)

    np.testing.assert_allclose(recorder.activation_time, sim.dt)


def test_the_upstroke_velocity_is_the_greatest_rate_of_rise(strand):
    sim = ramp_simulation(strand, rates=[10.0, 20.0, 30.0])
    recorder = ActivationRecorder(sim)

    sim.run(10 * ms, record=(), callback=recorder)

    np.testing.assert_allclose(recorder.max_upstroke_velocity, [10.0, 20.0, 30.0], rtol=1e-12)


def test_the_upstroke_velocity_is_a_centred_difference(strand):
    """Supplementary (S8): max over n of (v[n+1] - v[n-1]) / (2*dt), not a forward difference.

    A potential that jumps by one step's worth in a single step has a forward difference of the
    full jump rate and a centred difference of half of it. The two disagree here by construction.
    """
    jump = 40.0

    class Jump(Ramp):
        def step(self, states, t, dt, parameters):
            rising = np.isclose(t, 4 * float(RAMP_DT.m_as("ms")))
            return states + (jump if rising else 0.0)

    sim = Simulation(
        uncoupled(strand), Jump(rates=[0.0, 0.0, 0.0]), dt=RAMP_DT, variant=Variant.SKNM_UE0
    )
    recorder = ActivationRecorder(sim)

    sim.run(10 * RAMP_DT, record=(), callback=recorder)

    centred = jump / (2 * float(RAMP_DT.m_as("ms")))
    np.testing.assert_allclose(recorder.max_upstroke_velocity, centred, rtol=1e-12)


def test_nothing_is_recorded_before_the_first_step(strand):
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])
    recorder = ActivationRecorder(sim)

    assert np.isnan(recorder.activation_time).all()
    assert np.isnan(recorder.max_upstroke_velocity).all()


def test_the_recorded_arrays_are_read_only(strand):
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])
    recorder = ActivationRecorder(sim)

    with pytest.raises(ValueError, match="read-only"):
        recorder.activation_time[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        recorder.max_upstroke_velocity[0] = 1.0


def test_a_second_run_carries_on_recording(strand):
    """The recorder observes a simulation, not a single call to `run`."""
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)

    sim.run(3 * ms, record=(), callback=recorder)
    assert np.isnan(recorder.activation_time).all()

    sim.run(10 * ms, record=(), callback=recorder)
    np.testing.assert_allclose(recorder.activation_time, 6.25)


# --- Stopping a run -------------------------------------------------------------------------


def test_a_run_stops_when_the_named_cell_activates(strand):
    sim = ramp_simulation(strand, rates=[30.0, 20.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV, stop_when_activated=1)

    sim.run(100 * ms, record=(), callback=recorder)

    assert sim.t == pytest.approx(3.25)
    assert np.isnan(recorder.activation_time[2])


def test_a_run_stops_only_once_every_named_cell_has_activated(strand):
    sim = ramp_simulation(strand, rates=[30.0, 20.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV, stop_when_activated=[0, 1])

    sim.run(100 * ms, record=(), callback=recorder)

    assert sim.t == pytest.approx(3.25)


def test_a_run_goes_to_its_end_when_no_stopping_cell_is_named(strand):
    sim = ramp_simulation(strand, rates=[30.0, 20.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)

    sim.run(8 * ms, record=(), callback=recorder)

    assert sim.t == pytest.approx(8.0)


def test_a_stopping_cell_that_does_not_exist_is_refused(strand):
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])

    with pytest.raises(IndexError, match="stop_when_activated"):
        ActivationRecorder(sim, stop_when_activated=7)


# --- Conduction velocity --------------------------------------------------------------------


def test_conduction_velocity_is_the_distance_over_the_activation_delay(strand):
    # Cell 0 activates at 2.25 ms and cell 2 at 6.25 ms; 400 um in 4 ms is 10 cm/s.
    sim = ramp_simulation(strand, rates=[30.0, 0.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)
    sim.run(10 * ms, record=(), callback=recorder)

    velocity = conduction_velocity(recorder, ConductionPath(0, 2, distance=400 * um))

    assert velocity.m_as(cm / s) == pytest.approx(10.0)


def test_conduction_velocity_comes_back_as_a_quantity(strand):
    sim = ramp_simulation(strand, rates=[30.0, 0.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)
    sim.run(10 * ms, record=(), callback=recorder)

    velocity = conduction_velocity(recorder, ConductionPath(0, 2, distance=400 * um))

    assert velocity.check("[length] / [time]")
    assert velocity.m_as("mm / s") == pytest.approx(100.0)


def test_conduction_velocity_refuses_a_wave_that_never_reached_the_end(strand):
    sim = ramp_simulation(strand, rates=[30.0, 0.0, 0.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)
    sim.run(10 * ms, record=(), callback=recorder)

    with pytest.raises(ValueError, match="cell 2 did not reach"):
        conduction_velocity(recorder, ConductionPath(0, 2, distance=400 * um))


def test_conduction_velocity_refuses_a_wave_that_never_left_the_start(strand):
    sim = ramp_simulation(strand, rates=[0.0, 0.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)
    sim.run(10 * ms, record=(), callback=recorder)

    with pytest.raises(ValueError, match="cell 0 did not reach"):
        conduction_velocity(recorder, ConductionPath(0, 2, distance=400 * um))


def test_conduction_velocity_refuses_a_path_the_wave_travelled_backwards(strand):
    sim = ramp_simulation(strand, rates=[10.0, 0.0, 30.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)
    sim.run(10 * ms, record=(), callback=recorder)

    with pytest.raises(ValueError, match="activated before"):
        conduction_velocity(recorder, ConductionPath(0, 2, distance=400 * um))


def test_conduction_velocity_refuses_two_cells_that_activated_in_the_same_step(strand):
    sim = ramp_simulation(strand, rates=[10.0, 10.0, 10.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)
    sim.run(10 * ms, record=(), callback=recorder)

    with pytest.raises(ValueError, match="activated before"):
        conduction_velocity(recorder, ConductionPath(0, 2, distance=400 * um))


def test_conduction_velocity_refuses_a_path_off_the_end_of_the_network(strand):
    sim = ramp_simulation(strand, rates=[10.0, 20.0, 30.0])
    recorder = ActivationRecorder(sim, threshold=-20 * mV)
    sim.run(10 * ms, record=(), callback=recorder)

    with pytest.raises(IndexError):
        conduction_velocity(recorder, ConductionPath(0, 9, distance=400 * um))


# --- Against a real simulation --------------------------------------------------------------


def test_the_recorder_sees_the_potential_after_the_spatial_solve():
    """The membrane step and the solve give different potentials; only one of them is `v`."""
    network = chain(4, membrane_area=MEMBRANE_AREA, **geometry())
    sim = Simulation(network, fitzhugh_nagumo(), dt=DT)
    sim.states[0, 0] = 2.0

    # Starting from the potential before the first step, which is where the recorder starts too.
    potentials = [sim.v.copy()]
    recorder = ActivationRecorder(sim, threshold=0 * mV)

    def both(simulation):
        potentials.append(simulation.v.copy())
        return recorder(simulation)

    sim.run(20 * DT, record=(), callback=both)

    trace = np.stack(potentials, axis=1)
    rates = (trace[:, 2:] - trace[:, :-2]) / (2 * sim.dt)
    np.testing.assert_allclose(recorder.max_upstroke_velocity, rates.max(axis=1), rtol=1e-9)


def test_a_wave_along_a_chain_has_a_finite_conduction_velocity():
    """The end-to-end shape, on a network small enough to have no published number.

    FitzHugh-Nagumo carries no units and rests near -1.2, so the threshold here is a number in
    its own range rather than the -20 mV a cardiac model is measured at.
    """
    network = chain(6, membrane_area=MEMBRANE_AREA, **geometry())
    sim = Simulation(network, fitzhugh_nagumo(), dt=DT)
    sim.set_parameter("stim_amplitude", 1.0, cells=[0])

    path = ConductionPath(1, 5, distance=4 * 21 * um)
    recorder = ActivationRecorder(sim, threshold=0 * mV, stop_when_activated=path.end)
    sim.run(200 * ms, record=(), callback=recorder)

    velocity = conduction_velocity(recorder, path)

    assert velocity.m_as("cm / s") > 0.0
    assert sim.t < 200.0, "the run should have stopped as soon as the wave arrived"
