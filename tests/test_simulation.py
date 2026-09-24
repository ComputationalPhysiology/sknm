"""The time stepper: the Godunov loop, the recording, and what a simulation refuses to do.

Two tests carry the weight. `test_one_step_is_the_membrane_step_then_the_spatial_solve`
composes the step by hand out of the two seams and demands the same answer, which pins the
order of the splitting, the time the membrane model is handed and which potential is fed to the
operator. `test_two_quiescent_cells_relax_by_the_exact_backward_euler_factor` drives a membrane
model with no dynamics, so that the whole simulation reduces to a closed form that was derived
without reference to any of this code.

The membrane models here are real implementations rather than mocks: FitzHugh-Nagumo where
dynamics are wanted, and a `Quiescent` model where they are in the way. The one exception is
`CountingSolver`, which counts calls rather than changing answers. It exists because no
assertion on a returned value can tell a cached factorization from one redone every step.
"""

import dataclasses

import numpy as np
import pint
import pytest

from sknm import Simulation, Variant, assemble, chain, sheet
from sknm.linalg import BiCGSTABSolver, CGSolver, ConvergenceError, DirectSolver
from sknm.membrane import base_model_IM, fitzhugh_nagumo, from_gotranx
from sknm.membrane.fitzhugh_nagumo import FitzHughNagumo
from sknm.units import cm, kohm, mS, ms, uF, um

DELTA_E = 0.2
SIGMA_I = 4.0 * mS / cm
SIGMA_E = 20.0 * mS / cm
GG = (1 / (5e3 * kohm)).to(mS)
MEMBRANE_AREA = 1.8e-5 * cm**2
DT = 0.02 * ms

VARIANTS = list(Variant)
SOLVERS = [DirectSolver(), CGSolver(), BiCGSTABSolver()]


def geometry(lx_um=16, ly_um=16):
    """The paper's hiPSC-CM material properties at one of its measured cell sizes."""
    ly = ly_um * um
    return {
        "lx": lx_um * um,
        "ly": ly,
        "lz": (1 + DELTA_E) * ly,
        "delta_e": DELTA_E,
        "sigma_i": SIGMA_I,
        "sigma_e": SIGMA_E,
        "Gg": GG,
    }


@pytest.fixture
def strand():
    """A short chain, the smallest network a wave can travel along."""
    return chain(6, membrane_area=MEMBRANE_AREA, **geometry())


@pytest.fixture
def two_cells():
    return chain(2, membrane_area=MEMBRANE_AREA, **geometry())


@pytest.fixture
def varied_sheet():
    """An anisotropic sheet with a different membrane area on every cell."""
    nx, ny = 3, 4
    areas = np.linspace(1.0, 2.5, nx * ny) * MEMBRANE_AREA
    return sheet(nx, ny, membrane_area=areas, **geometry(21, 14))


class Quiescent:
    """A membrane model with no dynamics: one state, and a step that changes nothing.

    Reduces a simulation to the spatial operator alone, so that its answer can be compared
    against a closed form derived by hand rather than against another run of the same code.
    """

    v_name = "v"
    v_index = 0
    num_states = 1
    num_parameters = 1
    capacitance = None

    def state_index(self, name):
        if name != "v":
            raise KeyError(f"unknown state {name!r}")
        return 0

    def parameter_index(self, name):
        if name != "unused":
            raise KeyError(f"unknown parameter {name!r}")
        return 0

    def initial_states(self, n_cells):
        return np.zeros((1, n_cells))

    def initial_parameters(self, n_cells):
        return np.zeros((1, n_cells))

    def step(self, states, t, dt, parameters):
        return states.copy()


class BuriedPotential:
    """A quiescent model whose membrane potential is its second state rather than its first.

    Both models the tests otherwise use put the membrane potential in row zero, so nothing
    distinguishes "the row the model names" from "the first row", while the shipped hiPSC-CM
    model puts `V_m` in row twenty-four. The states on either side of it are constants, so a
    simulation that reached for the wrong row would both read and write the wrong numbers.
    """

    v_name = "V"
    v_index = 1
    num_states = 3
    num_parameters = 1
    capacitance = None

    def state_index(self, name):
        try:
            return ("before", "V", "after").index(name)
        except ValueError:
            raise KeyError(f"unknown state {name!r}") from None

    def parameter_index(self, name):
        raise KeyError(f"unknown parameter {name!r}")

    def initial_states(self, n_cells):
        return np.tile(np.array([[-1.0], [0.0], [1.0]]), (1, n_cells))

    def initial_parameters(self, n_cells):
        return np.zeros((1, n_cells))

    def step(self, states, t, dt, parameters):
        return states.copy()


class InPlace(Quiescent):
    """A membrane model that writes into the array it was handed and returns that same array.

    The protocol asks for a new array and both shipped models give one, but a recorded trace
    must not depend on that: a sample that aliased the live state array would rewrite its own
    history every step, and `Simulation.states` is public and writable too.
    """

    def step(self, states, t, dt, parameters):
        states += 0.0
        return states


class DeclaredCapacitance(FitzHughNagumo):
    """FitzHugh-Nagumo that declares a capacitance, so the check has something to compare.

    Neither shipped model declares one, since the hiPSC-CM model's voltage equation does not
    use the capacitance it exposes, so the check would otherwise be unreachable from a test.
    """

    def __init__(self, capacitance):
        super().__init__()
        self._declared = capacitance

    @property
    def capacitance(self):
        return self._declared


@dataclasses.dataclass(frozen=True)
class CountingSolver:
    """A `DirectSolver` that records how often it was asked to factorize.

    The count lives in a list because the solver is frozen, and frozen is what every solver is.
    """

    calls: list[int] = dataclasses.field(default_factory=list)

    def factorize(self, matrix):
        self.calls.append(matrix.shape[0])
        return DirectSolver().factorize(matrix)


def stimulated(simulation, cells=(0,), amplitude=1.0):
    """Drive some cells hard enough to fire, and hand the simulation back."""
    simulation.set_parameter("stim_amplitude", amplitude, cells=list(cells))
    return simulation


# --- The step ------------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_one_step_is_the_membrane_step_then_the_spatial_solve(varied_sheet, variant):
    """Compose the step by hand from the two seams and demand the same answer.

    Catches the order of the splitting, the time handed to the membrane model, and a spatial
    solve fed the potential from before the membrane step rather than after it.
    """
    model = fitzhugh_nagumo()
    simulation = stimulated(Simulation(varied_sheet, model, variant=variant, dt=DT), cells=range(3))

    operator = assemble(varied_sheet, dt=DT, variant=variant)
    states = model.step(simulation.states, 0.0, DT.m_as("ms"), simulation.parameters)
    solution = DirectSolver().factorize(operator.matrix)(operator.rhs(states[model.v_index]))
    expected = operator.membrane_potential(solution)

    simulation.step()

    np.testing.assert_allclose(simulation.v, expected, rtol=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_network_at_rest_stays_at_rest(strand, variant):
    """A constant potential is in the kernel of every Laplacian, and rest has no dynamics."""
    simulation = Simulation(strand, fitzhugh_nagumo(), variant=variant, dt=DT)
    initial = simulation.v.copy()

    for _ in range(50):
        simulation.step()

    np.testing.assert_allclose(simulation.v, initial, rtol=1e-13)


@pytest.mark.parametrize("variant", VARIANTS)
def test_two_quiescent_cells_relax_by_the_exact_backward_euler_factor(two_cells, variant):
    """With no membrane dynamics the whole simulation is one closed form.

    The potential difference between two cells decays by `D / (D + 2 * G)` per step, where `G`
    is the conductance each variant couples them through: the intracellular conductance alone
    for SKNM(u_e=0), that in series with the extracellular one for KNM, and the intracellular
    one reduced by `lam / (1 + lam)` for SKNM.
    """
    intra, extra = two_cells.Gi[0], two_cells.Ge[0]
    coupling = {
        Variant.SKNM_UE0: intra,
        Variant.KNM: intra * extra / (intra + extra),
        Variant.SKNM: intra * two_cells.lam / (1.0 + two_cells.lam),
    }[variant]
    capacitive = two_cells.Cm * two_cells.membrane_area[0] / DT.m_as("ms")
    per_step = capacitive / (capacitive + 2 * coupling)

    simulation = Simulation(two_cells, Quiescent(), variant=variant, dt=DT)
    simulation.states[0] = [-80.0, 20.0]
    for _ in range(10):
        simulation.step()

    gap = (simulation.v[0] - simulation.v[1]) / (-80.0 - 20.0)
    assert gap == pytest.approx(per_step**10, rel=1e-11)


def test_the_solve_moves_the_state_row_the_model_names_and_no_other(two_cells):
    """`v_index` is a row number the model chooses; the hiPSC-CM model's is twenty-four."""
    model = BuriedPotential()
    simulation = Simulation(two_cells, model, dt=DT)
    simulation.states[model.v_index] = [-80.0, 20.0]
    neighbours = simulation.states[[0, 2]].copy()

    simulation.step()

    assert -80.0 < simulation.states[1, 0] < simulation.states[1, 1] < 20.0
    np.testing.assert_array_equal(simulation.states[[0, 2]], neighbours)


def test_the_membrane_potential_is_recorded_from_the_row_the_model_names(two_cells):
    model = BuriedPotential()
    simulation = Simulation(two_cells, model, dt=DT)
    simulation.states[model.v_index] = [-80.0, 20.0]

    result = simulation.run(0.1 * ms, record_every=0.1 * ms)

    np.testing.assert_array_equal(result.v[:, 0], [-80.0, 20.0])


def test_the_initial_states_and_parameters_are_the_models_own(strand):
    """A parameter nobody sets must still be the value the model chose for it."""
    model = fitzhugh_nagumo()
    n_cells = strand.n_cells
    simulation = Simulation(strand, model, dt=DT)

    np.testing.assert_array_equal(simulation.states, model.initial_states(n_cells))
    np.testing.assert_array_equal(simulation.parameters, model.initial_parameters(n_cells))


def test_a_trace_survives_a_membrane_model_that_reuses_its_buffer(two_cells):
    """A sample is a snapshot. Aliasing the live states would rewrite a trace as it is taken."""
    simulation = Simulation(two_cells, InPlace(), dt=DT)
    simulation.states[0] = [-80.0, 20.0]

    result = simulation.run(1 * ms, record_every=0.1 * ms)

    assert result.v[0, 0] == -80.0
    assert (np.diff(result.v[0]) > 0.0).all()


def test_the_membrane_potential_is_a_row_of_the_state_array(strand):
    """No second copy of `v` to fall out of step with the states around it."""
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    simulation.step()

    assert np.shares_memory(simulation.v, simulation.states)
    np.testing.assert_array_equal(simulation.v, simulation.states[0])


def test_the_clock_advances_by_one_step_at_a_time(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    assert simulation.t == 0.0

    for expected in (1, 2, 3):
        simulation.step()
        assert simulation.t == pytest.approx(expected * DT.m_as("ms"), rel=1e-15)


def test_the_membrane_model_is_handed_the_time_at_the_start_of_the_step(strand):
    """A model whose currents are gated on time sees the step it is about to take, not the
    one just finished."""
    seen = []

    class Recording(Quiescent):
        def step(self, states, t, dt, parameters):
            seen.append((t, dt))
            return states.copy()

    simulation = Simulation(strand, Recording(), dt=DT)
    simulation.step()
    simulation.step()

    assert seen == [(0.0, DT.m_as("ms")), (DT.m_as("ms"), DT.m_as("ms"))]


# --- Propagation ---------------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_stimulus_at_one_end_travels_to_the_other(strand, variant):
    """The one thing these models exist to do."""
    simulation = stimulated(Simulation(strand, fitzhugh_nagumo(), variant=variant, dt=DT))
    result = simulation.run(60 * ms, record_every=DT)

    peaks = result.v.max(axis=1)
    assert (peaks > 1.0).all()

    crossings = [np.argmax(trace > 0.0) for trace in result.v]
    assert crossings == sorted(crossings)
    assert crossings[0] < crossings[-1]


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("solver", SOLVERS)
def test_every_solver_runs_the_same_simulation(strand, variant, solver):
    """The solver is an implementation detail of one step; it cannot change the physics.

    An iterative solver leaves a residual of `rtol` behind on every one of the thousand steps
    here, and the membrane model is nonlinear, so the tolerance is against the amplitude of the
    wave, about three, rather than against each value. A solver that had actually changed the
    physics would move the upstroke, which is far larger than this.
    """
    exact = stimulated(
        Simulation(strand, fitzhugh_nagumo(), variant=variant, dt=DT, solver=DirectSolver())
    ).run(20 * ms, record_every=1 * ms)
    theirs = stimulated(
        Simulation(strand, fitzhugh_nagumo(), variant=variant, dt=DT, solver=solver)
    ).run(20 * ms, record_every=1 * ms)

    np.testing.assert_allclose(theirs.v, exact.v, rtol=1e-3, atol=1e-3)


def test_a_solver_can_be_named_by_a_string(strand):
    named = Simulation(strand, fitzhugh_nagumo(), dt=DT, solver="cg")
    named.step()

    assert np.isfinite(named.v).all()


def test_a_solver_that_cannot_converge_reports_it(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT, solver=CGSolver(maxiter=1))

    with pytest.raises(ConvergenceError):
        for _ in range(10):
            simulation.step()


# --- Factorizing once ----------------------------------------------------------------------


def test_the_operator_is_factorized_once_across_many_steps(strand):
    """No assertion on a returned value can tell a cached factorization from a repeated one."""
    solver = CountingSolver()
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT, solver=solver)

    for _ in range(25):
        simulation.step()

    assert solver.calls == [strand.n_cells]


def test_factorization_is_deferred_until_the_first_step(strand):
    """Building a sweep of simulations must not pay for every factorization up front."""
    solver = CountingSolver()
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT, solver=solver)

    assert solver.calls == []
    simulation.step()
    assert solver.calls == [strand.n_cells]


def test_setting_a_membrane_parameter_does_not_refactorize(strand):
    """Membrane parameters live in the ODEs; the network and its matrix are frozen."""
    solver = CountingSolver()
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT, solver=solver)
    simulation.step()

    simulation.set_parameter("stim_amplitude", 1.0, cells=[0])
    simulation.step()

    assert solver.calls == [strand.n_cells]


# --- Parameters ----------------------------------------------------------------------------


def test_a_parameter_can_be_set_on_a_subset_of_cells(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    index = simulation.model.parameter_index("stim_amplitude")

    simulation.set_parameter("stim_amplitude", 20.0, cells=[0, 2])

    np.testing.assert_array_equal(simulation.parameters[index], [20, 0, 20, 0, 0, 0])


def test_a_parameter_can_be_set_on_every_cell(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    index = simulation.model.parameter_index("eps")

    simulation.set_parameter("eps", 0.5)

    np.testing.assert_array_equal(simulation.parameters[index], np.full(6, 0.5))


def test_a_parameter_can_be_set_from_a_boolean_mask(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    index = simulation.model.parameter_index("stim_amplitude")
    mask = np.array([True, False, False, False, False, True])

    simulation.set_parameter("stim_amplitude", 20.0, cells=mask)

    np.testing.assert_array_equal(simulation.parameters[index], [20, 0, 0, 0, 0, 20])


def test_a_parameter_can_be_given_one_value_per_selected_cell(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    index = simulation.model.parameter_index("stim_amplitude")

    simulation.set_parameter("stim_amplitude", [1.0, 2.0], cells=[3, 4])

    np.testing.assert_array_equal(simulation.parameters[index], [0, 0, 0, 1, 2, 0])


def test_an_unknown_parameter_is_refused(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(KeyError, match="stim_ampltiude"):
        simulation.set_parameter("stim_ampltiude", 20.0)


def test_a_stimulus_actually_reaches_the_membrane_model(strand):
    """Writing into the parameter array must be what the step reads, not a private copy."""
    quiet = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    loud = stimulated(Simulation(strand, fitzhugh_nagumo(), dt=DT))

    quiet.step()
    loud.step()

    assert loud.v[0] > quiet.v[0]


# --- Recording -----------------------------------------------------------------------------


def test_recording_produces_one_sample_per_interval_plus_the_initial_state(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    result = simulation.run(10 * ms, record_every=1 * ms)

    assert result.v.shape == (6, 11)
    np.testing.assert_allclose(result.t, np.linspace(0.0, 10.0, 11), atol=1e-12)


def test_recording_every_step_is_recording_at_the_time_step(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    result = simulation.run(1 * ms, record_every=DT)

    assert result.v.shape == (6, 51)


def test_the_first_sample_is_the_state_before_any_step(strand):
    simulation = stimulated(Simulation(strand, fitzhugh_nagumo(), dt=DT))
    initial = simulation.v.copy()

    result = simulation.run(1 * ms, record_every=1 * ms)

    np.testing.assert_array_equal(result.v[:, 0], initial)
    assert not np.array_equal(result.v[:, 1], initial)


def test_a_run_leaves_the_simulation_at_the_end_time(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    simulation.run(10 * ms, record_every=1 * ms)

    assert simulation.t == pytest.approx(10.0, rel=1e-12)


def test_runs_continue_from_where_the_last_one_stopped(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    simulation.run(5 * ms, record_every=1 * ms)
    second = simulation.run(10 * ms, record_every=1 * ms)

    assert simulation.t == pytest.approx(10.0, rel=1e-12)
    np.testing.assert_allclose(second.t, np.linspace(5.0, 10.0, 6), atol=1e-12)


def test_any_state_can_be_recorded_by_name(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    result = simulation.run(1 * ms, record_every=1 * ms, record=("v", "w"))

    assert set(result.names) == {"v", "w"}
    assert result["w"].shape == (6, 2)
    np.testing.assert_array_equal(result["w"][:, 0], np.full(6, simulation.states[1, 0]))


def test_the_membrane_potential_is_recordable_as_v_whatever_the_model_calls_it(strand):
    """`record=("v",)` is the default, so it has to mean the membrane potential everywhere."""

    class Renamed(Quiescent):
        v_name = "u"

        def state_index(self, name):
            if name != "u":
                raise KeyError(f"unknown state {name!r}")
            return 0

    result = Simulation(strand, Renamed(), dt=DT).run(1 * ms, record_every=1 * ms)

    assert result.v.shape == (6, 2)


def test_an_unknown_recording_name_is_refused(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(ValueError, match="cannot be recorded"):
        simulation.run(1 * ms, record_every=1 * ms, record=("v", "calcium"))


def test_the_extracellular_potential_is_recorded_for_knm(varied_sheet):
    simulation = Simulation(varied_sheet, fitzhugh_nagumo(), variant=Variant.KNM, dt=DT)
    stimulated(simulation, cells=[0])

    result = simulation.run(2 * ms, record_every=1 * ms, record=("v", "u_e"))

    assert result.u_e.shape == (12, 3)
    # The solve produces the extracellular potential, so the initial sample has none to show.
    assert np.isnan(result.u_e[:, 0]).all()
    assert np.isfinite(result.u_e[:, 1:]).all()
    assert np.abs(result.u_e[:, 1:]).max() > 0.0


def test_the_extracellular_potential_is_zero_for_sknm_ue0(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), variant=Variant.SKNM_UE0, dt=DT)

    result = simulation.run(1 * ms, record_every=1 * ms, record=("u_e",))

    np.testing.assert_array_equal(result.u_e[:, 1:], 0.0)


def test_the_extracellular_potential_is_unavailable_for_sknm(strand):
    """SKNM eliminates it rather than solving for it, so there is nothing to report."""
    simulation = Simulation(strand, fitzhugh_nagumo(), variant=Variant.SKNM, dt=DT)

    result = simulation.run(1 * ms, record_every=1 * ms, record=("v", "u_e"))

    assert result.u_e is None
    assert "u_e" not in result


def test_a_callback_can_stop_a_run_early(strand):
    simulation = stimulated(Simulation(strand, fitzhugh_nagumo(), dt=DT))

    def stop_once_the_far_end_fires(simulation):
        return bool(simulation.v[-1] > 0.0)

    result = simulation.run(200 * ms, record_every=1 * ms, callback=stop_once_the_far_end_fires)

    assert simulation.t < 200.0
    assert result.t[-1] <= simulation.t
    assert simulation.v[-1] > 0.0


def test_a_callback_that_never_stops_runs_to_the_end(strand):
    seen = []
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    simulation.run(1 * ms, record_every=1 * ms, callback=lambda sim: seen.append(sim.t))

    assert len(seen) == 50


def test_a_result_is_frozen_and_its_traces_are_read_only(strand):
    result = Simulation(strand, fitzhugh_nagumo(), dt=DT).run(1 * ms, record_every=1 * ms)

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.t = np.zeros(3)
    with pytest.raises(ValueError, match="read-only"):
        result.v[0, 0] = 1.0


def test_a_result_reports_its_shape_and_variant(strand):
    result = Simulation(strand, fitzhugh_nagumo(), variant=Variant.KNM, dt=DT).run(
        2 * ms, record_every=1 * ms
    )

    assert result.n_cells == 6
    assert result.n_samples == 3
    assert result.variant is Variant.KNM
    assert "KNM" in repr(result)


def test_asking_a_result_for_something_it_did_not_record_says_what_it_did(strand):
    result = Simulation(strand, fitzhugh_nagumo(), dt=DT).run(1 * ms, record_every=1 * ms)

    with pytest.raises(KeyError, match="'v'"):
        result["w"]


# --- What it refuses to do -----------------------------------------------------------------


def test_the_time_step_cannot_be_rebound(strand):
    """The operator is assembled around `dt`; changing it afterwards would not reassemble it."""
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(AttributeError):
        simulation.dt = 0.01


def test_strang_splitting_is_refused_with_a_reason(strand):
    with pytest.raises(NotImplementedError, match="Godunov"):
        Simulation(strand, fitzhugh_nagumo(), dt=DT, splitting="strang")


def test_an_unknown_splitting_is_refused(strand):
    with pytest.raises(ValueError, match="splitting must be one of"):
        Simulation(strand, fitzhugh_nagumo(), dt=DT, splitting="symplectic")


def test_a_capacitance_mismatch_raises(strand):
    """Not a warning: the symptom is a plausible wave at the wrong speed, and a warning
    scrolls past unseen in a sweep of dozens of runs."""
    with pytest.raises(ValueError, match="capacitance"):
        Simulation(strand, DeclaredCapacitance(2.0), dt=DT)


def test_a_matching_capacitance_is_accepted(strand):
    """A model's capacitance is absolute, so it matches `Cm` times the cell's membrane area."""
    absolute = strand.Cm * strand.membrane_area[0]
    simulation = Simulation(strand, DeclaredCapacitance(absolute), dt=DT)

    assert simulation.model.capacitance == absolute


def test_a_capacitance_matching_the_specific_one_is_a_mismatch(strand):
    """The two conventions differ by the membrane area, which is five orders of magnitude.

    Comparing a declared absolute capacitance against the network's specific one would accept
    a model whose voltage equation runs at 1e5 times the right capacitance.
    """
    with pytest.raises(ValueError, match="capacitance"):
        Simulation(strand, DeclaredCapacitance(strand.Cm), dt=DT)


def test_the_capacitance_is_checked_against_every_cell(varied_sheet):
    """Per cell, not against one representative area.

    A network may give each cell its own membrane area, and one membrane model serves all of
    them, so a single declared capacitance can only be right if every cell agrees with it.
    """
    absolute = varied_sheet.Cm * varied_sheet.membrane_area[0]

    with pytest.raises(ValueError, match="capacitance"):
        Simulation(varied_sheet, DeclaredCapacitance(absolute), dt=DT)


def test_a_uniform_area_network_accepts_the_capacitance_it_implies(strand):
    """The companion to the test above: identical areas, so one capacitance covers them all."""
    areas = np.unique(strand.membrane_area)
    assert areas.size == 1

    simulation = Simulation(strand, DeclaredCapacitance(strand.Cm * areas[0]), dt=DT)

    assert simulation.model.capacitance == pytest.approx(strand.Cm * areas[0])


def test_the_capacitance_check_can_be_bypassed(strand):
    simulation = Simulation(strand, DeclaredCapacitance(2.0), dt=DT, check_capacitance=False)

    assert simulation.model.capacitance == 2.0


def test_a_model_that_declares_no_capacitance_is_not_a_mismatch(strand):
    """`None` means the model's voltage equation does not expose one, which cannot be checked."""
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    assert simulation.model.capacitance is None


def test_a_run_that_ends_before_it_starts_is_refused(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    simulation.run(5 * ms, record_every=1 * ms)

    with pytest.raises(ValueError, match="t_end"):
        simulation.run(1 * ms, record_every=1 * ms)


def test_a_non_positive_recording_interval_is_refused(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(ValueError, match="record_every must be positive"):
        simulation.run(1 * ms, record_every=0 * ms)


# --- Units ---------------------------------------------------------------------------------


def test_dt_must_carry_a_unit(strand):
    with pytest.raises(TypeError, match="dt must be a quantity"):
        Simulation(strand, fitzhugh_nagumo(), dt=0.02)


def test_t_end_must_carry_a_unit(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(TypeError, match="t_end must be a quantity"):
        simulation.run(10.0)


def test_record_every_must_carry_a_unit(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(TypeError, match="record_every must be a quantity"):
        simulation.run(10 * ms, record_every=1.0)


def test_a_time_in_other_units_is_converted_not_assumed(strand):
    from sknm.units import s, us

    in_milliseconds = Simulation(strand, fitzhugh_nagumo(), dt=20 * us).run(
        1 * ms, record_every=0.5 * ms
    )
    in_seconds = Simulation(strand, fitzhugh_nagumo(), dt=2e-5 * s).run(
        1e-3 * s, record_every=5e-4 * s
    )

    np.testing.assert_allclose(in_seconds.v, in_milliseconds.v, rtol=1e-13)
    np.testing.assert_allclose(in_seconds.t, in_milliseconds.t, atol=1e-12)


def test_t_end_must_be_a_time(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(pint.DimensionalityError):
        simulation.run(10 * um)


# --- The model the package is for ----------------------------------------------------------


def test_the_hipsc_cm_model_produces_a_propagating_action_potential():
    """The one test that runs the model this package exists to run.

    Its membrane potential is the twenty-fifth of twenty-five states, which is the case every
    other test here has to construct deliberately, and its numbers are millivolts rather than
    the dimensionless ones FitzHugh-Nagumo works in.
    """
    model = from_gotranx(base_model_IM)
    network = chain(10, membrane_area=MEMBRANE_AREA, **geometry())
    simulation = Simulation(network, model, dt=DT)
    simulation.set_parameter("stim_amplitude", 20.0, cells=[0, 1])

    result = simulation.run(20 * ms, record_every=0.1 * ms)

    assert model.v_index == 24
    assert result.v[:, 0] == pytest.approx(-76.3, abs=0.5)
    assert result.v.max() > 0.0
    # Ten cells with two of them stimulated are within an electrotonic length of each other,
    # so this is a depolarization spreading rather than a free-running wave; it says the far
    # cell follows the near one, and nothing about conduction velocity.
    crossings = [int(np.argmax(trace > 0.0)) for trace in result.v]
    assert crossings == sorted(crossings)
    assert crossings[0] < crossings[-1]


# --- The rest of the surface ---------------------------------------------------------------


def test_the_operator_is_reachable_without_reaching_into_privates(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), variant=Variant.KNM, dt=DT)

    assert simulation.operator.variant is Variant.KNM
    assert simulation.operator.n_dofs == 2 * strand.n_cells - 1


def test_the_operator_cannot_be_rebound(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), dt=DT)

    with pytest.raises(AttributeError):
        simulation.operator = None


def test_a_variant_can_be_named_by_its_value(strand):
    assert Simulation(strand, fitzhugh_nagumo(), variant="knm", dt=DT).variant is Variant.KNM


def test_an_explicit_ground_is_passed_to_the_assembly(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), variant=Variant.KNM, dt=DT, ground=2)

    np.testing.assert_array_equal(simulation.operator.ground, [2])


def test_a_simulation_reports_what_it_is(strand):
    simulation = Simulation(strand, fitzhugh_nagumo(), variant=Variant.SKNM, dt=DT)

    assert "SKNM" in repr(simulation)
    assert "6" in repr(simulation)


def test_the_capacitance_comes_from_the_network(strand):
    doubled = chain(6, membrane_area=MEMBRANE_AREA, Cm=2.0 * uF / cm**2, **geometry())

    single = Simulation(strand, fitzhugh_nagumo(), dt=DT)
    twice = Simulation(doubled, fitzhugh_nagumo(), dt=DT)

    np.testing.assert_allclose(twice.operator.D, 2 * single.operator.D, rtol=1e-14)
