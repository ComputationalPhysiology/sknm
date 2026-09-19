"""The published numbers.

Table S1 of the supplementary reports, for a 40x40 sheet of hiPSC-CMs at a time step of
0.02 ms, a conduction velocity of 3.73 cm/s and a maximal upstroke velocity of 18.81 V/s, the
same under KNM as under SKNM. This file asserts them, and one row further up the table so that
the trend towards them is pinned as well as the values.

These are deliberately not `test_<module>.py`: they are claims about the package as a whole, and
every module it has takes part in each of them. They are also the only slow tests in the suite,
about five seconds between them, which is the price of a wave that has to travel twenty-five
cells before it can be timed.

Two tolerances, both argued rather than tuned:

- **Conduction velocity, 1%.** The measurement comes out at 0.22%. Below about 0.5% the
  assertion would be pinning noise rather than behaviour: the table quotes three significant
  figures, which is 0.13% on its own, and a threshold crossing is resolved to one step at each
  end of a transit of some five hundred, which is another 0.37%.
- **Maximal upstroke velocity, 2%**, the paper's own stated accuracy at this time step. The
  measurement comes out at 1.6%, and the gap is understood: the reference post-processes a trace
  it writes *before* each step's spatial solve, so its estimate and this one bracket the true
  value and converge to it from opposite sides as the step shrinks. Sampling the potential
  mid-step to reproduce that would mean publishing an intermediate of the operator splitting as
  API, which is not worth 1.6%.

Only `alpha=1` is used here, because that is what the table was computed at; the rest of the
suite prefers `alpha=1.5`, where a sheet cannot confuse its two axes.
"""

import numpy as np
import pytest

from sknm import Simulation, Variant, chain, presets
from sknm.analysis import ActivationRecorder, conduction_velocity
from sknm.membrane import base_model_IM, fitzhugh_nagumo, from_gotranx
from sknm.units import mS, ms, mV

NX, NY = 40, 40
T_END = 50 * ms
THRESHOLD = -20 * mV

#: Table S1, hiPSC-CMs, at the time step the paper settles on.
PUBLISHED_CONDUCTION_VELOCITY = 3.73
PUBLISHED_UPSTROKE_VELOCITY = 18.81
#: Table S1 one row coarser, where both quantities are still converging.
PUBLISHED_CONDUCTION_VELOCITY_AT_A_COARSER_STEP = 3.60
#: The finest step in Table S1, which both estimators are converging towards.
CONVERGED_CONDUCTION_VELOCITY = 3.76


def measure(variant, dt):
    """Run the paper's simulation and report its conduction and upstroke velocities.

    Returns
    -------
    tuple of (float, float)
        Conduction velocity in cm/s, and maximal upstroke velocity at the centre cell in V/s.
    """
    network = presets.hipsc_sheet(NX, NY, alpha=1.0)
    simulation = Simulation(network, from_gotranx(base_model_IM), variant=variant, dt=dt)
    simulation.set_parameter("stim_amplitude", presets.hipsc_stimulus_amplitude(NX, NY))

    path = presets.hipsc_conduction_path(NX, NY, alpha=1.0)
    recorder = ActivationRecorder(simulation, threshold=THRESHOLD, stop_when_activated=path.end)
    simulation.run(T_END, record=(), callback=recorder)

    velocity = conduction_velocity(recorder, path).m_as("cm / s")
    upstroke = recorder.max_upstroke_velocity[presets.hipsc_centre_cell(NX, NY)]
    return float(velocity), float(upstroke)


@pytest.fixture(scope="module")
def sknm_at_the_published_step():
    return measure(Variant.SKNM, 0.02 * ms)


@pytest.fixture(scope="module")
def knm_at_the_published_step():
    return measure(Variant.KNM, 0.02 * ms)


@pytest.fixture(scope="module")
def sknm_at_a_coarser_step():
    return measure(Variant.SKNM, 0.1 * ms)


# --- Table S1 -------------------------------------------------------------------------------


def test_sknm_reproduces_the_published_conduction_velocity(sknm_at_the_published_step):
    velocity, _ = sknm_at_the_published_step

    assert velocity == pytest.approx(PUBLISHED_CONDUCTION_VELOCITY, rel=0.01)


def test_sknm_reproduces_the_published_upstroke_velocity(sknm_at_the_published_step):
    _, upstroke = sknm_at_the_published_step

    assert upstroke == pytest.approx(PUBLISHED_UPSTROKE_VELOCITY, rel=0.02)


def test_knm_reproduces_the_published_conduction_velocity(knm_at_the_published_step):
    velocity, _ = knm_at_the_published_step

    assert velocity == pytest.approx(PUBLISHED_CONDUCTION_VELOCITY, rel=0.01)


def test_knm_reproduces_the_published_upstroke_velocity(knm_at_the_published_step):
    _, upstroke = knm_at_the_published_step

    assert upstroke == pytest.approx(PUBLISHED_UPSTROKE_VELOCITY, rel=0.02)


def test_the_two_models_agree_on_the_paper_s_sheet(
    sknm_at_the_published_step, knm_at_the_published_step
):
    """Table S1 reports the same three figures for both, which is the paper's central claim."""
    np.testing.assert_allclose(knm_at_the_published_step, sknm_at_the_published_step, rtol=1e-9)


# --- Convergence in the time step -------------------------------------------------------------


def test_a_coarser_step_reproduces_its_own_row_of_the_table(sknm_at_a_coarser_step):
    velocity, _ = sknm_at_a_coarser_step

    assert velocity == pytest.approx(PUBLISHED_CONDUCTION_VELOCITY_AT_A_COARSER_STEP, rel=0.01)


def test_halving_the_step_moves_towards_the_converged_velocity(
    sknm_at_the_published_step, sknm_at_a_coarser_step
):
    """The published step is not a resolution the answer happens to be right at."""
    fine, _ = sknm_at_the_published_step
    coarse, _ = sknm_at_a_coarser_step

    assert coarse < fine < CONVERGED_CONDUCTION_VELOCITY


# --- The simplification itself ----------------------------------------------------------------


def uniform_chain(n_cells=6):
    """A chain whose connections all carry the same conductances, so that Ge = lam * Gi holds."""
    lx, ly = presets.CELL_DIMENSIONS[1.5]
    return chain(
        n_cells,
        lx=lx,
        ly=ly,
        lz=(1 + presets.DELTA_E) * ly,
        delta_e=presets.DELTA_E,
        sigma_i=presets.SIGMA_I,
        sigma_e=presets.SIGMA_E,
        Gg=presets.GAP_JUNCTION_CONDUCTANCE,
        membrane_area=presets.MEMBRANE_AREA,
        Cm=presets.CM,
    )


def test_lambda_is_the_exact_conductance_ratio_of_a_uniform_network():
    """Equation (30) fits Ge = lam * Gi over the whole network; where every connection is the
    same, the fit is exact rather than a least-squares compromise."""
    network = uniform_chain()

    np.testing.assert_allclose(network.Ge, network.lam * network.Gi, rtol=1e-12)


def test_sknm_is_knm_where_the_conductance_ratio_is_exact():
    """The paper's Figure 2 claim, on trajectories rather than on a single summary number.

    Eliminating the extracellular potential from the KNM block system leaves the intracellular
    coupling reduced by lam/(1 + lam), which is what SKNM solves. Where Ge = lam * Gi holds
    connection by connection, that elimination is exact and the two must agree to the solver's
    tolerance at every step, not merely close.
    """
    network = uniform_chain()

    traces = []
    for variant in (Variant.KNM, Variant.SKNM):
        simulation = Simulation(network, fitzhugh_nagumo(), variant=variant, dt=0.02 * ms)
        simulation.set_parameter("stim_amplitude", 1.0, cells=[0])
        traces.append(simulation.run(40 * ms, record_every=0.02 * ms).v)

    assert traces[0].max() > 1.0, "a wave has to have happened for the agreement to mean anything"
    np.testing.assert_allclose(traces[0], traces[1], rtol=1e-9, atol=1e-12)


def test_sknm_differs_from_knm_where_the_conductance_ratio_is_not_exact():
    """The converse, so the agreement above is a property of the network and not of the code."""
    network = uniform_chain()
    spread = network.Gg * np.linspace(0.2, 5.0, network.n_connections)
    varied = network.with_conductances(spread * mS)

    traces = []
    for variant in (Variant.KNM, Variant.SKNM):
        simulation = Simulation(varied, fitzhugh_nagumo(), variant=variant, dt=0.02 * ms)
        simulation.set_parameter("stim_amplitude", 1.0, cells=[0])
        traces.append(simulation.run(40 * ms, record_every=0.02 * ms).v)

    assert not np.allclose(traces[0], traces[1], rtol=1e-6)
