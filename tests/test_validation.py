"""The published numbers, on both of the paper's setups.

Table S1 reports, for a 40x40 sheet of hiPSC-CMs at a time step of 0.02 ms, a conduction
velocity of 3.73 cm/s and a maximal upstroke velocity of 18.81 V/s, the same under KNM as under
SKNM. Table S4 reports 0.0243 cm/s and 0.340 V/s for a 15x15 sheet of pancreatic beta cells.
This file asserts both, and a coarser row of each table so that the trend towards them is
pinned as well as the values.

The two are independent evidence and not one claim twice. The beta setup runs a membrane model
with 5 states rather than 25, over twenty times as long, at a conduction velocity 150 times
slower, and it exercises a capacitance the cardiac model does not even declare.

These are deliberately not `test_<module>.py`: they are claims about the package as a whole, and
every module it has takes part in each of them. They are also the only slow tests in the suite,
which is the price of a wave that has to cross a sheet before it can be timed.

Four tolerances, all argued from measurement rather than tuned:

- **hiPSC conduction velocity, 1%.** The measurement comes out at 0.22%. Below about 0.5% the
  assertion would be pinning noise rather than behaviour: the table quotes three significant
  figures, which is 0.13% on its own, and a threshold crossing is resolved to one step at each
  end of a transit of some five hundred, which is another 0.37%.
- **hiPSC maximal upstroke velocity, 2%**, the paper's own stated accuracy at this time step.
  The measurement comes out at 1.6%, and the gap is understood: the reference post-processes a
  trace it writes *before* each step's spatial solve, so its estimate and this one bracket the
  true value and converge to it from opposite sides as the step shrinks. Sampling the potential
  mid-step to reproduce that would mean publishing an intermediate of the operator splitting as
  API, which is not worth 1.6%.
- **Beta conduction velocity, 0.5%** -- tighter than the cardiac one, because the floors are.
  The measurement comes out at 0.08%. The beta transit takes 21,000 steps rather than 500, so
  the step quantization that costs 0.37% above costs 0.005% here, and what is left is the
  table's three significant figures, 0.2% at 0.0243.
- **Beta maximal upstroke velocity, 2%.** The measurement comes out at 0.89% and converges to
  about 1.0% as the step shrinks. The pre-solve and post-solve estimators differ by only 0.06%
  on this setup, because the beta upstroke is slow, so unlike the cardiac case that gap is not
  what the tolerance is covering; it covers a residual that shrinking the step does not remove.

Only `alpha=1` is used for the cardiac sheet, because that is what the table was computed at;
the rest of the suite prefers `alpha=1.5`, where a sheet cannot confuse its two axes. A beta
cell has no anisotropy factor at all.
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

BETA_NX, BETA_NY = 15, 15
BETA_T_END = 1000 * ms

#: Table S4, pancreatic beta cells, at the time step the paper settles on. It is the only row
#: of that table this implementation can be held to: the reference substeps its membrane model
#: `round(dt / 0.02)` times, so every coarser row integrates the ODEs at a finer step than the
#: one it is labelled with, while this package takes one membrane step per network step.
PUBLISHED_BETA_CONDUCTION_VELOCITY = 0.0243
PUBLISHED_BETA_UPSTROKE_VELOCITY = 0.340


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


def measure_beta(variant, dt):
    """Run the paper's beta cell simulation and report the same two velocities.

    Returns
    -------
    tuple of (float, float)
        Conduction velocity in cm/s, and maximal upstroke velocity at the centre cell in V/s.
    """
    network = presets.beta_sheet(BETA_NX, BETA_NY)
    simulation = Simulation(network, presets.beta_membrane_model(), variant=variant, dt=dt)
    simulation.set_parameter("gkatpbar", presets.beta_stimulus_conductance(BETA_NX, BETA_NY))

    path = presets.beta_conduction_path(BETA_NX, BETA_NY)
    recorder = ActivationRecorder(
        simulation, threshold=presets.BETA_THRESHOLD, stop_when_activated=path.end
    )
    simulation.run(BETA_T_END, record=(), callback=recorder)

    velocity = conduction_velocity(recorder, path).m_as("cm / s")
    upstroke = recorder.max_upstroke_velocity[presets.beta_centre_cell(BETA_NX, BETA_NY)]
    return float(velocity), float(upstroke)


@pytest.fixture(scope="module")
def beta_sknm_at_the_published_step():
    return measure_beta(Variant.SKNM, 0.02 * ms)


@pytest.fixture(scope="module")
def beta_knm_at_the_published_step():
    return measure_beta(Variant.KNM, 0.02 * ms)


@pytest.fixture(scope="module")
def beta_sknm_ue0_at_the_published_step():
    return measure_beta(Variant.SKNM_UE0, 0.02 * ms)


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


# --- Table S4 -------------------------------------------------------------------------------


def test_sknm_reproduces_the_published_beta_conduction_velocity(beta_sknm_at_the_published_step):
    velocity, _ = beta_sknm_at_the_published_step

    assert velocity == pytest.approx(PUBLISHED_BETA_CONDUCTION_VELOCITY, rel=0.005)


def test_sknm_reproduces_the_published_beta_upstroke_velocity(beta_sknm_at_the_published_step):
    _, upstroke = beta_sknm_at_the_published_step

    assert upstroke == pytest.approx(PUBLISHED_BETA_UPSTROKE_VELOCITY, rel=0.02)


def test_knm_reproduces_the_published_beta_conduction_velocity(beta_knm_at_the_published_step):
    velocity, _ = beta_knm_at_the_published_step

    assert velocity == pytest.approx(PUBLISHED_BETA_CONDUCTION_VELOCITY, rel=0.005)


def test_knm_reproduces_the_published_beta_upstroke_velocity(beta_knm_at_the_published_step):
    _, upstroke = beta_knm_at_the_published_step

    assert upstroke == pytest.approx(PUBLISHED_BETA_UPSTROKE_VELOCITY, rel=0.02)


def test_all_three_variants_agree_on_the_beta_sheet(
    beta_sknm_at_the_published_step,
    beta_knm_at_the_published_step,
    beta_sknm_ue0_at_the_published_step,
):
    """What Figure S1 reports, and the reason for it is `lam`.

    The gap junction resistance between beta cells is a thousand times the cardiac one, so it
    swamps both conductivities: lam is around 65,000, SKNM's `lam / (1 + lam)` is 1 to five
    figures, and the three models become the same algebraic system. For hiPSC-CMs, where lam is
    about 40, SKNM(ue=0) is 12% out at a small extracellular volume.
    """
    np.testing.assert_allclose(
        beta_knm_at_the_published_step, beta_sknm_at_the_published_step, rtol=1e-9
    )
    np.testing.assert_allclose(
        beta_knm_at_the_published_step, beta_sknm_ue0_at_the_published_step, rtol=1e-4
    )


def test_the_beta_conductance_ratio_is_what_makes_the_three_variants_coincide():
    """The companion to the test above: the agreement is a property of the network.

    Asserted on the network rather than on a run, so that a change making the variants agree
    for some other reason cannot pass it.
    """
    beta = presets.beta_sheet(BETA_NX, BETA_NY)
    cardiac = presets.hipsc_sheet(NX, NY, alpha=1.0)

    assert beta.lam > 1e4
    assert beta.lam / (1 + beta.lam) == pytest.approx(1.0, abs=1e-4)
    assert cardiac.lam < 100
    assert cardiac.lam / (1 + cardiac.lam) < 0.98


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
