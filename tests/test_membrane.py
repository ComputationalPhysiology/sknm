"""The membrane seam: the `MembraneModel` protocol and the models that satisfy it.

Every model is exercised through the same parametrized cases, so a new model only has to be
added to `MODELS` to inherit the whole contract.
"""

import numpy as np
import pytest

from sknm.membrane import MembraneModel, base_model_IM, fitzhugh_nagumo, from_gotranx

MODELS = pytest.mark.parametrize(
    ("make_model", "stimulus_parameter"),
    [
        pytest.param(lambda: from_gotranx(base_model_IM), "stim_amplitude", id="base_model_IM"),
        pytest.param(fitzhugh_nagumo, "stim_amplitude", id="fitzhugh_nagumo"),
    ],
)


@MODELS
def test_model_satisfies_the_protocol(make_model, stimulus_parameter):
    assert isinstance(make_model(), MembraneModel)


@MODELS
def test_v_index_agrees_with_state_index(make_model, stimulus_parameter):
    model = make_model()
    assert model.v_index == model.state_index(model.v_name)
    assert 0 <= model.v_index < model.num_states


@MODELS
def test_initial_arrays_are_state_major_and_float64(make_model, stimulus_parameter):
    """State-major `(n_states, n_cells)` is mandatory.

    The generated code accepts a cell-major array without raising and returns garbage, so
    nothing downstream would catch the layout being wrong.
    """
    model = make_model()
    states = model.initial_states(7)
    params = model.initial_parameters(7)
    assert states.shape == (model.num_states, 7)
    assert params.shape == (model.num_parameters, 7)
    assert states.dtype == np.float64
    assert params.dtype == np.float64


@MODELS
def test_step_is_functional_not_in_place(make_model, stimulus_parameter):
    """`step` returns a new array and leaves its input untouched.

    Callers rely on this to rebind rather than copy.
    """
    model = make_model()
    states = model.initial_states(3)
    params = model.initial_parameters(3)
    before = states.copy()
    returned = model.step(states, 0.0, 0.01, params)
    assert returned is not states
    np.testing.assert_array_equal(states, before)


@MODELS
def test_step_is_vectorised_over_cells(make_model, stimulus_parameter):
    """N identical cells must step identically to one cell: this is the property that

    makes the whole struct-of-arrays design work.
    """
    model = make_model()
    one = model.step(model.initial_states(1), 0.0, 0.01, model.initial_parameters(1))
    many = model.step(model.initial_states(5), 0.0, 0.01, model.initial_parameters(5))
    for cell in range(5):
        np.testing.assert_allclose(many[:, cell], one[:, 0], rtol=0, atol=0)


@MODELS
def test_per_cell_parameters_take_effect(make_model, stimulus_parameter):
    """Per-cell parameter arrays take effect.

    Applying a stimulus to part of a network is exactly this operation, so the per-cell array
    is load-bearing rather than a convenience.
    """
    model = make_model()
    states = model.initial_states(2)
    params = model.initial_parameters(2)
    params[model.parameter_index(stimulus_parameter), 0] = 20.0
    stepped = model.step(states, 0.0, 0.01, params)
    assert stepped[model.v_index, 0] != stepped[model.v_index, 1]


def test_base_model_IM_declares_no_capacitance():
    """`base_model_IM` integrates `dV/dt = -I_tot`, an implicit specific capacitance.

    Its `Cm` parameter appears only in the ion-flux conversions, never in the voltage
    equation, so there is no capacitance to declare and no mismatch a caller could check.
    """
    assert from_gotranx(base_model_IM).capacitance is None


def test_generated_module_imports_only_numpy():
    """The generated module must import nothing but numpy.

    If it grew another import, the runtime dependency set would silently stop being
    numpy + scipy.
    """
    source = __import__("inspect").getsource(base_model_IM)
    imports = {
        line.strip() for line in source.splitlines() if line.startswith(("import ", "from "))
    }
    assert imports == {"import numpy"}


def test_fitzhugh_nagumo_is_excitable():
    """A stimulated cell fires; an unstimulated one holds its resting state."""
    model = fitzhugh_nagumo()
    params = model.initial_parameters(2)
    params[model.parameter_index("stim_amplitude"), 0] = 1.0

    states = model.initial_states(2)
    v0 = states[model.v_index].copy()
    t = 0.0
    peak = v0.copy()
    for _ in range(4000):
        states = model.step(states, t, 0.01, params)
        t += 0.01
        peak = np.maximum(peak, states[model.v_index])

    assert peak[0] > v0[0] + 1.0, "stimulated cell should fire"
    assert peak[1] < v0[1] + 0.1, "unstimulated cell should stay at rest"
