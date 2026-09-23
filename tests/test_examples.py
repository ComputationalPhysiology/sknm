"""The machinery the figure scripts share, which is where a figure can go quietly wrong.

The scripts themselves are checked by running them; what cannot be checked that way is the
bookkeeping underneath. Two things here can produce a plausible figure made of wrong numbers:
a cache key that leaves out something the result depends on, so a point measured under one set
of parameters is served under another; and a reduced sample that is not a subset of the full
one, so the fast figure and the full figure disagree at the same marker.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from sknm import Variant

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def _load_common():
    """Import `examples/common.py`, which is a script's neighbour rather than a package."""
    spec = importlib.util.spec_from_file_location(
        "sknm_examples_common", EXAMPLES_DIR / "common.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


common = _load_common()

FULL = common.Options(full=True)
FAST = common.Options(full=False)


# --------------------------------------------------------------------------------------
# the options the environment carries
#
# The scripts are run both as scripts and as notebooks, and a notebook's process is the
# kernel's, whose command line belongs to the kernel. Everything a run can be asked for
# therefore arrives through the environment, where both ways of running can reach it.
# --------------------------------------------------------------------------------------


def test_an_unconfigured_run_is_fast_cached_and_writes_to_the_figure_directory(monkeypatch):
    for name in common.OPTION_VARIABLES:
        monkeypatch.delenv(name, raising=False)
    assert common.options() == common.Options(
        full=False, no_cache=False, output_dir=common.FIGURE_DIR
    )


def test_options_asked_for_nothing_in_particular_are_the_same_fast_cached_run():
    """`Options()` is what a script gets before the environment is consulted at all."""
    assert common.Options() == common.Options(
        full=False, no_cache=False, output_dir=common.FIGURE_DIR
    )


def test_the_full_sweep_is_asked_for_through_the_environment(monkeypatch):
    monkeypatch.setenv("SKNM_EXAMPLES_FULL", "1")
    assert common.options().full is True


def test_the_cache_is_disabled_through_the_environment(monkeypatch):
    monkeypatch.setenv("SKNM_EXAMPLES_NO_CACHE", "1")
    assert common.options().no_cache is True


def test_the_output_directory_is_taken_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SKNM_EXAMPLES_OUTPUT_DIR", str(tmp_path / "elsewhere"))
    assert common.options().output_dir == tmp_path / "elsewhere"


def test_an_empty_output_directory_means_the_default(monkeypatch):
    """An exported-but-empty variable is how a shell spells "unset", and must read as one."""
    monkeypatch.setenv("SKNM_EXAMPLES_OUTPUT_DIR", "")
    assert common.options().output_dir == common.FIGURE_DIR


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", "yes", "on"])
def test_a_flag_is_set_by_any_of_the_usual_spellings(monkeypatch, value):
    monkeypatch.setenv("SKNM_EXAMPLES_FULL", value)
    assert common.options().full is True


@pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", ""])
def test_a_flag_is_cleared_by_any_of_the_usual_spellings(monkeypatch, value):
    monkeypatch.setenv("SKNM_EXAMPLES_FULL", value)
    assert common.options().full is False


@pytest.mark.parametrize("name", ["SKNM_EXAMPLES_FULL", "SKNM_EXAMPLES_NO_CACHE"])
def test_a_flag_that_cannot_be_read_as_a_boolean_is_refused(monkeypatch, name):
    """Silently reading a misspelling as false would run the wrong sweep and say nothing."""
    monkeypatch.setenv(name, "flase")
    with pytest.raises(ValueError, match=name):
        common.options()


def test_the_refusal_quotes_the_value_it_could_not_read(monkeypatch):
    monkeypatch.setenv("SKNM_EXAMPLES_FULL", "sometimes")
    with pytest.raises(ValueError, match="sometimes"):
        common.options()


@pytest.mark.parametrize("value", [" 1 ", "1\n", "\ttrue "])
def test_a_flag_is_read_through_the_whitespace_a_shell_leaves_behind(monkeypatch, value):
    monkeypatch.setenv("SKNM_EXAMPLES_FULL", value)
    assert common.options().full is True


def test_an_output_directory_of_only_whitespace_means_the_default(monkeypatch):
    monkeypatch.setenv("SKNM_EXAMPLES_OUTPUT_DIR", "   ")
    assert common.options().output_dir == common.FIGURE_DIR


def test_every_option_has_a_variable_that_sets_it(monkeypatch):
    """The names the scripts document are the names `options` reads, with nothing left over."""
    assert set(common.OPTION_VARIABLES) == {
        "SKNM_EXAMPLES_FULL",
        "SKNM_EXAMPLES_NO_CACHE",
        "SKNM_EXAMPLES_OUTPUT_DIR",
    }


# --------------------------------------------------------------------------------------
# the reduced sample
# --------------------------------------------------------------------------------------


def test_the_full_sample_is_the_whole_sweep():
    assert common.sample(common.ANISOTROPY_FACTORS, FULL) == common.ANISOTROPY_FACTORS
    assert common.sample(common.GAP_JUNCTION_VARIATIONS, FULL) == common.GAP_JUNCTION_VARIATIONS


@pytest.mark.parametrize(
    "points",
    [
        common.ANISOTROPY_FACTORS,
        common.GAP_JUNCTION_VARIATIONS,
        common.VOLUME_FRACTIONS,
        common.EXTRACELLULAR_FRACTIONS,
        (1.0, 2.0),
        (1.0, 2.0, 3.0),
        (1.0, 2.0, 3.0, 4.0),
    ],
)
def test_a_reduced_sample_is_a_subset_of_the_full_one(points):
    """A fast figure must plot the same values a full figure does, only fewer of them."""
    fewer = common.sample(points, FAST)
    assert set(fewer) <= set(points)
    assert list(fewer) == sorted(fewer, key=points.index)


@pytest.mark.parametrize(
    "points",
    [common.ANISOTROPY_FACTORS, common.GAP_JUNCTION_VARIATIONS, common.VOLUME_FRACTIONS],
)
def test_a_reduced_sample_keeps_both_ends_of_the_sweep(points):
    """The ends carry the message: uniform coupling at one end, the widest spread at the other."""
    fewer = common.sample(points, FAST)
    assert fewer[0] == points[0]
    assert fewer[-1] == points[-1]


@pytest.mark.parametrize(
    "points",
    [common.ANISOTROPY_FACTORS, common.GAP_JUNCTION_VARIATIONS, common.VOLUME_FRACTIONS],
)
def test_a_reduced_sample_is_shorter_but_still_a_curve(points):
    fewer = common.sample(points, FAST)
    assert 3 <= len(fewer) < len(points)


def test_an_even_length_sweep_keeps_its_last_point():
    """Taking every other point of an even-length sweep would otherwise drop the far end."""
    assert common.sample((0.0, 0.2, 0.4, 0.6, 0.8, 1.0), FAST) == (0.0, 0.4, 0.8, 1.0)


def test_an_odd_length_sweep_is_not_given_its_last_point_twice():
    assert common.sample((1.0, 1.5, 2.0, 3.0, 4.0), FAST) == (1.0, 2.0, 4.0)


# --------------------------------------------------------------------------------------
# the snapshots the wave figures are drawn from
#
# Three scripts draw a sheet at a series of fixed times, and each is a linear page rather
# than an importable module, so what they share lives here rather than in one of them.
# --------------------------------------------------------------------------------------

SMALL = dict(nx=3, ny=3, t_end=10.0)

#: `snapshot_figure` is the only thing in this module that draws, and matplotlib arrives with
#: the `examples` extra rather than `test`. Everything else here -- the sweeps, the cache keys
#: and the snapshot arithmetic -- is checked on a machine without it.
needs_matplotlib = pytest.mark.skipif(
    importlib.util.find_spec("matplotlib") is None,
    reason="matplotlib is an examples extra",
)


def test_a_snapshot_is_recorded_for_every_time_asked_for():
    setup = common.Setup(**SMALL)

    frames = common.snapshots(setup, Variant.KNM, (2.0, 4.0, 6.0), 2.0)

    assert frames.shape == (3, setup.nx * setup.ny)
    assert np.isfinite(frames).all()


def test_a_snapshot_is_taken_at_the_nearest_recorded_moment():
    """A time given with floating point slop still resolves to the sample it means.

    The last time asked for sets how long the run is, so it is held fixed and only the
    first one is nudged, by less than half a step.
    """
    setup = common.Setup(**SMALL)

    nudged = common.snapshots(setup, Variant.KNM, (4.0 + 0.4 * setup.dt, 6.0), 2.0)
    exact = common.snapshots(setup, Variant.KNM, (4.0, 6.0), 2.0)

    np.testing.assert_array_equal(nudged[0], exact[0])


def test_the_snapshots_come_back_in_the_order_they_were_asked_for():
    """Asked for in descending order, so that sorting the moments would be visible."""
    setup = common.Setup(**SMALL)

    ascending = common.snapshots(setup, Variant.KNM, (2.0, 6.0), 2.0)
    descending = common.snapshots(setup, Variant.KNM, (6.0, 2.0), 2.0)

    np.testing.assert_array_equal(ascending[0], descending[1])
    np.testing.assert_array_equal(ascending[1], descending[0])
    assert not np.array_equal(ascending[0], ascending[1])


def test_a_moment_no_sample_lands_on_is_refused():
    """A recording interval that does not divide the times asked for would draw panels
    labelled with one moment and computed at another, which nothing downstream could see."""
    setup = common.Setup(**SMALL)

    with pytest.raises(ValueError, match=r"3\.0"):
        common.snapshots(setup, Variant.KNM, (3.0, 6.0), 4.0)


def test_the_snapshot_report_prints_a_row_for_every_moment(capsys):
    setup = common.Setup(**SMALL)
    times = (2.0, 4.0)
    recorded = {v: common.snapshots(setup, v, times, 2.0) for v in (Variant.KNM, Variant.SKNM)}

    common.snapshot_report(recorded, ["2 ms", "4 ms"], (Variant.KNM, Variant.SKNM), "t")

    printed = capsys.readouterr().out
    assert printed.count("2 ms") == 1
    assert printed.count("4 ms") == 1


def test_the_snapshot_report_names_the_models_in_the_order_it_was_given_them(capsys):
    setup = common.Setup(**SMALL)
    recorded = {v: common.snapshots(setup, v, (2.0,), 2.0) for v in (Variant.KNM, Variant.SKNM)}

    common.snapshot_report(recorded, ["2 ms"], (Variant.SKNM, Variant.KNM), "t")

    header = capsys.readouterr().out.splitlines()[0]
    assert header.index("SKNM min") < header.index("KNM min")
    assert "max |SKNM - KNM|" in header


def test_the_snapshot_report_measures_one_model_against_the_other(capsys):
    """Differencing a model with itself would print a column of exact zeros and read as
    perfect agreement."""
    setup = common.Setup(**SMALL)
    variants = (Variant.KNM, Variant.SKNM)
    recorded = {v: common.snapshots(setup, v, (2.0, 4.0), 2.0) for v in variants}
    recorded[Variant.SKNM] = recorded[Variant.SKNM] + 1.0

    common.snapshot_report(recorded, ["2 ms", "4 ms"], variants, "t")

    assert "0.00e+00" not in capsys.readouterr().out


@needs_matplotlib
def test_the_snapshot_figure_has_a_panel_for_every_model_and_moment():
    setup = common.Setup(**SMALL)
    titles = ["2 ms", "4 ms", "6 ms"]
    variants = (Variant.KNM, Variant.SKNM)
    recorded = {v: common.snapshots(setup, v, (2.0, 4.0, 6.0), 2.0) for v in variants}

    figure = common.snapshot_figure(setup, recorded, variants, titles)

    # Two rows of three panels, plus the axis the shared colour scale is drawn into.
    assert len(figure.axes) == len(variants) * len(titles) + 1


@needs_matplotlib
def test_the_snapshot_figure_puts_a_moment_in_every_column_and_a_model_in_every_row():
    """A transposed grid has the same number of panels, so only their labelling shows it."""
    setup = common.Setup(**SMALL)
    titles = ["2 ms", "4 ms", "6 ms"]
    variants = (Variant.KNM, Variant.SKNM)
    recorded = {v: common.snapshots(setup, v, (2.0, 4.0, 6.0), 2.0) for v in variants}

    figure = common.snapshot_figure(setup, recorded, variants, titles)

    assert [a.get_title() for a in figure.axes if a.get_title()] == titles


@needs_matplotlib
def test_every_panel_of_a_snapshot_figure_is_on_one_colour_scale():
    """Drawn to their own ranges, two sheets differing by a rounding error look different."""
    setup = common.Setup(**SMALL)
    variants = (Variant.KNM, Variant.SKNM)
    recorded = {v: common.snapshots(setup, v, (2.0, 6.0), 2.0) for v in variants}

    figure = common.snapshot_figure(setup, recorded, variants, ["2 ms", "6 ms"])

    scales = {image.get_clim() for axis in figure.axes for image in axis.images}
    assert len(scales) == 1


# --------------------------------------------------------------------------------------
# the cache key
# --------------------------------------------------------------------------------------


def test_the_label_names_every_field_of_the_setup():
    """The guard against a cache key that has fallen behind the parameters it stands for."""
    setup = common.Setup()
    label = setup.label()
    for field in setup.__dataclass_fields__:
        assert f"{field}=" in label


def test_setups_differing_in_any_one_field_get_different_labels():
    import dataclasses

    base = common.Setup()
    changed = {
        "alpha": 2.0,
        "delta_e": 0.5,
        "gamma": 1.0,
        "nx": 36,
        "ny": 20,
        "dt": 0.1,
        "t_end": 80.0,
        "threshold": -30.0,
        "seed": 1,
    }
    assert set(changed) == set(base.__dataclass_fields__)
    for field, value in changed.items():
        assert dataclasses.replace(base, **{field: value}).label() != base.label()


def test_equal_setups_get_equal_labels():
    assert common.Setup(alpha=1.5).label(variant="knm") == common.Setup(alpha=1.5).label(
        variant="knm"
    )


def test_the_label_separates_floats_that_print_the_same_way():
    """`str` would round these together; the key has to tell them apart."""
    assert common.Setup(alpha=0.1).label() != common.Setup(alpha=0.1 + 1e-17).label()


def test_the_label_distinguishes_the_variants():
    setup = common.Setup()
    labels = {setup.label(variant=variant) for variant in Variant}
    assert len(labels) == len(Variant)


def test_the_label_writes_a_variant_by_name():
    assert "variant=knm" in common.Setup().label(variant=Variant.KNM)


# --------------------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------------------


def test_a_value_is_measured_once_and_then_read_back(tmp_path):
    cache = common.ResultCache("sweep", directory=tmp_path)
    calls = []

    def produce():
        calls.append(1)
        return 3.7383

    assert cache.compute("a", produce) == pytest.approx(3.7383)
    assert cache.compute("a", produce) == pytest.approx(3.7383)
    assert len(calls) == 1


def test_a_later_run_reuses_what_an_earlier_one_measured(tmp_path):
    common.ResultCache("sweep", directory=tmp_path).compute("a", lambda: 3.7383)
    reopened = common.ResultCache("sweep", directory=tmp_path)
    assert reopened.compute("a", _refuse) == pytest.approx(3.7383)


def test_a_cached_value_comes_back_exactly_as_it_was_measured(tmp_path):
    """The scripts print what they plot, to be read against the published tables, so a
    cached number has to be the number and not a near one."""
    measured = 3.738317757009346
    common.ResultCache("sweep", directory=tmp_path).compute("a", lambda: measured)
    reopened = common.ResultCache("sweep", directory=tmp_path)
    assert float(reopened.compute("a", _refuse)) == measured


def test_an_interrupted_sweep_resumes_from_what_it_finished(tmp_path):
    """Written after each point, not at the end, so two minutes of solving survives a Ctrl-C.

    The labels are measured out of alphabetical order deliberately. They are stored as one
    array and the values as another, so anything that reorders one of them and not the other
    hands back the right number under the wrong parameters -- which looks like a figure, not
    like a failure.
    """
    measured = {"gamma=1.0": 1.0, "beta=0.0": 2.0, "alpha=0.5": 3.0}
    cache = common.ResultCache("sweep", directory=tmp_path)
    for label, value in measured.items():
        cache.compute(label, lambda v=value: v)
    del cache

    resumed = common.ResultCache("sweep", directory=tmp_path)
    for label, value in measured.items():
        assert resumed.compute(label, _refuse) == pytest.approx(value)
    assert resumed.compute("delta=2.0", lambda: 4.0) == pytest.approx(4.0)


def test_a_disabled_cache_writes_nothing(tmp_path):
    """It still holds what this run measured -- a value this code just produced cannot be
    stale -- but it leaves nothing behind for the next run to find."""
    cache = common.ResultCache("sweep", enabled=False, directory=tmp_path)
    cache.compute("a", lambda: 1.0)
    assert not list(tmp_path.iterdir())


def test_a_disabled_cache_ignores_what_is_already_stored(tmp_path):
    """`--no-cache` has to mean *measure it again*, or it cannot be used to check the cache."""
    common.ResultCache("sweep", directory=tmp_path).compute("a", lambda: 1.0)
    fresh = common.ResultCache("sweep", enabled=False, directory=tmp_path)
    assert fresh.compute("a", lambda: 2.0) == pytest.approx(2.0)


def test_results_written_for_other_invariants_are_discarded(tmp_path):
    """The cache format and the library version are not in a label, so they discard the file.

    The run length and the threshold used to be here too. They are fields of a setup now, so
    they invalidate a single label rather than the whole file -- which they have to be, since
    the cardiac and beta scripts do not share either of them.
    """
    cache = common.ResultCache("sweep", directory=tmp_path)
    cache.compute("a", lambda: 1.0)
    with np.load(cache.path) as stored:
        spec = json.loads(str(stored["spec"].item()))
        labels, values = stored["labels"], stored["values"]
    spec["sknm"] = spec["sknm"] + ".1"
    np.savez(
        cache.path, spec=np.array(json.dumps(spec, sort_keys=True)), labels=labels, values=values
    )

    reopened = common.ResultCache("sweep", directory=tmp_path)
    assert reopened.compute("a", lambda: 2.0) == pytest.approx(2.0)


def test_an_array_valued_result_survives_the_round_trip(tmp_path):
    snapshot = np.linspace(-80.0, 20.0, 16)
    common.ResultCache("snapshots", directory=tmp_path).compute("a", lambda: snapshot)
    reopened = common.ResultCache("snapshots", directory=tmp_path)
    np.testing.assert_allclose(reopened.compute("a", _refuse), snapshot)


def test_two_shapes_in_one_cache_are_refused(tmp_path):
    cache = common.ResultCache("sweep", directory=tmp_path)
    cache.compute("a", lambda: 1.0)
    with pytest.raises(ValueError, match="cache of its own"):
        cache.compute("b", lambda: np.zeros(4))


def test_separate_names_do_not_share_a_file(tmp_path):
    common.ResultCache("fast", directory=tmp_path).compute("a", lambda: 1.0)
    other = common.ResultCache("full", directory=tmp_path)
    assert other.compute("a", lambda: 2.0) == pytest.approx(2.0)


# --------------------------------------------------------------------------------------
# the rest
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta_e", "expected"), [(0.5, "50% E"), (0.2, "20% E"), (0.1, "10% E"), (0.02, "2% E")]
)
def test_the_panel_titles_are_the_paper_s(delta_e, expected):
    assert common.panel_title(delta_e) == expected


def test_every_extracellular_fraction_has_a_panel_title():
    titles = {common.panel_title(fraction) for fraction in common.EXTRACELLULAR_FRACTIONS}
    assert titles == {"50% E", "20% E", "10% E", "2% E"}


def test_the_anisotropy_sample_is_the_five_published_cell_sizes():
    from sknm import presets

    assert common.ANISOTROPY_FACTORS == tuple(sorted(presets.CELL_DIMENSIONS))


def test_the_setup_builds_a_sheet_whose_conduction_path_matches_its_cells():
    """The one silent error available here: a distance measured at a different cell size."""
    setup = common.Setup(alpha=4.0, nx=40, ny=40)
    from sknm import presets

    lx, _ = presets.hipsc_cell_size(4.0)
    assert setup.conduction_path().distance == pytest.approx(25 * lx.m_as("cm"))
    assert setup.network().n_cells == 1600


def test_the_setup_spreads_the_conductances_only_when_gamma_is_positive():
    uniform = common.Setup(nx=6, ny=6).network()
    spread = common.Setup(nx=6, ny=6, gamma=1.0).network()
    np.testing.assert_allclose(uniform.Gg, uniform.Gg[0])
    assert spread.Gg.std() > 0.0


def test_a_different_seed_draws_different_conductances():
    first = common.Setup(nx=6, ny=6, gamma=1.0, seed=0).network()
    second = common.Setup(nx=6, ny=6, gamma=1.0, seed=1).network()
    assert not np.allclose(first.Gg, second.Gg)


# --------------------------------------------------------------------------------------
# the setup builds what it was asked for
#
# Every assertion below uses a sheet that is wider than it is tall, and names the variant and
# the time step it expects. On the paper's square sheet the two axes are interchangeable, and
# a setup that quietly solved one variant for all three curves, or ran at its own time step,
# would draw a figure that looks entirely reasonable and is wrong.
# --------------------------------------------------------------------------------------

WIDE = {"nx": 36, "ny": 20}


def test_the_setup_solves_the_variant_it_was_given():
    for variant in Variant:
        assert common.Setup(**WIDE).simulation(variant).variant is variant


def test_the_setup_steps_at_its_own_time_step():
    assert common.Setup(**WIDE, dt=0.005).simulation(Variant.SKNM).dt == pytest.approx(0.005)


def test_the_setup_lays_the_sheet_out_wide_rather_than_tall():
    """36 by 20, not 20 by 36: the cell at index 1 is the next one along x."""
    network = common.Setup(**WIDE).network()
    assert network.n_cells == 36 * 20
    assert network.connections[0].tolist() == [0, 1]
    # Row-major over 36 columns puts 36 connections along y in the first row of them.
    along_x = sum(1 for first, second in network.connections if second - first == 1)
    assert along_x == 35 * 20


def test_the_conduction_path_runs_along_the_wide_axis():
    """Columns 9 and 34 of the row halfway up a 20-row sheet, which is row 9."""
    path = common.Setup(**WIDE).conduction_path()
    assert (path.start, path.end) == (9 * 36 + 9, 9 * 36 + 34)


def test_the_stimulus_sits_on_the_left_edge_of_the_wide_sheet():
    from sknm import presets

    simulation = common.Setup(**WIDE).simulation(Variant.SKNM)
    index = simulation.model.parameter_index("stim_amplitude")
    stimulated = np.flatnonzero(simulation.parameters[index])
    expected = np.flatnonzero(presets.hipsc_stimulus_amplitude(36, 20))
    np.testing.assert_array_equal(stimulated, expected)
    # Two leftmost columns of eleven rows, not two topmost rows of eleven columns.
    assert set((stimulated % 36).tolist()) == {0, 1}


def test_a_measured_velocity_needs_the_wave_to_reach_the_far_end():
    """A run that stopped at the near end of the path would have nothing to measure.

    The smallest sheet the paper's measurement columns fit on, so that this costs half a
    second rather than two.
    """
    velocity = common.measure_conduction_velocity(common.Setup(nx=36, ny=8), Variant.SKNM)
    assert 1.0 < velocity < 20.0


def test_a_measurement_uses_the_setup_s_threshold_and_not_a_module_constant():
    """The seam the two setups exist for.

    A beta setup carries -50 mV and a thousand-millisecond run, and reading either from a
    module constant would silently measure it at the cardiac -20 mV over 50 ms. Pinned on a
    cardiac sheet, where the same substitution is invisible because the two agree, by moving
    the threshold away from the constant and demanding the answer move with it.
    """
    small = {"nx": 36, "ny": 8}
    at_default = common.measure_conduction_velocity(common.Setup(**small), Variant.SKNM)
    raised = common.measure_conduction_velocity(common.Setup(**small, threshold=10.0), Variant.SKNM)

    assert raised != pytest.approx(at_default)


def test_a_measurement_stops_at_the_setup_s_run_length_and_not_a_module_constant():
    """The companion to the test above, for the other constant a setup carries.

    A run too short for the wave to arrive has no velocity to report, and `conduction_velocity`
    raises rather than returning `nan`. Reading the run length from the module would let this
    one finish.
    """
    with pytest.raises(ValueError, match="did not reach the threshold"):
        common.measure_conduction_velocity(common.Setup(nx=36, ny=8, t_end=1.0), Variant.SKNM)


def _refuse():
    raise AssertionError("this value should have come from the cache")


# --------------------------------------------------------------------------------------
# the beta setup
#
# The same shape of assertion as the cardiac setup above, and for the same reason. The beta
# sheet is square too, so every axis-swapping mistake is invisible on it; these use a sheet
# wider than it is tall. The two setups are also checked against each other, because the
# failure mode a sibling dataclass introduces is one of them quietly building the other's.
# --------------------------------------------------------------------------------------

#: A seed is required off the paper's square sheet: the committed draws describe only it.
BETA_WIDE = {"nx": 20, "ny": 8, "seed": 0}


def test_the_beta_setup_builds_a_beta_sheet_and_not_a_cardiac_one():
    """The cell type has to reach the network, or the beta figures plot the cardiac model."""
    from sknm import presets

    network = common.BetaSetup().network()

    assert network.n_cells == 225
    np.testing.assert_allclose(network.membrane_area, presets.BETA_MEMBRANE_AREA.m_as("cm ** 2"))
    assert network.Cm == pytest.approx(presets.BETA_CM.m_as("uF / cm ** 2"))
    assert network.lam > 1e4


def test_the_beta_setup_runs_the_beta_membrane_model():
    simulation = common.BetaSetup(**BETA_WIDE).simulation(Variant.SKNM)

    assert simulation.model.num_states == 5
    assert simulation.model.v_name == "v"


def test_the_beta_setup_carries_its_own_threshold_and_run_length():
    """The two constants that are catastrophic if they come from the cardiac setup."""
    beta = common.BetaSetup()
    cardiac = common.Setup()

    assert beta.threshold == pytest.approx(-50.0)
    assert beta.t_end == pytest.approx(1000.0)
    assert cardiac.threshold == pytest.approx(-20.0)
    assert cardiac.t_end == pytest.approx(50.0)


def test_the_beta_label_names_every_field_of_the_setup():
    setup = common.BetaSetup()
    label = setup.label()
    for field in setup.__dataclass_fields__:
        assert f"{field}=" in label


def test_beta_setups_differing_in_any_one_field_get_different_labels():
    import dataclasses

    base = common.BetaSetup()
    changed = {
        "delta_e": 0.02,
        "gamma": 1.0,
        "nx": 20,
        "ny": 8,
        "dt": 0.1,
        "t_end": 500.0,
        "threshold": -40.0,
        "seed": 3,
    }
    assert set(changed) == set(base.__dataclass_fields__)
    for field, value in changed.items():
        assert dataclasses.replace(base, **{field: value}).label() != base.label()


def test_the_two_setups_do_not_share_a_cache_key():
    """They carry different fields, so a label cannot mean both -- but check it, because a
    collision would serve a cardiac velocity under a beta label without any other symptom."""
    assert common.Setup().label() != common.BetaSetup().label()


def test_the_beta_setup_solves_the_variant_it_was_given():
    for variant in Variant:
        assert common.BetaSetup(**BETA_WIDE).simulation(variant).variant is variant


def test_the_beta_setup_steps_at_its_own_time_step():
    setup = common.BetaSetup(**BETA_WIDE, dt=0.005)
    assert setup.simulation(Variant.SKNM).dt == pytest.approx(0.005)


def test_the_beta_setup_lays_the_sheet_out_wide_rather_than_tall():
    network = common.BetaSetup(**BETA_WIDE).network()

    assert network.n_cells == 20 * 8
    assert network.connections[0].tolist() == [0, 1]
    along_x = sum(1 for first, second in network.connections if second - first == 1)
    assert along_x == 19 * 8


def test_the_beta_conduction_path_runs_along_the_wide_axis():
    """Columns 4 and 12 of the row halfway up an 8-row sheet, which is row 3."""
    path = common.BetaSetup(**BETA_WIDE).conduction_path()
    assert (path.start, path.end) == (3 * 20 + 4, 3 * 20 + 12)


def test_the_beta_stimulus_sits_on_the_left_edge_of_the_wide_sheet():
    from sknm import presets

    simulation = common.BetaSetup(**BETA_WIDE).simulation(Variant.SKNM)
    index = simulation.model.parameter_index("gkatpbar")
    values = simulation.parameters[index]
    stimulated = np.flatnonzero(values == presets.BETA_STIMULUS_KATP_CONDUCTANCE)
    expected = np.flatnonzero(
        presets.beta_stimulus_conductance(20, 8) == presets.BETA_STIMULUS_KATP_CONDUCTANCE
    )

    np.testing.assert_array_equal(stimulated, expected)
    # Two leftmost columns of five rows, not two topmost rows of five columns.
    assert set((stimulated % 20).tolist()) == {0, 1}
    # And the unstimulated cells keep the model's own default rather than being zeroed.
    assert np.all(values[values != presets.BETA_STIMULUS_KATP_CONDUCTANCE] == 500.0)


def test_the_beta_setup_spreads_the_conductances_only_when_gamma_is_positive():
    uniform = common.BetaSetup().network()
    spread = common.BetaSetup(gamma=1.0).network()

    np.testing.assert_allclose(uniform.Gg, uniform.Gg[0])
    assert spread.Gg.std() > 0.0


def test_the_beta_draws_are_the_reference_s_and_there_is_one_per_connection():
    draws = common.beta_draws()

    assert draws.shape == (common.BetaSetup().network().n_connections,)
    assert 0.0 <= draws.min() and draws.max() <= 1.0
    # The first value of the reference's own gj_scale_15x15_x.txt, and the last of its _y.txt.
    assert draws[0] == pytest.approx(0.297354)
    assert draws[-1] == pytest.approx(0.120178)


def test_a_beta_sheet_of_another_shape_is_refused_rather_than_silently_reseeded():
    """The draws are a fixed 420 numbers, so they only describe one sheet.

    Refused rather than quietly seeded, because the figures depend on getting the reference's
    own draws and a silent fallback would hand them someone else's without saying so.
    """
    with pytest.raises(ValueError, match="committed draws"):
        common.BetaSetup(nx=6, ny=6).network()


def test_a_seeded_beta_setup_draws_its_own_and_two_seeds_differ():
    first = common.BetaSetup(nx=6, ny=6, gamma=1.0, seed=0).network()
    second = common.BetaSetup(nx=6, ny=6, gamma=1.0, seed=1).network()

    assert not np.allclose(first.Gg, second.Gg)


def test_the_figures_use_the_reference_draws_rather_than_a_seed():
    """The default is what the scripts build, and it has to be the committed draws.

    On this sheet a seed moves the velocity at gamma = 1 by 20% and falls off the bottom of
    the published axis, so a setup that quietly seeded would draw a wrong Figure S1.
    """
    assert common.BetaSetup().seed is None
    np.testing.assert_array_equal(
        common.BetaSetup(gamma=1.0).network().Gg,
        common.BetaSetup(gamma=1.0).network().Gg,
    )
    seeded = common.BetaSetup(gamma=1.0, seed=0).network()
    assert not np.allclose(common.BetaSetup(gamma=1.0).network().Gg, seeded.Gg)
