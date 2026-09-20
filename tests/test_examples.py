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
from argparse import Namespace
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

FULL = Namespace(full=True, no_cache=False, output_dir=None)
FAST = Namespace(full=False, no_cache=False, output_dir=None)


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
    """The run length and threshold are not in a label, so they invalidate the file instead."""
    cache = common.ResultCache("sweep", directory=tmp_path)
    cache.compute("a", lambda: 1.0)
    with np.load(cache.path) as stored:
        spec = json.loads(str(stored["spec"].item()))
        labels, values = stored["labels"], stored["values"]
    spec["threshold"] = spec["threshold"] + 1.0
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


def _refuse():
    raise AssertionError("this value should have come from the cache")
