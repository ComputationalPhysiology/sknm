"""The paper's own setup: its cell sizes, its material constants, its stimulus and its probes.

Every geometric test here uses `alpha=1.5`, the 21x14 um cell. At `alpha=1` the paper's cells
are 16x16 um, so an x-direction connection and a y-direction connection have the same length
*and* the same cross-section, and a sheet that confused its two axes would look identical. Only
`tests/test_validation.py` uses `alpha=1`, because that is what the published table was
computed at.
"""

import numpy as np
import pint
import pytest

from sknm import presets
from sknm.analysis import ConductionPath
from sknm.units import cm, mS, ms, uF, um

NX, NY = 40, 40


def test_the_five_cell_sizes_are_the_published_ones():
    """The sizes the paper's Figure 3 is drawn at, and what `hipsc_cell_size` must reproduce."""
    measured = {
        1.0: (16.0, 16.0),
        1.5: (21.0, 14.0),
        2.0: (25.0, 12.5),
        3.0: (33.0, 11.0),
        4.0: (40.0, 10.0),
    }
    assert set(presets.CELL_DIMENSIONS) == set(measured)
    for alpha, (lx, ly) in measured.items():
        actual_x, actual_y = presets.CELL_DIMENSIONS[alpha]
        assert actual_x.m_as(um) == pytest.approx(lx)
        assert actual_y.m_as(um) == pytest.approx(ly)


def test_an_anisotropy_factor_of_zero_is_refused():
    with pytest.raises(ValueError, match="alpha must be positive"):
        presets.hipsc_sheet(4, 4, alpha=0.0)


def test_the_anisotropy_factor_may_be_written_as_a_whole_number():
    assert presets.hipsc_sheet(3, 3, alpha=1).n_cells == 9


def test_the_anisotropy_factor_is_a_ratio_and_takes_no_unit():
    with pytest.raises(pint.DimensionalityError, match="alpha is a ratio"):
        presets.hipsc_sheet(3, 3, alpha=1.5 * um)


# --- The sheet ------------------------------------------------------------------------------


def test_the_sheet_has_one_cell_per_grid_point_and_four_neighbours_each():
    network = presets.hipsc_sheet(5, 7, alpha=1.5)

    assert network.n_cells == 35
    assert network.n_connections == (5 - 1) * 7 + 5 * (7 - 1)


def test_the_cells_take_their_size_from_the_anisotropy_factor():
    network = presets.hipsc_sheet(3, 4, alpha=1.5)

    n_x = (3 - 1) * 4
    np.testing.assert_allclose(network.length[:n_x], 21e-4)
    np.testing.assert_allclose(network.length[n_x:], 14e-4)


def test_the_third_dimension_follows_the_extracellular_volume_fraction():
    """lz = (1 + delta_e) * ly, so it shows up in every connection's cross-section."""
    network = presets.hipsc_sheet(3, 4, alpha=1.5, delta_e=0.5)

    lx, ly = 21e-4, 14e-4
    lz = 1.5 * ly
    n_x = (3 - 1) * 4
    np.testing.assert_allclose(network.cross_section[:n_x], ly * lz)
    np.testing.assert_allclose(network.cross_section[n_x:], lx * lz)


def test_the_membrane_area_is_the_reference_constant_not_the_surface_of_a_cuboid():
    """The reference fixes Am at 1.8e-5 cm^2 for every alpha; the cuboid area is 3.4% smaller."""
    network = presets.hipsc_sheet(3, 3, alpha=1.0)

    lx = ly = 16e-4
    lz = 1.2 * ly
    cuboid = 2 * (lx * ly + lx * lz + ly * lz)

    np.testing.assert_allclose(network.membrane_area, 1.8e-5)
    assert cuboid == pytest.approx(1.7408e-5)


def test_the_membrane_area_does_not_change_with_the_anisotropy_factor():
    for alpha in presets.CELL_DIMENSIONS:
        network = presets.hipsc_sheet(2, 2, alpha=alpha)
        np.testing.assert_allclose(network.membrane_area, 1.8e-5)


def test_the_material_constants_are_the_paper_s():
    network = presets.hipsc_sheet(3, 3, alpha=1.5)

    assert network.sigma_i == pytest.approx(4.0)
    assert network.sigma_e == pytest.approx(20.0)
    assert network.Cm == pytest.approx(1.0)
    np.testing.assert_allclose(network.delta_e, 0.2)
    np.testing.assert_allclose(network.Gg, 2e-4)


def test_the_gap_junction_conductance_is_the_reference_resistance_inverted():
    """The reference writes Rg = 5e3 kOhm; the paper tunes the same number as Gg = 2e-4 mS."""
    assert presets.GAP_JUNCTION_CONDUCTANCE.m_as(mS) == pytest.approx(2e-4)


def test_the_gap_junction_conductance_can_be_replaced():
    network = presets.hipsc_sheet(3, 3, alpha=1.5, Gg=5e-4 * mS)

    np.testing.assert_allclose(network.Gg, 5e-4)


def test_the_constants_carry_units():
    assert presets.SIGMA_I.m_as(mS / cm) == pytest.approx(4.0)
    assert presets.SIGMA_E.m_as(mS / cm) == pytest.approx(20.0)
    assert presets.CM.m_as(uF / cm**2) == pytest.approx(1.0)
    assert presets.MEMBRANE_AREA.m_as(cm**2) == pytest.approx(1.8e-5)


# --- The stimulus ---------------------------------------------------------------------------


def test_the_stimulus_covers_the_two_leftmost_columns_over_eleven_rows():
    """The reference's region, x < 2*lx and 14*ly < y < 25*ly, with a node at each cell centre."""
    amplitude = presets.hipsc_stimulus_amplitude(NX, NY).reshape((NY, NX))

    rows, columns = np.nonzero(amplitude)
    assert sorted(set(columns)) == [0, 1]
    assert sorted(set(rows)) == list(range(14, 25))


def test_the_stimulus_is_the_reference_amplitude_where_it_applies():
    amplitude = presets.hipsc_stimulus_amplitude(NX, NY)

    assert set(np.unique(amplitude)) == {0.0, 20.0}
    assert presets.STIMULUS_AMPLITUDE == pytest.approx(20.0)


def test_the_unstimulated_cells_are_given_an_amplitude_of_zero():
    """Not left at the membrane model's own default, which would make every cell self-stimulate."""
    amplitude = presets.hipsc_stimulus_amplitude(NX, NY)

    assert amplitude.shape == (NX * NY,)
    assert np.count_nonzero(amplitude == 0.0) == NX * NY - 2 * 11


def test_the_stimulus_knows_which_axis_is_which():
    tall = presets.hipsc_stimulus_amplitude(6, 40).reshape((40, 6))
    wide = presets.hipsc_stimulus_amplitude(40, 6).reshape((6, 40))

    assert np.count_nonzero(tall) == 2 * 11
    assert sorted(set(np.nonzero(tall)[1])) == [0, 1]
    assert sorted(set(np.nonzero(wide)[1])) == [0, 1]


def test_the_stimulus_is_centred_on_the_row_the_velocity_is_measured_along():
    amplitude = presets.hipsc_stimulus_amplitude(NX, NY).reshape((NY, NX))
    rows = np.nonzero(amplitude)[0]

    measured_row = presets.hipsc_conduction_path(NX, NY).start // NX
    assert rows.mean() == pytest.approx(measured_row)


def test_the_stimulus_is_clipped_to_a_sheet_too_short_to_hold_it():
    amplitude = presets.hipsc_stimulus_amplitude(4, 5).reshape((5, 4))

    assert np.count_nonzero(amplitude) == 2 * 5
    assert (amplitude[:, :2] == 20.0).all()


def test_a_sheet_with_no_stimulated_cells_is_refused():
    with pytest.raises(ValueError, match="at least one cell"):
        presets.hipsc_stimulus_amplitude(0, 5)


# --- The probes -----------------------------------------------------------------------------


def test_the_conduction_path_runs_between_the_reference_s_two_cells():
    path = presets.hipsc_conduction_path(NX, NY, alpha=1.5)

    row = round(NY / 2) - 1
    assert path.start == row * NX + 9
    assert path.end == row * NX + 34


def test_the_conduction_distance_is_twenty_five_cells_of_the_chosen_size():
    assert presets.hipsc_conduction_path(NX, NY, alpha=1.0).distance == pytest.approx(25 * 16e-4)
    assert presets.hipsc_conduction_path(NX, NY, alpha=1.5).distance == pytest.approx(25 * 21e-4)


def test_the_conduction_path_is_a_conduction_path():
    assert isinstance(presets.hipsc_conduction_path(NX, NY), ConductionPath)


def test_a_sheet_too_narrow_to_hold_the_conduction_path_is_refused():
    with pytest.raises(ValueError, match="at least 35 cells wide"):
        presets.hipsc_conduction_path(20, 20)


def test_the_narrowest_sheet_that_holds_the_conduction_path_is_accepted():
    """Column 34 is the 35th, so 35 cells is exactly enough and 34 is one too few."""
    assert presets.hipsc_conduction_path(35, 40).end % 35 == 34

    with pytest.raises(ValueError, match="at least 35 cells wide"):
        presets.hipsc_conduction_path(34, 40)


def test_the_conduction_path_reads_the_two_axes_the_right_way_round():
    """On a square sheet nx and ny are interchangeable, and a row would hide behind a column."""
    path = presets.hipsc_conduction_path(36, 20)

    assert path.start // 36 == round(20 / 2) - 1
    assert path.start % 36 == 9


def test_the_centre_cell_is_the_one_the_reference_saves_its_trace_from():
    row = round(NY / 2) - 1
    assert presets.hipsc_centre_cell(NX, NY) == row * NX + NX // 2


def test_the_centre_cell_reads_the_two_axes_the_right_way_round():
    assert presets.hipsc_centre_cell(36, 20) == (round(20 / 2) - 1) * 36 + 18


def test_the_probes_point_inside_the_sheet_they_were_built_for():
    network = presets.hipsc_sheet(NX, NY)
    path = presets.hipsc_conduction_path(NX, NY)

    assert max(path.start, path.end, presets.hipsc_centre_cell(NX, NY)) < network.n_cells


# --- Gap junction variation -------------------------------------------------------------------


@pytest.fixture
def small_sheet():
    return presets.hipsc_sheet(4, 5, alpha=1.5)


def draws_for(network, seed=0):
    return np.random.default_rng(seed).random(network.n_connections)


def test_no_variation_leaves_every_conductance_exactly_as_it_was(small_sheet):
    """gamma = 0 has to be the identity, or a sweep starting there starts somewhere else."""
    varied = presets.vary_conductances(small_sheet, 0.0, draws_for(small_sheet))

    np.testing.assert_array_equal(varied.m_as(mS), small_sheet.Gg)


def test_a_conductance_is_scaled_by_the_reference_s_resistance_formula(small_sheet):
    """Rg = Rg0 / (a*(1-gamma) + (1-a)*(1+gamma)), and Gg is 1/Rg."""
    draws = draws_for(small_sheet)
    gamma = 0.4

    varied = presets.vary_conductances(small_sheet, gamma, draws)

    expected = small_sheet.Gg * (draws * (1 - gamma) + (1 - draws) * (1 + gamma))
    np.testing.assert_allclose(varied.m_as(mS), expected, rtol=1e-14)


def test_the_spread_of_conductances_is_the_variation_either_side_of_the_nominal(small_sheet):
    draws = np.array([0.0, 0.5, 1.0] + [0.5] * (small_sheet.n_connections - 3))

    varied = presets.vary_conductances(small_sheet, 0.25, draws).m_as(mS)

    np.testing.assert_allclose(varied[:3] / small_sheet.Gg[:3], [1.25, 1.0, 0.75], rtol=1e-14)


def test_the_varied_conductances_build_a_network(small_sheet):
    varied = presets.vary_conductances(small_sheet, 0.5, draws_for(small_sheet))

    rebuilt = small_sheet.with_conductances(varied)

    assert rebuilt.n_connections == small_sheet.n_connections
    assert not np.array_equal(rebuilt.Gg, small_sheet.Gg)


def test_the_same_draws_give_the_same_conductances_at_every_variation(small_sheet):
    """What makes the paper's sweep comparable point for point: one fixed set of draws."""
    draws = draws_for(small_sheet)
    ordering = np.argsort(presets.vary_conductances(small_sheet, 0.3, draws).m_as(mS))

    for gamma in (0.1, 0.6, 1.0):
        varied = presets.vary_conductances(small_sheet, gamma, draws).m_as(mS)
        np.testing.assert_array_equal(np.argsort(varied), ordering)


@pytest.mark.parametrize("gamma", [-0.1, 1.5])
def test_a_variation_outside_the_unit_interval_is_refused(small_sheet, gamma):
    with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
        presets.vary_conductances(small_sheet, gamma, draws_for(small_sheet))


def test_a_variation_is_a_ratio_and_takes_no_unit(small_sheet):
    with pytest.raises(pint.DimensionalityError, match="gamma is a ratio"):
        presets.vary_conductances(small_sheet, 0.5 * ms, draws_for(small_sheet))


def test_one_draw_per_connection_is_required(small_sheet):
    with pytest.raises(ValueError, match="one draw per connection"):
        presets.vary_conductances(small_sheet, 0.5, np.zeros(3))


@pytest.mark.parametrize("bad", [-0.01, 1.01])
def test_a_draw_outside_the_unit_interval_is_refused(small_sheet, bad):
    draws = draws_for(small_sheet)
    draws[2] = bad

    with pytest.raises(ValueError, match="draws must be between 0 and 1"):
        presets.vary_conductances(small_sheet, 0.5, draws)


def test_the_cell_size_formula_reproduces_every_measured_pair():
    for alpha, (lx, ly) in presets.CELL_DIMENSIONS.items():
        computed_x, computed_y = presets.hipsc_cell_size(alpha)
        assert computed_x.m_as(um) == pytest.approx(lx.m_as(um))
        assert computed_y.m_as(um) == pytest.approx(ly.m_as(um))


def test_the_default_anisotropy_factor_is_the_one_the_sheet_defaults_to():
    """The two defaults have to agree, or `hipsc_sheet()` and `hipsc_cell_size()` disagree."""
    lx, ly = presets.hipsc_cell_size()
    assert (lx.m_as(um), ly.m_as(um)) == (16.0, 16.0)
    assert presets.hipsc_cell_size() == presets.hipsc_cell_size(1.0)


def test_an_anisotropy_factor_between_the_measured_ones_is_accepted():
    lx, ly = presets.hipsc_cell_size(2.5)
    assert lx.m_as(um) == pytest.approx(2.5 * ly.m_as(um))
    network = presets.hipsc_sheet(4, 4, alpha=2.5)
    assert network.n_cells == 16


def test_the_cell_size_holds_the_intracellular_volume_near_four_picolitres():
    """The width is the one that would give exactly 4 pL, moved at most a quarter micrometre.

    The volume is therefore near 4 pL rather than at it, and how near depends on where the
    rounding lands: 3802 um^3 at alpha=2.5, against 4096 at alpha=1.
    """
    for alpha in (1.0, 1.7, 2.5, 3.3, 4.0):
        lx, ly = presets.hipsc_cell_size(alpha)
        assert abs(ly.m_as(um) - np.cbrt(4000.0 / alpha)) <= 0.25
        assert lx.m_as(um) * ly.m_as(um) ** 2 == pytest.approx(4000.0, rel=0.08)


def test_the_cell_width_is_rounded_to_half_a_micrometre():
    for alpha in (1.0, 1.7, 2.5, 3.3, 4.0):
        _, ly = presets.hipsc_cell_size(alpha)
        assert (2.0 * ly.m_as(um)) % 1.0 == 0.0


def test_an_anisotropy_factor_of_zero_or_less_is_refused():
    with pytest.raises(ValueError, match="alpha must be positive"):
        presets.hipsc_cell_size(0.0)


# --- the pancreatic beta cell setup ---------------------------------------------------------

BETA_NX, BETA_NY = 15, 15


def test_the_beta_cell_is_a_thirteen_micrometre_cube():
    """Every connection is the same length and has the same cross-section, in both directions."""
    network = presets.beta_sheet(BETA_NX, BETA_NY)
    side = (13 * um).m_as(cm)

    np.testing.assert_allclose(network.length, side)
    np.testing.assert_allclose(network.cross_section, side * side)


def test_the_third_dimension_is_fixed_rather_than_following_the_volume_fraction():
    """The divergence from the hiPSC sheet, which derives `lz` from `delta_e`.

    At `delta_e = 0.5` the hiPSC rule would give 19.5 um; the reference fixes 13 um.
    """
    side = (13 * um).m_as(cm)
    for delta_e in (0.5, 0.2, 0.02):
        network = presets.beta_sheet(BETA_NX, BETA_NY, delta_e=delta_e)
        np.testing.assert_allclose(network.cross_section, side * side)

    # The rule the hiPSC sheet uses, for contrast: it would make the cross-section move.
    hipsc_like = {d: (1 + d) * side * side for d in (0.5, 0.2, 0.02)}
    assert len(set(hipsc_like.values())) == 3


def test_the_membrane_area_is_the_surface_of_a_sphere_of_that_diameter():
    """A beta cell is modelled as a sphere, not as the cuboid its cell size describes.

    pi * d**2 at d = 13 um is 5.30929e-6 cm2, which is the reference's stated constant to
    every digit it quotes. The cuboid's surface, 6 * l**2, is nearly twice that.
    """
    network = presets.beta_sheet(BETA_NX, BETA_NY)
    diameter = (13 * um).m_as(cm)

    assert presets.BETA_MEMBRANE_AREA.m_as(cm**2) == pytest.approx(np.pi * diameter**2)
    assert presets.BETA_MEMBRANE_AREA.m_as(cm**2) == pytest.approx(5.3093e-6, rel=1e-5)
    assert np.all(network.membrane_area == pytest.approx(np.pi * diameter**2))


def test_the_specific_capacitance_is_derived_from_the_model_s_own_capacitance():
    """`Cm * Am` has to equal what PBM's voltage equation divides by, exactly.

    The reference declares 5300 fF in the membrane model and 1.0 uF/cm2 in the network, which
    disagree by 0.18% against a check that admits none. Of the three numbers only the specific
    capacitance is a generic constant rather than a measurement, so it is the one derived.
    """
    network = presets.beta_sheet(BETA_NX, BETA_NY)
    model = presets.beta_membrane_model()

    assert network.Cm * network.membrane_area[0] == pytest.approx(model.capacitance, rel=1e-15)
    assert network.Cm == pytest.approx(0.998248, rel=1e-5)


def test_the_beta_network_and_membrane_model_build_a_simulation():
    """The check the derivation exists for: the reference's own numbers would raise here."""
    from sknm import Simulation

    simulation = Simulation(
        presets.beta_sheet(BETA_NX, BETA_NY), presets.beta_membrane_model(), dt=0.02 * ms
    )

    assert simulation.model.capacitance == pytest.approx(5.3e-6)


def test_a_beta_network_built_with_a_round_specific_capacitance_is_refused():
    """The companion to the test above, and the reason the derivation is not cosmetic."""
    from sknm import Simulation, sheet

    network = sheet(
        4,
        4,
        lx=13 * um,
        ly=13 * um,
        lz=13 * um,
        delta_e=0.5,
        sigma_i=presets.SIGMA_I,
        sigma_e=presets.SIGMA_E,
        Gg=presets.BETA_GAP_JUNCTION_CONDUCTANCE,
        membrane_area=presets.BETA_MEMBRANE_AREA,
        Cm=1.0 * uF / cm**2,
    )

    with pytest.raises(ValueError, match="capacitance"):
        Simulation(network, presets.beta_membrane_model(), dt=0.02 * ms)


def test_the_gap_junction_conductance_is_a_thousandfold_weaker_than_the_cardiac_one():
    """5e6 kOhm against 5e3 kOhm, which is what makes a beta wave 150 times slower."""
    assert presets.BETA_GAP_JUNCTION_CONDUCTANCE.m_as(mS) == pytest.approx(2e-7)
    ratio = presets.GAP_JUNCTION_CONDUCTANCE / presets.BETA_GAP_JUNCTION_CONDUCTANCE
    assert float(ratio) == pytest.approx(1000.0)


def test_the_beta_stimulus_covers_two_columns_over_five_rows():
    conductance = presets.beta_stimulus_conductance(BETA_NX, BETA_NY)
    grid = conductance.reshape(BETA_NY, BETA_NX)
    stimulated = grid == presets.BETA_STIMULUS_KATP_CONDUCTANCE

    assert stimulated.sum() == 2 * 5
    assert set(np.flatnonzero(stimulated.any(axis=0))) == {0, 1}
    assert set(np.flatnonzero(stimulated.any(axis=1))) == {5, 6, 7, 8, 9}


def test_the_beta_stimulus_halves_the_conductance_rather_than_zeroing_it():
    """Unlike the cardiac stimulus, the unstimulated value is the model's own default."""
    conductance = presets.beta_stimulus_conductance(BETA_NX, BETA_NY)

    assert set(np.unique(conductance)) == {
        presets.BETA_STIMULUS_KATP_CONDUCTANCE,
        presets.BETA_KATP_CONDUCTANCE,
    }
    assert presets.BETA_STIMULUS_KATP_CONDUCTANCE == pytest.approx(
        0.5 * presets.BETA_KATP_CONDUCTANCE
    )


def test_the_beta_stimulus_knows_which_axis_is_which():
    conductance = presets.beta_stimulus_conductance(20, 8)
    grid = conductance.reshape(8, 20)
    stimulated = grid == presets.BETA_STIMULUS_KATP_CONDUCTANCE

    assert set(np.flatnonzero(stimulated.any(axis=0))) == {0, 1}
    assert set(np.flatnonzero(stimulated.any(axis=1))) == {1, 2, 3, 4, 5}


def test_the_beta_conduction_path_runs_between_the_reference_s_two_cells():
    """Columns 4 and 12 of row 6, which is the reference's `start_idx` and `end_idx` at 15x15."""
    path = presets.beta_conduction_path(BETA_NX, BETA_NY)

    assert (path.start, path.end) == (94, 102)
    assert path.distance == pytest.approx((8 * 13 * um).m_as(cm))


def test_a_sheet_too_narrow_to_hold_the_beta_conduction_path_is_refused():
    with pytest.raises(ValueError, match="at least 13"):
        presets.beta_conduction_path(12, BETA_NY)


def test_the_narrowest_sheet_that_holds_the_beta_conduction_path_is_accepted():
    assert presets.beta_conduction_path(13, BETA_NY).end % 13 == 12


def test_the_beta_conduction_path_reads_the_two_axes_the_right_way_round():
    path = presets.beta_conduction_path(20, 8)

    assert path.start == (8 // 2 - 1) * 20 + 4


def test_the_beta_centre_cell_is_the_one_the_reference_saves_its_trace_from():
    """The reference's `save_idx` at 15x15, computed with C integer division."""
    assert presets.beta_centre_cell(BETA_NX, BETA_NY) == 97


def test_the_beta_centre_cell_reads_the_two_axes_the_right_way_round():
    assert presets.beta_centre_cell(20, 8) == (8 // 2 - 1) * 20 + 10


def test_the_two_setups_measure_at_the_same_cell_of_a_sheet_of_the_same_shape():
    """Both reference drivers use the same `save_idx`, so this is one formula, not two."""
    assert presets.beta_centre_cell(24, 18) == presets.hipsc_centre_cell(24, 18)


def test_the_two_thresholds_are_the_reference_s():
    from sknm.units import mV

    assert presets.HIPSC_THRESHOLD.m_as(mV) == pytest.approx(-20.0)
    assert presets.BETA_THRESHOLD.m_as(mV) == pytest.approx(-50.0)


def test_the_beta_extracellular_conductance_dominates_the_intracellular_one():
    """Why KNM, SKNM and SKNM(ue=0) coincide for beta cells and do not for hiPSC-CMs.

    The gap junction resistance is so large that it swamps both conductivities, so lambda is
    enormous and SKNM's `lambda / (1 + lambda)` is indistinguishable from 1.
    """
    network = presets.beta_sheet(BETA_NX, BETA_NY)

    assert network.lam > 1e4
    assert network.lam / (1 + network.lam) == pytest.approx(1.0, abs=1e-4)
    assert presets.hipsc_sheet(40, 40).lam < 100


# --- the index arithmetic both families share -----------------------------------------------


@pytest.mark.parametrize(
    ("ny", "row"),
    [(40, 19), (15, 6), (20, 9), (8, 3), (9, 3), (13, 5), (36, 17), (25, 11), (4, 1)],
)
def test_the_measurement_row_is_the_reference_s_integer_division(ny, row):
    """`round(num_cells_y/2) - 1` in the C, where the division is between two ints.

    So it truncates before it rounds, and `ny // 2 - 1` is the Python for it. Reading it as
    `round(ny / 2) - 1` agrees at every even `ny` and is one row out at every odd one, which
    no hiPSC sheet would ever show: the paper's is 40 by 40.
    """
    assert presets.hipsc_centre_cell(7, ny) // 7 == row
    assert presets.beta_centre_cell(7, ny) // 7 == row


@pytest.mark.parametrize(
    ("nx", "ny", "first", "rows"),
    [(40, 40, 14, 11), (15, 15, 5, 5)],
)
def test_the_stimulated_block_is_the_reference_s_rows(nx, ny, first, rows):
    """The C stimulates a window in y given in cell widths, with a node at each cell centre.

    For hiPSC-CMs `14*ly < y < 25*ly` is rows 14 to 24; for beta cells `5*ly < y < 10*ly` is
    rows 5 to 9. Both come out as the block of `rows` rows centred in the sheet -- which for
    the cardiac sheet also happens to be the measurement row, and for the beta sheet is one
    row above it.
    """
    if nx == 40:
        stimulated = presets.hipsc_stimulus_amplitude(nx, ny) > 0
    else:
        stimulated = (
            presets.beta_stimulus_conductance(nx, ny) == presets.BETA_STIMULUS_KATP_CONDUCTANCE
        )
    covered = np.flatnonzero(stimulated.reshape(ny, nx).any(axis=1))

    np.testing.assert_array_equal(covered, np.arange(first, first + rows))


def test_the_beta_stimulus_is_not_centred_on_the_measurement_row():
    """Unlike the cardiac one, and the reference is explicit about it.

    Worth pinning rather than quietly aligning the two: the beta block sits one row above the
    row the velocity is read along, and a helper that centred it on that row would move the
    stimulus off the reference's cells.
    """
    stimulated = presets.beta_stimulus_conductance(15, 15) == presets.BETA_STIMULUS_KATP_CONDUCTANCE
    rows = np.flatnonzero(stimulated.reshape(15, 15).any(axis=1))
    measurement_row = presets.beta_conduction_path(15, 15).start // 15

    assert int(np.median(rows)) == measurement_row + 1
