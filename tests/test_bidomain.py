"""The bidomain and monodomain machinery the Figure 6 to 9 scripts share.

Two layers are checked here, and they have different requirements. The conductivity field,
the conductance ratio and the regions the stimulus and the ground occupy are plain numpy and
are checked unconditionally. Everything that builds a mesh needs `dolfinx` and `fenicsx-beat`,
which are not `sknm` dependencies, so those tests carry the `dolfinx` marker and skip without
them.

The oracle throughout is the reference implementation's own formulas, transcribed here from
its source rather than imported from the code under test.
"""

import itertools
from pathlib import Path

import numpy as np
import pytest

import bidomain

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_the_intracellular_conductivity_is_the_reference_s_at_the_paper_s_default():
    """Both components of `M_i` at alpha = 1, gamma = 0, the value every other test rests on.

    Worked from the reference's own expression with exact rationals::

        Mi_x = delta_i*sigma_i / (1 + sigma_i*Rg*delta_i*ly*lz/lx)
             = 3.2 / (1 + 16000*19.2e-4) = 3.2/31.72
    """
    setup = bidomain.BidomainSetup()
    Mi_x, Mi_y = setup.intracellular_conductivity()

    assert Mi_x.shape == (1600,)
    assert Mi_y.shape == (1600,)
    np.testing.assert_allclose(Mi_x, 0.1008827238335435)
    np.testing.assert_allclose(Mi_y, 0.1008827238335435)


def test_a_long_cell_conducts_better_along_its_length():
    """The two components at alpha = 4, where the cell is 40 by 10 um.

    Every part of this expression except the geometric factor is the same in both directions,
    and the factor is the one that carries the cell's shape: a component divides the cell's
    cross-section by its length in that direction, so the long axis gets the larger
    conductivity. Worked from the reference's own expression with exact rationals.

    The paper's default sheet cannot see any of this. Its cells are square, so the two
    components are equal and a mistake that exchanged them would be invisible there, while
    transposing the conductivity tensor of Figure 7's sheet.
    """
    Mi_x, Mi_y = bidomain.BidomainSetup(alpha=4.0).intracellular_conductivity()

    np.testing.assert_allclose(Mi_x, 0.5517241379310345)
    np.testing.assert_allclose(Mi_y, 0.04113110539845758)
    assert Mi_x[0] / Mi_y[0] > 13.0


def test_the_draws_are_the_reference_s_own_and_there_is_one_per_cell():
    """One draw per cell, where the network model's are one per connection.

    1600 against 1560 on the same 40 by 40 sheet, and the two are not interchangeable.
    """
    draws = bidomain.gap_junction_draws()

    assert draws.shape == (1600,)
    assert 0.0 <= draws.min() and draws.max() <= 1.0
    # The first and last values of the reference's own gj_scale_x_BD.txt.
    assert draws[0] == pytest.approx(0.860448)
    assert draws[-1] == pytest.approx(0.473289)


def test_a_sheet_of_another_shape_is_refused_rather_than_silently_truncated():
    """The committed draws are a fixed 1600 numbers, so they describe one sheet only."""
    with pytest.raises(ValueError, match="committed draws"):
        bidomain.BidomainSetup(nx=20, ny=20, gamma=0.5).gap_junction_resistance()


def test_the_variation_spreads_the_resistance_the_way_the_reference_does():
    """``Rg / (a(1 - gamma) + (1 - a)(1 + gamma))``, transcribed from the reference."""
    setup = bidomain.BidomainSetup(gamma=0.4)
    draws = bidomain.gap_junction_draws()

    expected = bidomain.GAP_JUNCTION_RESISTANCE / (draws * 0.6 + (1.0 - draws) * 1.4)

    np.testing.assert_allclose(setup.gap_junction_resistance(), expected)
    assert setup.gap_junction_resistance().std() > 0.0


def test_no_variation_leaves_the_conductivity_uniform_whatever_the_draws_say():
    """The algebra that makes Figures 6 and 7 what they are.

    At gamma = 0 the two terms are ``a`` and ``1 - a``, which sum to one for every draw, so
    the resistance is the unvaried one everywhere and `M_i` is spatially constant. A bidomain
    sheet with a uniform, isotropic `M_i` reduces exactly to a monodomain one, which is why
    Figure 6's two rows agree and Figure 8's, at gamma > 0, do not.
    """
    setup = bidomain.BidomainSetup(gamma=0.0)

    np.testing.assert_allclose(setup.gap_junction_resistance(), bidomain.GAP_JUNCTION_RESISTANCE)


# --------------------------------------------------------------------------------------
# the conductance ratio and the misfit it minimizes
#
# Equation (31) is an area integral of conductivities, in mS^2, and equation (32) is its
# minimizer. It is not the network model's equation (29), which sums conductances over
# connections and comes out in (mS/cm)^2. The two differ by four orders of magnitude, and
# `sknm.CellNetwork.conductance_misfit` cannot stand in for this one.
# --------------------------------------------------------------------------------------


def test_the_conductance_ratio_at_the_paper_s_default_setup():
    """4 mS/cm over 3.2/31.72 mS/cm, which comes out at exactly 39.65."""
    assert bidomain.BidomainSetup().conductance_ratio() == pytest.approx(39.65)


def test_a_uniform_isotropic_conductivity_is_matched_exactly():
    """At alpha = 1 and gamma = 0 one ratio relates the two conductivities everywhere.

    That is the assumption the monodomain model rests on, so the misfit is zero and the two
    continuum models coincide, as Figure 6 shows.
    """
    setup = bidomain.BidomainSetup()

    assert setup.conductivity_misfit() == pytest.approx(0.0, abs=1e-18)


def test_the_ratio_is_the_one_that_minimizes_the_misfit():
    """Equation (32) is not an independent choice; it is where equation (31) bottoms out."""
    setup = bidomain.BidomainSetup(alpha=4.0, gamma=0.6)
    best = setup.conductance_ratio()

    at_best = setup.conductivity_misfit(best)
    for factor in (0.9, 0.99, 1.01, 1.1):
        assert setup.conductivity_misfit(best * factor) > at_best


def test_the_misfit_is_the_area_weighted_sum_the_reference_integrates():
    """Worked from the reference's own expression with exact rationals, at alpha = 4.

        F = 1/2 * 1600 * lx*ly * [(Me - lam*Mi_x)^2 + (Me - lam*Mi_y)^2]

    An anisotropic cell with no variation, so every cell carries the same two components and
    the sum is 1600 identical terms, which makes the expected value hand-computable.
    """
    setup = bidomain.BidomainSetup(alpha=4.0)

    assert setup.conductance_ratio() == pytest.approx(7.747430370263272)
    assert setup.conductivity_misfit() == pytest.approx(0.04360825961803867)


def test_the_misfit_grows_with_the_gap_junction_variation():
    """Figure 9's left panel: the spread is what pushes the conductivities off a single ratio."""
    misfits = [
        bidomain.BidomainSetup(gamma=gamma).conductivity_misfit()
        for gamma in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    ]

    assert misfits == sorted(misfits)
    assert misfits[0] == pytest.approx(0.0, abs=1e-18)
    assert misfits[-1] > 0.0


def test_the_membrane_substeps_add_up_to_one_time_step():
    """The substeps have to cover the step exactly, or the two clocks drift apart.

    The membrane model's stimulus is a function of time, so a membrane clock that ran slow
    would hold the stimulus on past when the equation says it ends.
    """
    for dt in (1.0, 0.2, 0.1, 0.02, 0.01, 0.005, 0.002, 0.0005):
        step, count = bidomain.membrane_substeps(dt)

        assert count >= 1
        assert step <= bidomain.MAX_ODE_STEP
        assert count * step == pytest.approx(dt, rel=1e-12)

    # The reference's own step, and the two rows of Table S3 that substep at all.
    assert bidomain.membrane_substeps(0.02) == (0.01, 2)
    assert bidomain.membrane_substeps(1.0) == (0.01, 100)


def test_the_membrane_is_advanced_however_short_the_time_step_is():
    """The reference's own arithmetic stops advancing the membrane model below dt = 0.005.

    It fixes the ODE step at 0.01 and takes ``round(dt/0.01)`` of them, which is zero there --
    so the states never move, the clock never moves, and the resting state is written over the
    spatial solve on every step. Every row of Tables S1 and S3 below that step, and all of
    Table S2, is therefore not reproducible from the published code as it stands.

    The first assertion is what the reference computes, kept so that this test says what it is
    guarding against rather than only that the guard is in place.
    """
    for dt in (0.004, 0.002, 0.001, 0.0005):
        assert round(dt / 0.01) == 0
        assert bidomain.membrane_substeps(dt)[1] == 1


# --------------------------------------------------------------------------------------
# the mesh, the stimulus and the ground
#
# These are predicates on coordinates, so they are checked here on a grid built by hand rather
# than on a mesh, which is how the stimulus can be tested without `dolfinx`. The stimulus is the
# one place in this module where an off-by-one in the comparison stops the wave from launching
# at all.
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("alpha", "expected"),
    [(1.0, (2, 2)), (1.5, (3, 2)), (2.0, (3, 2)), (3.0, (4, 1)), (4.0, (5, 1))],
)
def test_the_mesh_divides_every_cell_into_a_whole_number_of_elements(alpha, expected):
    """No element straddles a cell boundary, so the per-cell conductivity field is exact.

    The count is the one that brings the element closest to the target size, and it is taken
    per axis: at alpha = 4 a cell is 40 by 10 um, so five elements along x and one along y
    both come out near the 8 um target, where a single count for both axes could not.
    """
    setup = bidomain.BidomainSetup(alpha=alpha)

    assert setup.elements_per_cell() == expected


def test_every_element_is_close_to_the_target_size():
    for alpha in (1.0, 1.5, 2.0, 3.0, 4.0):
        setup = bidomain.BidomainSetup(alpha=alpha)
        lx, ly, _ = setup.cell_size()
        per_x, per_y = setup.elements_per_cell()
        assert 0.5 <= (lx / per_x) / setup.element_size <= 1.5
        assert 0.5 <= (ly / per_y) / setup.element_size <= 1.5


def test_the_paper_s_sheet_is_meshed_at_eight_micrometres():
    setup = bidomain.BidomainSetup()

    assert setup.elements() == (80, 80)
    assert setup.domain_size() == pytest.approx((640e-4, 640e-4))


def _node_grid(setup):
    """The coordinates of a structured mesh's nodes, as the mesh would hold them."""
    nx, ny = setup.elements()
    width, height = setup.domain_size()
    x, y = np.meshgrid(np.linspace(0.0, width, nx + 1), np.linspace(0.0, height, ny + 1))
    return np.column_stack([x.ravel(), y.ravel()])


def _node_indices(setup):
    """The same nodes as integer grid indices, which carry no rounding error."""
    nx, ny = setup.elements()
    columns, rows = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1))
    return columns.ravel(), rows.ravel()


def _lumped_mass(setup):
    """The area each node of the structured grid carries, and so the area a nodal stimulus
    drives: a full element square inside the sheet, half of one along an edge of it."""
    nx, ny = setup.elements()
    width, height = setup.domain_size()
    columns, rows = _node_indices(setup)
    along_x = np.where((columns == 0) | (columns == nx), 0.5, 1.0) * (width / nx)
    along_y = np.where((rows == 0) | (rows == ny), 0.5, 1.0) * (height / ny)
    return along_x * along_y


@pytest.mark.parametrize("element_size", [16e-4, 8e-4, 4e-4, 5e-4, 3e-4])
@pytest.mark.parametrize("alpha", [1.0, 4.0])
def test_the_stimulated_membrane_is_exactly_the_strip_s_own_area(element_size, alpha):
    """The property the edge weighting exists for, and the one that decides the launch time.

    A nodal stimulus drives each node's share of the membrane, so the stimulated area is the
    sum of the lumped masses of the nodes it reaches. That has to come out at the strip's own
    area, two cells by eleven, or the sheet is given more or less current than the paper
    specifies, and it fires early or not at all.

    Checked across meshes that divide the cell and meshes that do not, because the two fail in
    opposite directions: on a dividing mesh every node of the edge is included or excluded
    together, and on any other none of them is on the edge at all. Checked at alpha = 4 as
    well, where the strip is 80 by 110 um rather than square, so that the two cell dimensions
    cannot be exchanged without it showing.
    """
    setup = bidomain.BidomainSetup(alpha=alpha, element_size=element_size)
    lx, ly, _ = setup.cell_size()

    stimulated = (bidomain.stimulus_weight(setup, _node_grid(setup)) * _lumped_mass(setup)).sum()

    assert stimulated == pytest.approx(2 * lx * 11 * ly, rel=1e-12)


def test_a_whole_or_nothing_comparison_gets_the_stimulated_area_wrong():
    """What the weighting is there to prevent, stated as the difference it makes.

    Neither of the two answers a strict or a closed comparison can give is right on a mesh
    whose nodes land on the edge of the strip, and the two miss in opposite directions. The
    closed one launches the wave 5.7 ms early and steepens the upstroke by 2.2%; the strict
    one leaves the sheet unable to propagate at all.

    Only the closed figure is pinned. The strict one is not a fixed number: 14*ly and 25*ly
    are not representable, so whether each edge row survives ``>`` and ``<`` comes down to
    which way its own coordinate rounded, and on this mesh the two edges go opposite ways.
    That is the deeper reason not to write the comparison either way round.
    """
    setup = bidomain.BidomainSetup()
    points = _node_grid(setup)
    lx, ly, _ = setup.cell_size()
    mass = _lumped_mass(setup)
    strip = 2 * lx * 11 * ly

    closed = (points[:, 0] <= 2 * lx) | np.isclose(points[:, 0], 2 * lx)
    closed &= (points[:, 1] >= 14 * ly) | np.isclose(points[:, 1], 14 * ly)
    closed &= (points[:, 1] <= 25 * ly) | np.isclose(points[:, 1], 25 * ly)
    strict = (points[:, 0] < 2 * lx) & (points[:, 1] > 14 * ly) & (points[:, 1] < 25 * ly)

    assert (mass * closed).sum() / strip == pytest.approx(1.176, abs=5e-4)
    assert (mass * strict).sum() / strip < 0.9


def test_the_stimulus_covers_the_cells_the_network_model_stimulates():
    """The cross-validation rests on both models stimulating the same tissue.

    The network model names cells and this one names a region of space, so the two are
    written differently and can disagree; they are compared as cells here.
    """
    from sknm import presets

    setup = bidomain.BidomainSetup()
    points = _node_grid(setup)
    weight = bidomain.stimulus_weight(setup, points)
    reached = points[weight > 0.0]

    lx, ly, _ = setup.cell_size()
    # A node on a cell boundary touches both cells, so the nodes the stimulus reaches span one
    # more row and column of cell corners than the cells the strip covers.
    assert set(np.round(reached[:, 0] / lx).astype(int).tolist()) == {0, 1, 2}
    assert set(np.round(reached[:, 1] / ly).astype(int).tolist()) == set(range(14, 26))

    stimulated = np.flatnonzero(presets.hipsc_stimulus_amplitude(setup.nx, setup.ny))
    assert sorted({int(cell % setup.nx) for cell in stimulated}) == [0, 1]
    assert sorted({int(cell // setup.nx) for cell in stimulated}) == list(range(14, 25))


def test_the_nodes_on_the_edge_of_the_strip_get_half_the_stimulus():
    """Written in node indices, which carry no rounding error, so the expected set is exact."""
    setup = bidomain.BidomainSetup()
    per_x, per_y = setup.elements_per_cell()
    columns, rows = _node_indices(setup)

    inside = (columns <= 2 * per_x) & (rows >= 14 * per_y) & (rows <= 25 * per_y)
    on_the_edge = (columns == 2 * per_x) | (rows == 14 * per_y) | (rows == 25 * per_y)
    expected = np.where(inside, np.where(on_the_edge, 0.5, 1.0), 0.0)
    # A corner of the strip is on two edges at once and takes a quarter: the strip covers one
    # of the four quadrants its basis function spans, so the total comes out at
    # the strip's area rather than a half-element over it.
    corner = inside & (columns == 2 * per_x) & ((rows == 14 * per_y) | (rows == 25 * per_y))
    expected[corner] = 0.25
    assert corner.sum() == 2

    np.testing.assert_array_equal(bidomain.stimulus_weight(setup, _node_grid(setup)), expected)


def test_the_probes_sit_where_the_paper_measures():
    """The wave is timed between the centres of cell columns 9 and 34, halfway up the sheet.

    Worth pinning directly rather than leaving to the published numbers. Moving the far probe
    down to a quarter of the sheet's height changes the velocity by 1.0%: enough to matter,
    and not enough for a table reproduced to within 1% to notice.
    """
    setup = bidomain.BidomainSetup()
    lx, _, _ = setup.cell_size()
    width, height = setup.domain_size()

    near, far, centre = bidomain.probe_points(setup)

    np.testing.assert_allclose(near, [9.5 * lx, 0.5 * height, 0.0])
    np.testing.assert_allclose(far, [34.5 * lx, 0.5 * height, 0.0])
    np.testing.assert_allclose(centre, [0.5 * width, 0.5 * height, 0.0])


def test_the_probes_are_twenty_five_cells_apart():
    """The distance the velocity divides by, which the paper states as 25 cells."""
    setup = bidomain.BidomainSetup(alpha=4.0)
    lx, _, _ = setup.cell_size()

    near, far, _ = bidomain.probe_points(setup)

    assert far[0] - near[0] == pytest.approx(25 * lx)


def test_the_ground_is_the_reference_s_l_shaped_corner():
    """Two segments one cell long, meeting at the origin, as the reference's mesh marks them."""
    setup = bidomain.BidomainSetup()
    points = _node_grid(setup)
    lx, ly, _ = setup.cell_size()

    grounded = points[bidomain.is_grounded(setup, points)]

    along_x = grounded[np.isclose(grounded[:, 1], 0.0)]
    along_y = grounded[np.isclose(grounded[:, 0], 0.0)]
    assert along_x[:, 0].max() == pytest.approx(lx)
    assert along_y[:, 1].max() == pytest.approx(ly)
    # Nothing off the two edges, and nothing beyond one cell along either of them.
    assert np.all(np.isclose(grounded[:, 0], 0.0) | np.isclose(grounded[:, 1], 0.0))
    assert len(grounded) == 2 * setup.elements_per_cell()[0] + 1


# --------------------------------------------------------------------------------------
# the mesh and the fields on it
#
# Everything below needs dolfinx and fenicsx-beat. `sknm` depends on neither, so these skip
# rather than fail; CI runs them in a dolfinx container.
# --------------------------------------------------------------------------------------


def needs_dolfinx(test):
    """Mark a test as one of the continuum ones, and skip it where they cannot run.

    Two things at once, because they always go together: the marker selects these tests, and
    the condition is what lets the rest of the suite stay green on a machine with neither
    package, which is every machine the other CI jobs run on.
    """
    return pytest.mark.skipif(not bidomain.available(), reason=bidomain.REQUIREMENT)(
        pytest.mark.dolfinx(test)
    )


#: A sheet meshed one element to a cell. The paper's own layout, so the cell indexing, the
#: stimulus and the probes are all exercised, at a quarter of the figures' node count.
COARSE = {"element_size": 16e-4}


@needs_dolfinx
def test_the_mesh_covers_the_sheet_at_the_element_size_asked_for():
    setup = bidomain.BidomainSetup(**COARSE)
    mesh = bidomain.build_mesh(setup)

    coordinates = mesh.geometry.x
    width, height = setup.domain_size()
    assert coordinates[:, 0].min() == pytest.approx(0.0)
    assert coordinates[:, 1].min() == pytest.approx(0.0)
    assert coordinates[:, 0].max() == pytest.approx(width)
    assert coordinates[:, 1].max() == pytest.approx(height)
    assert len(coordinates) == 41 * 41


@needs_dolfinx
def test_the_mesh_is_cut_by_one_diagonal_rather_than_two():
    """Two triangles per quadrilateral, not four.

    The crossed pattern adds a node at the centre of every quadrilateral and costs 2.1 times
    as much for a mesh that is finer but no better behaved: measured on this setup the two
    give the same conduction velocity to every digit.
    """
    setup = bidomain.BidomainSetup(**COARSE)
    mesh = bidomain.build_mesh(setup)
    mesh.topology.create_entities(2)

    assert mesh.topology.index_map(2).size_local == 2 * 40 * 40


def _reference_conductivity(setup, midpoints):
    """`M_i` at each of `midpoints`, transcribed from the reference's `Mi_function`.

    Independent of the code under test: it walks one point at a time, indexes the draws the
    way the C does, and writes the expression out in full rather than factoring it.
    """
    from sknm import presets

    lx, ly = presets.hipsc_cell_size(setup.alpha)
    lx, ly = float(lx.m_as("cm")), float(ly.m_as("cm"))
    lz = (1 + setup.delta_e) * ly
    delta_i = 1 - setup.delta_e
    sigma_i = 4.0
    draws = np.loadtxt(bidomain.common.DATA_DIR / "gj_scale_x_BD.txt")

    values = np.zeros((len(midpoints), 2))
    for index, point in enumerate(midpoints):
        cell_x = int(np.floor(point[0] / lx))
        cell_y = int(np.floor(point[1] / ly))
        cell = cell_y * setup.nx + cell_x
        a = draws[cell]
        Rg = 5e3 / (a * (1 - setup.gamma) + (1 - a) * (1 + setup.gamma))
        values[index, 0] = delta_i * sigma_i / (1 + sigma_i * Rg * delta_i * ly * lz / lx)
        values[index, 1] = delta_i * sigma_i / (1 + sigma_i * Rg * delta_i * lx * lz / ly)
    return values


@needs_dolfinx
@pytest.mark.parametrize(("gamma", "alpha"), [(0.0, 1.0), (0.7, 1.0), (0.7, 4.0)])
def test_the_conductivity_field_is_the_reference_s_cell_by_cell(gamma, alpha):
    """Element by element against a transcription of the reference's own function.

    The reference cannot be run here, so the oracle is its source: the same expression,
    written out separately, evaluated at each element's midpoint. Checked at a positive
    variation as well as at none, because at gamma = 0 the draws cancel and a field that
    ignored them entirely would pass.
    """
    import dolfinx

    setup = bidomain.BidomainSetup(gamma=gamma, **COARSE)
    mesh = bidomain.build_mesh(setup)
    field = bidomain.intracellular_conductivity_field(setup, mesh)

    cells = np.arange(mesh.topology.index_map(2).size_local, dtype=np.int32)
    midpoints = dolfinx.mesh.compute_midpoints(mesh, 2, cells)
    expected = _reference_conductivity(setup, midpoints)

    values = field.x.array.reshape(-1, 2, 2)
    chi = bidomain.MEMBRANE_AREA / np.prod(setup.cell_size())

    np.testing.assert_allclose(values[:, 0, 0], expected[:, 0] / chi, rtol=1e-13)
    np.testing.assert_allclose(values[:, 1, 1], expected[:, 1] / chi, rtol=1e-13)
    np.testing.assert_array_equal(values[:, 0, 1], 0.0)
    np.testing.assert_array_equal(values[:, 1, 0], 0.0)


@needs_dolfinx
def test_a_varied_field_really_does_vary_from_cell_to_cell():
    """The guard against a check that would pass on a constant field."""
    setup = bidomain.BidomainSetup(gamma=0.7, **COARSE)
    field = bidomain.intracellular_conductivity_field(setup, bidomain.build_mesh(setup))

    values = field.x.array.reshape(-1, 2, 2)
    assert values[:, 0, 0].std() > 0.0
    assert len(np.unique(np.round(values[:, 0, 0], 12))) == 1600


@needs_dolfinx
def test_the_two_triangles_of_a_cell_carry_the_same_conductivity():
    """What choosing an element size that divides the cell buys.

    Every element lies inside one cell, so the piecewise constant field is exact rather than
    averaged across a boundary. At the reference's own 10 um against a 16 um cell, four out
    of five elements straddle one.
    """
    setup = bidomain.BidomainSetup(gamma=0.7, **COARSE)
    field = bidomain.intracellular_conductivity_field(setup, bidomain.build_mesh(setup))

    values = field.x.array.reshape(-1, 2, 2)[:, 0, 0]
    assert len(np.unique(np.round(values, 12))) == 1600
    assert len(values) == 2 * 1600


# --------------------------------------------------------------------------------------
# the models
# --------------------------------------------------------------------------------------


@needs_dolfinx
def test_the_bidomain_model_grounds_the_extracellular_potential_at_the_corner():
    """A Dirichlet corner, not the zero-mean multiplier `beat` reaches for by default.

    The paper grounds there, and the network model grounds at the same lower-left corner, which
    is what makes the two comparable. The multiplier is also worse on every measured axis here:
    it has no diagonal entry, so an incomplete factorization fails outright on it, and
    truncating the Krylov iteration on the multiplier system moves ``max(u_e) - min(u_e)`` by
    3%, which is the quantity Figure 9's right panel plots.
    """
    setup = bidomain.BidomainSetup(**COARSE)
    model = bidomain.build_model(setup, "bidomain")

    assert model.bcs
    assert model._multiplier is None

    grounded = np.concatenate([bc._cpp_object.dof_indices()[0] for bc in model.bcs])
    coordinates = model.V_ue.tabulate_dof_coordinates()[grounded]
    np.testing.assert_array_equal(bidomain.is_grounded(setup, coordinates), True)
    assert len(grounded) == 3


@needs_dolfinx
@pytest.mark.parametrize("model_name", ["bidomain", "monodomain"])
def test_the_models_step_fully_implicitly(model_name):
    """Backward Euler, as the reference does it: the whole system at the new time."""
    model = bidomain.build_model(bidomain.BidomainSetup(**COARSE), model_name)

    assert model.parameters["theta"] == pytest.approx(1.0)


@needs_dolfinx
def test_the_monodomain_conductivity_is_the_bidomain_one_scaled_by_the_ratio():
    """``lambda/(1 + lambda) * M_i``, the reference's own monodomain coefficient."""
    setup = bidomain.BidomainSetup(gamma=0.7, **COARSE)
    bi = bidomain.build_model(setup, "bidomain")
    mono = bidomain.build_model(setup, "monodomain")

    ratio = setup.conductance_ratio()
    expected = ratio / (1.0 + ratio) * bi._M_i.x.array

    np.testing.assert_allclose(mono._M.x.array, expected, rtol=1e-14)


@needs_dolfinx
def test_the_extracellular_conductivity_is_isotropic_and_carries_the_ratio_out():
    """`M_e` is ``delta_e*sigma_e`` in both directions, with chi divided out as `M_i` has it."""
    setup = bidomain.BidomainSetup(**COARSE)
    model = bidomain.build_model(setup, "bidomain")

    expected = setup.extracellular_conductivity() / setup.surface_to_volume_ratio()
    np.testing.assert_allclose(model._M_e.value, np.diag([expected, expected]), rtol=1e-14)


@needs_dolfinx
def test_an_unknown_model_is_refused():
    with pytest.raises(ValueError, match="bidomain"):
        bidomain.build_model(bidomain.BidomainSetup(**COARSE), "trisomething")


# --------------------------------------------------------------------------------------
# the run
# --------------------------------------------------------------------------------------


@needs_dolfinx
def test_a_run_launches_a_wave_and_measures_its_velocity():
    """One element to a cell, which is coarse for a figure and enough to see a wave.

    The velocity is quoted against the published 3.76 cm/s only loosely here: Table S2 puts a
    20 um mesh about 1% above a 10 um one, and this is 16 um.
    """
    measured = bidomain.run(bidomain.BidomainSetup(**COARSE), "bidomain")

    assert 3.0 < measured.conduction_velocity < 4.5
    assert 15.0 < measured.upstroke_rate < 22.0
    assert measured.peak_potential > 0.0
    assert 0.0 < measured.start_time < measured.end_time < 50.0


@needs_dolfinx
def test_a_sheet_the_wave_never_crosses_has_no_velocity_to_report():
    """Refused rather than returned as a number, since a wave that stalled is not a slow one."""
    with pytest.raises(ValueError, match="did not reach"):
        bidomain.run(bidomain.BidomainSetup(t_end=2.0, **COARSE), "bidomain")


@needs_dolfinx
def test_the_two_models_agree_where_the_monodomain_assumption_holds_exactly():
    """Figure 6's claim, and the sharpest check available on the bidomain wiring.

    At alpha = 1 and gamma = 0 the intracellular conductivity is uniform and isotropic, so a
    single ratio relates it to the extracellular one everywhere and the bidomain system
    reduces to the monodomain one exactly, so the two agree to the precision of the linear
    solves. Anything wrong with the extracellular block, the ground, or the ratio shows up here
    as a difference in the third digit rather than the eighth.

    The upstroke carries the evidence. The velocity is quantized by the time step, since both
    probe times are multiples of it, so two models that differed by a percent could still
    report the same velocity; that the two activation times match exactly says only that the
    difference is below a time step. The upstroke is measured at 8e-9 relative, which is two
    direct factorizations of different matrices disagreeing in the last few bits over four
    thousand steps rather than any difference between the models.
    """
    setup = bidomain.BidomainSetup(**COARSE)

    bi = bidomain.run(setup, "bidomain")
    mono = bidomain.run(setup, "monodomain")

    assert (bi.start_time, bi.end_time) == (mono.start_time, mono.end_time)
    assert bi.upstroke_rate == pytest.approx(mono.upstroke_rate, rel=1e-6)
    # The peak is looser than the upstroke, at 2e-6, because it is a value reached late in the
    # run rather than a difference taken across one step: five thousand steps of rounding have
    # gone into it. It is 5e-5 mV.
    assert bi.peak_potential == pytest.approx(mono.peak_potential, rel=1e-4)


@needs_dolfinx
def test_the_potential_starts_at_the_membrane_model_s_resting_state():
    """Not at zero, which is where a fresh finite element function starts.

    Nothing in a velocity measurement would notice, since the first step overwrites it, but the
    snapshots Figure 6 draws are read off this field, and a first frame of zeros is a frame of
    +77 mV.
    """
    setup = bidomain.BidomainSetup(t_end=2.0, **COARSE)
    seen = []

    with pytest.raises(ValueError, match="did not reach"):
        bidomain.run(setup, "bidomain", callback=lambda t, pde: seen.append(pde.v.x.array.min()))

    # One reading before any step, and one after each of them.
    assert len(seen) == round(2.0 / setup.dt) + 1
    assert -90.0 < seen[0] < -70.0


@needs_dolfinx
def test_a_run_can_be_held_open_past_the_wave_s_arrival():
    """Figure 6 reads the sheet at fixed times, which a run that stopped early might not reach."""
    setup = bidomain.BidomainSetup(**COARSE)
    times = []

    measured = bidomain.run(
        setup, "bidomain", callback=lambda t, pde: times.append(t), minimum_time=45.0
    )

    assert measured.end_time < 45.0
    assert times[-1] == pytest.approx(45.0)


@needs_dolfinx
def test_a_field_can_be_laid_out_as_the_grid_it_was_solved_on():
    """What the snapshots are drawn from.

    A finite element function's degrees of freedom come in the order the mesh numbers them,
    which is not row by row, so a plain reshape draws a scrambled sheet.
    """
    setup = bidomain.BidomainSetup(**COARSE)
    mesh = bidomain.build_mesh(setup)
    model = bidomain.build_model(setup, "monodomain", mesh)

    coordinates = model.V.tabulate_dof_coordinates()
    # A field that is its own x coordinate, so every entry says where it should have landed.
    field = np.ascontiguousarray(coordinates[:, 0])
    laid_out = bidomain.to_grid(setup, model.V, field)

    width, _ = setup.domain_size()
    nx, ny = setup.elements()
    assert laid_out.shape == (ny + 1, nx + 1)
    # The mesh's own coordinate for the origin is a few times 1e-20 rather than zero.
    np.testing.assert_allclose(laid_out[0], np.linspace(0.0, width, nx + 1), atol=1e-15)
    np.testing.assert_allclose(laid_out[-1], laid_out[0], atol=1e-15)


@needs_dolfinx
def test_laying_a_field_out_is_not_a_reshape():
    """The guard against the version of this that looks right and is not.

    If the degrees of freedom happened to be in grid order this test would be vacuous, so it
    asserts that they are not.
    """
    setup = bidomain.BidomainSetup(**COARSE)
    mesh = bidomain.build_mesh(setup)
    model = bidomain.build_model(setup, "monodomain", mesh)

    coordinates = model.V.tabulate_dof_coordinates()
    field = np.ascontiguousarray(coordinates[:, 1])
    nx, ny = setup.elements()

    assert not np.allclose(bidomain.to_grid(setup, model.V, field), field.reshape(ny + 1, nx + 1))


# --------------------------------------------------------------------------------------
# the published tables, and the two implementations against each other
# --------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def network_model_at_the_reference_step(request):
    """`sknm`'s own KNM on the paper's sheet, at the time step the bidomain reference uses.

    Module scoped and parametrized by anisotropy factor, so each of the two runs happens once.
    """
    alpha = request.param
    from sknm import Simulation, Variant, presets
    from sknm.analysis import ActivationRecorder, conduction_velocity
    from sknm.membrane import base_model_IM, from_gotranx
    from sknm.units import ms, mV

    network = presets.hipsc_sheet(40, 40, alpha=alpha)
    membrane = from_gotranx(base_model_IM)
    simulation = Simulation(network, membrane, variant=Variant.KNM, dt=0.01 * ms)
    simulation.set_parameter("stim_amplitude", presets.hipsc_stimulus_amplitude(40, 40))
    path = presets.hipsc_conduction_path(40, 40, alpha=alpha)
    recorder = ActivationRecorder(simulation, threshold=-20 * mV, stop_when_activated=path.end)
    simulation.run(50 * ms, record=(), callback=recorder)

    return (
        float(conduction_velocity(recorder, path).m_as("cm / s")),
        float(recorder.max_upstroke_velocity[presets.hipsc_centre_cell(40, 40)]),
        np.asarray(recorder.activation_time),
        path,
    )


@needs_dolfinx
@pytest.mark.parametrize("network_model_at_the_reference_step", [1.0, 4.0], indirect=True)
def test_the_continuum_and_network_models_agree_on_the_same_tissue(
    request, network_model_at_the_reference_step
):
    """Two independent implementations of the paper's central claim, on one setup.

    `sknm` solves a network of 1600 cells coupled at their junctions; `fenicsx-beat` solves a
    homogenized continuum on 6561 nodes, four to a cell. Neither is derived from the other and
    they share no code below the membrane model, so agreement between them is evidence about
    the models rather than about one implementation.

    They also share a stimulus and a ground: the same two columns of eleven cells, and the same
    lower-left corner, which tightens the comparison enough to make it worth making.

    Run at both ends of the anisotropy sweep. At alpha = 1 the cells are square and the
    conductivity is isotropic; at alpha = 4 they are 40 by 10 um and the two components of the
    conductivity differ by a factor of thirteen, which is the wiring that Figure 7 rests on and
    that nothing else compares against an independent implementation. The measurements come out
    at 0.38% and 0.33%.
    """
    alpha = request.node.callspec.params["network_model_at_the_reference_step"]
    velocity, upstroke, activation, path = network_model_at_the_reference_step
    continuum = bidomain.run(bidomain.BidomainSetup(alpha=alpha), "bidomain")

    assert continuum.conduction_velocity == pytest.approx(velocity, rel=0.015)
    assert continuum.upstroke_rate == pytest.approx(upstroke, rel=0.025)
    # And the wave passes each probe at the same moment, which a velocity alone would not
    # show: a velocity is a difference of two times and survives both being wrong together.
    assert continuum.start_time == pytest.approx(activation[path.start], abs=0.5)
    assert continuum.end_time == pytest.approx(activation[path.end], abs=0.5)


#: Table S3: conduction velocity and maximal upstroke velocity of the bidomain model against
#: the time step. The table is quoted at a 5 um mesh; these run at the 8 um one the figures use,
#: which reproduces it at least as closely at a third of the cost.
TABLE_S3 = {0.02: (3.74, 18.12), 0.01: (3.76, 18.26), 0.002: (3.77, 18.43)}


@pytest.fixture(scope="module")
def table_s3_rows():
    """One bidomain run per row of the table, measured once and shared.

    The three rows together cost about two and a half minutes, nearly all of it the finest
    one: it takes five times the steps of the row above it.
    """
    if not bidomain.available():
        pytest.skip(bidomain.REQUIREMENT)
    return {dt: bidomain.run(bidomain.BidomainSetup(dt=dt), "bidomain") for dt in TABLE_S3}


@needs_dolfinx
@pytest.mark.parametrize("dt", list(TABLE_S3))
def test_the_conduction_velocity_of_table_s3(dt, table_s3_rows):
    """Within 1%, where the measurements come out at 0.52%, 0.08% and 0.13%.

    The tolerance is not tighter because of what sets the floor at the coarse end. A threshold
    crossing is resolved to one step at each end of a transit, and at dt = 0.02 that transit is
    532 steps, so the velocity moves in visible increments of 0.38%; the table's own three
    significant figures are another 0.13%. This is the same tolerance, argued the same way, as
    the network models' conduction velocity carries.
    """
    published, _ = TABLE_S3[dt]

    assert table_s3_rows[dt].conduction_velocity == pytest.approx(published, rel=0.01)


@needs_dolfinx
@pytest.mark.parametrize("dt", list(TABLE_S3))
def test_the_maximal_upstroke_velocity_of_table_s3(dt, table_s3_rows):
    """Within 1.5%, where the measurements come out at 0.93%, 0.24% and -0.17%.

    Tighter than the 2% the network models carry, and it can be: that tolerance covers an
    estimator that disagrees with the reference's by construction, because the network
    reference post-processes a trace written *before* each step's spatial solve. The bidomain
    reference writes its trace *after* the solve, which is where this one reads it, so the two
    estimates are of the same quantity and the tolerance covers only discretization.
    """
    _, published = TABLE_S3[dt]

    assert table_s3_rows[dt].upstroke_rate == pytest.approx(published, rel=0.015)


@needs_dolfinx
def test_both_quantities_converge_as_the_step_shrinks(table_s3_rows):
    """The trend the table is drawn to show, which three independent tolerances would not pin.

    Each row could sit at the far edge of its tolerance in a different direction and every
    assertion above would still pass, leaving a table that reproduces no convergence at all.
    """
    velocities = [table_s3_rows[dt].conduction_velocity for dt in (0.02, 0.01, 0.002)]
    upstrokes = [table_s3_rows[dt].upstroke_rate for dt in (0.02, 0.01, 0.002)]

    assert velocities == sorted(velocities)
    assert upstrokes == sorted(upstrokes)


# --------------------------------------------------------------------------------------
# the scripts
# --------------------------------------------------------------------------------------

#: The figure scripts that need the continuum machinery, and so have to say so when it is
#: absent. Read as text rather than imported: each is a linear script, so importing one runs
#: the simulation it draws.
FIGURE_SCRIPTS = (
    "fig06_continuum_travelling_wave",
    "fig07_continuum_anisotropy",
    "fig08_continuum_gap_junction_variation",
    "fig09_continuum_sources_of_difference",
)


def test_a_missing_requirement_is_reported_rather_than_raised(monkeypatch, capsys):
    """What a machine without dolfinx sees: the install line, and no traceback.

    The scripts are linear, so there is no `main` to return early from; stopping means
    raising `SystemExit`, and it has to carry a zero so that a reader running the file is
    not told their machine is broken.
    """
    monkeypatch.setattr(bidomain, "available", lambda: False)

    with pytest.raises(SystemExit) as stop:
        bidomain.require()

    assert stop.value.code == 0
    assert bidomain.REQUIREMENT in capsys.readouterr().out


def test_a_met_requirement_says_nothing_and_carries_on(monkeypatch, capsys):
    monkeypatch.setattr(bidomain, "available", lambda: True)

    assert bidomain.require() is None
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("name", FIGURE_SCRIPTS)
def test_every_continuum_script_guards_itself_before_it_computes(name):
    """A script that reached `build_mesh` without dolfinx would raise `NameError`, not speak.

    The guard has to come before the first line that touches the machinery, so its position
    is what is checked here, not just that it is there.
    """
    source = (EXAMPLES_DIR / f"{name}.py").read_text()

    assert "bidomain.require()" in source
    first_use = min(
        source.index(token)
        for token in ("bidomain.BidomainSetup", "bidomain.run", "bidomain.snapshots")
        if token in source
    )
    assert source.index("bidomain.require()") < first_use


@needs_dolfinx
def test_the_snapshots_are_taken_at_the_times_asked_for():
    """Figure 6 reads the sheet at three fixed times, and nothing else checks that it did.

    A run stops when the wave reaches the far probe, which is before the last snapshot time,
    so `snapshots` has to hold it open; a callback that fired on the wrong step, or a run that
    ended early, would hand the figure two frames and a repeat rather than three.
    """
    setup = bidomain.BidomainSetup(**COARSE)
    times = (25.0, 30.0, 35.0)

    frames = bidomain.snapshots(setup, "monodomain", times)

    nx, ny = setup.elements()
    assert frames.shape == (len(times), ny + 1, nx + 1)
    assert np.isfinite(frames).all()
    # The sheet is at rest at the first time and depolarized somewhere by the last, and no two
    # frames are the same moment twice.
    assert frames[0].max() > frames[0].min()
    for earlier, later in itertools.pairwise(frames):
        assert not np.allclose(earlier, later)


@needs_dolfinx
def test_a_snapshot_is_the_sheet_at_the_moment_it_names():
    """Which step a time is caught on, anchored against the run rather than against itself.

    The oracle is a run that keeps every step and then picks the one nearest the time asked
    for. That is a different rule from the half-step window `snapshots` uses, so a window that
    slipped by a step would show here and nowhere else.
    """
    setup = bidomain.BidomainSetup(**COARSE)
    target = 25.0

    seen = {}

    def keep(t, pde):
        seen[t] = bidomain.to_grid(setup, pde.V, pde.v.x.array.copy())

    bidomain.run(setup, "monodomain", callback=keep, minimum_time=target)
    nearest = min(seen, key=lambda t: abs(t - target))

    frame = bidomain.snapshots(setup, "monodomain", (target,))[0]

    np.testing.assert_array_equal(frame, seen[nearest])


@needs_dolfinx
def test_a_snapshot_after_the_wave_has_landed_still_gets_taken():
    """A run stops when the wave reaches the far probe, which is before the sheet is done.

    Without holding the run open, a snapshot asked for after that moment is never captured,
    and the figure loses a panel.
    """
    setup = bidomain.BidomainSetup(**COARSE)
    unheld = bidomain.run(setup, "monodomain")
    late = unheld.end_time + 5.0
    assert late < setup.t_end, "the sheet must still be running at the time asked for"

    frames = bidomain.snapshots(setup, "monodomain", (late,))

    assert frames.shape[0] == 1


@needs_dolfinx
def test_a_run_too_short_for_the_snapshots_asked_for_is_refused():
    """Silently returning fewer frames than asked for would draw a figure with a missing panel."""
    setup = bidomain.BidomainSetup(**COARSE)

    with pytest.raises(ValueError, match="900"):
        bidomain.snapshots(setup, "monodomain", (25.0, 900.0))
