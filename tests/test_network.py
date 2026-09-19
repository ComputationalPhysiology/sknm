"""The cell network: topology, geometry, conductances and the Laplacians built from them.

The anchor is `test_laplacian_matches_the_reference_stencil`: the reference implementation
assembles its matrices by grid index arithmetic, and this suite rebuilds those matrices that
way, by hand, and demands the graph formulation reproduce them element for element.
"""

import dataclasses

import numpy as np
import pytest

from sknm import CellNetwork, Connection, chain, from_edges, sheet
from sknm.units import cm, mS, um

# The paper's hiPSC-CM conductivities and gap junction resistance.
DELTA_E = 0.2
SIGMA_I = 4.0
SIGMA_E = 20.0
RG = 5e3
GG = 1 / RG


def paper_geometry(lx_um, ly_um):
    """One of the paper's measured (lx, ly) cell sizes, as constructor keywords."""
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


# alpha = 1. Square cells, so x- and y-connections come out identical -- which is exactly why
# most of the geometry below is checked on the anisotropic sheet instead.
ISOTROPIC = paper_geometry(16, 16)
# alpha = 1.5. Distinct lengths, cross-sections and shape factors per direction.
ANISOTROPIC = paper_geometry(21, 14)

LX = ISOTROPIC["lx"]
LY = ISOTROPIC["ly"]
LZ = ISOTROPIC["lz"]


def reference_stencil_laplacians(nx, ny, lx, ly, lz, delta_e, sigma_i, sigma_e, Rg):
    """Assemble L_i and L_ie the way the reference C++ does, by grid index arithmetic.

    A transcription of the `Mi_xp` / `Mi_xm` / `Mi_yp` / `Mi_ym` loop and of the matrix
    construction that consumes it, kept deliberately literal -- including the `0.5*(d+d)`
    averaging that is redundant on a uniform sheet -- so that the comparison is against the
    reference's own arithmetic rather than a tidied-up restatement of it.
    """
    n = nx * ny
    de = np.full(n, delta_e)
    di = 1.0 - de
    xp, xm, yp, ym = (np.zeros(n) for _ in range(4))
    exp_, exm, eyp, eym = (np.zeros(n) for _ in range(4))
    for j in range(ny):
        for i in range(nx):
            idx = j * nx + i
            if i < nx - 1:
                xp[idx] = 1 / (lx / (0.5 * (di[idx] + di[idx + 1]) * ly * lz * sigma_i) + Rg)
                exp_[idx] = xp[idx] + 0.5 * (de[idx] + de[idx + 1]) * ly * lz * sigma_e / lx
            if i > 0:
                xm[idx] = 1 / (lx / (0.5 * (di[idx] + di[idx - 1]) * ly * lz * sigma_i) + Rg)
                exm[idx] = xm[idx] + 0.5 * (de[idx] + de[idx - 1]) * ly * lz * sigma_e / lx
            if j < ny - 1:
                yp[idx] = 1 / (ly / (0.5 * (di[idx] + di[idx + nx]) * lx * lz * sigma_i) + Rg)
                eyp[idx] = yp[idx] + 0.5 * (de[idx] + de[idx + nx]) * lx * lz * sigma_e / ly
            if j > 0:
                ym[idx] = 1 / (ly / (0.5 * (di[idx] + di[idx - nx]) * lx * lz * sigma_i) + Rg)
                eym[idx] = ym[idx] + 0.5 * (de[idx] + de[idx - nx]) * lx * lz * sigma_e / ly

    def assemble(p_x, m_x, p_y, m_y):
        matrix = np.zeros((n, n))
        for j in range(ny):
            for i in range(nx):
                idx = j * nx + i
                matrix[idx, idx] = m_x[idx] + p_x[idx] + m_y[idx] + p_y[idx]
                if i < nx - 1:
                    matrix[idx, idx + 1] = -p_x[idx]
                if i > 0:
                    matrix[idx, idx - 1] = -m_x[idx]
                if j < ny - 1:
                    matrix[idx, idx + nx] = -p_y[idx]
                if j > 0:
                    matrix[idx, idx - nx] = -m_y[idx]
        return matrix

    return assemble(xp, xm, yp, ym), assemble(exp_, exm, eyp, eym)


@pytest.fixture
def paper_sheet():
    return sheet(4, 3, **ISOTROPIC)


@pytest.mark.parametrize("geometry", [ISOTROPIC, ANISOTROPIC], ids=["alpha1", "alpha1.5"])
@pytest.mark.parametrize("kind", ["intra", "intra_extra"])
def test_laplacian_matches_the_reference_stencil(geometry, kind):
    network = sheet(4, 3, **geometry)
    intra, intra_extra = reference_stencil_laplacians(
        4, 3, geometry["lx"], geometry["ly"], geometry["lz"], DELTA_E, SIGMA_I, SIGMA_E, RG
    )
    expected = intra if kind == "intra" else intra_extra
    np.testing.assert_allclose(network.laplacian(kind).toarray(), expected, rtol=1e-14)


@pytest.mark.parametrize("kind", ["intra", "intra_extra"])
def test_laplacian_is_symmetric_with_zero_row_sums_and_positive_semidefinite(paper_sheet, kind):
    dense = paper_sheet.laplacian(kind).toarray()
    scale = float(np.abs(dense).max())
    np.testing.assert_allclose(dense, dense.T, rtol=0, atol=1e-14 * scale)
    np.testing.assert_allclose(dense.sum(axis=1), 0.0, atol=1e-14 * scale)
    assert np.linalg.eigvalsh(dense).min() > -1e-12 * scale


def test_laplacian_rejects_an_unknown_kind(paper_sheet):
    with pytest.raises(ValueError, match="intra_extra"):
        paper_sheet.laplacian("extra")


def test_laplacian_is_the_incidence_triple_product(paper_sheet):
    """`L = A^T diag(G) A`, with one incidence matrix shared by both kinds."""
    incidence = paper_sheet.incidence.toarray()
    assert incidence.shape == (paper_sheet.n_connections, paper_sheet.n_cells)
    for kind, conductance in [
        ("intra", paper_sheet.Gi),
        ("intra_extra", paper_sheet.Gi + paper_sheet.Ge),
    ]:
        expected = incidence.T @ np.diag(conductance) @ incidence
        np.testing.assert_allclose(paper_sheet.laplacian(kind).toarray(), expected, rtol=1e-14)


def test_lam_on_the_paper_sheet_is_the_uniform_conductance_ratio(paper_sheet):
    """Uniform conductances make the least-squares fit of Ge ~ lam * Gi exact.

    Square cells give every connection the same conductances, so the fit collapses to a
    single ratio and the published value can be asserted to roundoff.
    """
    ratio = paper_sheet.Ge / paper_sheet.Gi
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-14)
    assert paper_sheet.lam == pytest.approx(ratio[0], rel=1e-14)
    assert paper_sheet.lam == pytest.approx(39.65, rel=1e-12)


def test_lam_weights_each_connection_by_its_squared_shape_factor():
    """Paper eq. 30 weights by (l/A)^2, and the square is not decorative.

    On an anisotropic sheet the two directions have different shape factors, so dropping the
    square moves the answer -- which it cannot do on the square-celled sheet above.
    """
    network = sheet(4, 3, **ANISOTROPIC)
    shape_factor = network.length / network.cross_section
    assert len(np.unique(shape_factor)) == 2
    squared = np.sum(network.Ge * network.Gi * shape_factor**2) / np.sum(
        network.Gi**2 * shape_factor**2
    )
    unsquared = np.sum(network.Ge * network.Gi * shape_factor) / np.sum(
        network.Gi**2 * shape_factor
    )
    assert squared != pytest.approx(unsquared, rel=1e-6)
    assert network.lam == pytest.approx(squared, rel=1e-14)


def test_conductance_ratio_is_an_alias_for_lam(paper_sheet):
    assert paper_sheet.conductance_ratio == paper_sheet.lam


def test_lam_can_be_overridden(paper_sheet):
    overridden = sheet(4, 3, lam_override=2.5, **ISOTROPIC)
    assert overridden.lam == 2.5
    assert paper_sheet.lam != 2.5


def test_lam_is_a_weighted_least_squares_fit_when_conductances_differ():
    """Paper eq. 30: lam = sum(Ge*Gi*(l/A)^2) / sum(Gi^2*(l/A)^2), not a plain mean of ratios."""
    network = from_edges(
        [
            Connection((0, 1), length=LX, cross_section=LY * LZ, Gg=GG),
            Connection((1, 2), length=LX, cross_section=LY * LZ, Gg=GG / 10),
        ],
        membrane_area=1.8e-5,
        delta_e=DELTA_E,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    weight = (network.length / network.cross_section) ** 2
    expected = np.sum(network.Ge * network.Gi * weight) / np.sum(network.Gi**2 * weight)
    assert network.lam == pytest.approx(expected, rel=1e-14)
    assert network.lam != pytest.approx(np.mean(network.Ge / network.Gi))


def test_lam_refuses_a_network_with_no_intracellular_coupling():
    network = chain(3, **ISOTROPIC | {"Gg": 0.0})
    with pytest.raises(ValueError, match="lam_override"):
        _ = network.lam


def test_conductances_average_the_volume_fractions_of_both_endpoint_cells():
    """The mixing of both cells' delta is why this lives on the network, not on `Connection`."""
    network = from_edges(
        [Connection((0, 1), length=LX, cross_section=LY * LZ, Gg=GG)],
        membrane_area=1.8e-5,
        delta_e=[0.1, 0.3],
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    mean_delta_i = 0.5 * (0.9 + 0.7)
    mean_delta_e = 0.5 * (0.1 + 0.3)
    expected_Gi = 1 / (LX / (mean_delta_i * LY * LZ * SIGMA_I) + RG)
    expected_Ge = mean_delta_e * LY * LZ * SIGMA_E / LX
    assert network.Gi[0] == pytest.approx(expected_Gi, rel=1e-14)
    assert network.Ge[0] == pytest.approx(expected_Ge, rel=1e-14)


def test_a_zero_gap_junction_conductance_blocks_intracellular_current():
    """A connection whose gap junctions are shut still carries extracellular current."""
    network = from_edges(
        [Connection((0, 1), length=LX, cross_section=LY * LZ, Gg=0.0)],
        membrane_area=1.8e-5,
        delta_e=DELTA_E,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    assert network.Gi[0] == 0.0
    assert network.Ge[0] > 0.0
    np.testing.assert_array_equal(network.laplacian("intra").toarray(), np.zeros((2, 2)))


def test_chain_connects_each_cell_to_the_next():
    network = chain(5, **ISOTROPIC)
    assert network.n_cells == 5
    assert network.n_connections == 4
    np.testing.assert_array_equal(network.connections, [[0, 1], [1, 2], [2, 3], [3, 4]])
    np.testing.assert_allclose(network.length, LX)
    np.testing.assert_allclose(network.cross_section, LY * LZ)


def test_a_two_cell_chain_has_a_single_connection():
    network = chain(2, **ISOTROPIC)
    assert network.n_cells == 2
    assert network.n_connections == 1
    np.testing.assert_array_equal(network.connections, [[0, 1]])


def test_sheet_lays_out_x_connections_before_y_connections():
    """Row-major cell numbering and x-then-y connection ordering, matching the reference.

    Checked on the anisotropic geometry, where a connection's length and cross-section
    actually depend on which direction it runs in.
    """
    lx, ly, lz = ANISOTROPIC["lx"], ANISOTROPIC["ly"], ANISOTROPIC["lz"]
    network = sheet(3, 2, **ANISOTROPIC)
    assert network.n_cells == 6
    assert network.n_connections == (3 - 1) * 2 + 3 * (2 - 1)
    np.testing.assert_array_equal(
        network.connections,
        [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]],
    )
    np.testing.assert_allclose(network.length[:4], lx)
    np.testing.assert_allclose(network.cross_section[:4], ly * lz)
    np.testing.assert_allclose(network.length[4:], ly)
    np.testing.assert_allclose(network.cross_section[4:], lx * lz)


def test_sheet_accepts_a_per_connection_gap_junction_conductance():
    network = sheet(3, 2, **ISOTROPIC | {"Gg": np.arange(7) * GG})
    np.testing.assert_allclose(network.Gg, np.arange(7) * GG)


def test_sheet_defaults_the_membrane_area_to_the_cell_surface_area():
    network = sheet(2, 2, **ISOTROPIC)
    expected = 2 * (LX * LY + LX * LZ + LY * LZ)
    np.testing.assert_allclose(network.membrane_area, expected)
    explicit = sheet(2, 2, membrane_area=1.8e-5, **ISOTROPIC)
    np.testing.assert_allclose(explicit.membrane_area, 1.8e-5)


def test_from_edges_infers_the_cell_count_from_the_connections():
    network = from_edges(
        [Connection((0, 2), length=LX, cross_section=LY * LZ, Gg=GG)],
        membrane_area=1.8e-5,
        delta_e=DELTA_E,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    assert network.n_cells == 3


def test_from_edges_accepts_isolated_cells_through_an_explicit_count():
    network = from_edges(
        [Connection((0, 1), length=LX, cross_section=LY * LZ, Gg=GG)],
        n_cells=4,
        membrane_area=1.8e-5,
        delta_e=DELTA_E,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    assert network.n_cells == 4
    np.testing.assert_array_equal(network.component_labels, [0, 0, 1, 2])


def test_component_labels_separate_a_disconnected_network():
    """The ground for the extracellular potential is picked per component, so this must be right."""
    network = from_edges(
        [
            Connection((0, 1), length=LX, cross_section=LY * LZ, Gg=GG),
            Connection((1, 2), length=LX, cross_section=LY * LZ, Gg=GG),
            Connection((3, 4), length=LX, cross_section=LY * LZ, Gg=GG),
        ],
        membrane_area=1.8e-5,
        delta_e=DELTA_E,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    np.testing.assert_array_equal(network.component_labels, [0, 0, 0, 1, 1])


def test_a_connected_sheet_is_one_component(paper_sheet):
    np.testing.assert_array_equal(paper_sheet.component_labels, 0)


def test_rebinding_a_field_raises(paper_sheet):
    with pytest.raises(dataclasses.FrozenInstanceError):
        paper_sheet.sigma_i = 8.0


@pytest.mark.parametrize(
    "name", ["membrane_area", "delta_e", "connections", "length", "cross_section", "Gg"]
)
def test_writing_into_an_array_raises(paper_sheet, name):
    """`frozen=True` blocks rebinding but not in-place mutation, so the arrays are stamped."""
    with pytest.raises(ValueError, match="read-only"):
        getattr(paper_sheet, name)[0] = 1


def test_the_network_copies_its_inputs_rather_than_freezing_the_callers_arrays():
    delta_e = np.full(3, DELTA_E)
    network = from_edges(
        [Connection((0, 1), length=LX, cross_section=LY * LZ, Gg=GG)],
        n_cells=3,
        membrane_area=1.8e-5,
        delta_e=delta_e,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    delta_e[0] = 0.5
    assert network.delta_e[0] == DELTA_E


def test_with_conductances_returns_a_new_network_and_leaves_the_original_alone(paper_sheet):
    doubled = paper_sheet.with_conductances(paper_sheet.Gg * 2)
    assert doubled is not paper_sheet
    np.testing.assert_allclose(doubled.Gg, 2 * GG)
    np.testing.assert_allclose(paper_sheet.Gg, GG)
    assert doubled.Gi[0] > paper_sheet.Gi[0]
    assert doubled.lam < paper_sheet.lam
    assert paper_sheet.lam == pytest.approx(39.65, rel=1e-12)


def test_with_conductances_keeps_the_rest_of_the_geometry(paper_sheet):
    doubled = paper_sheet.with_conductances(paper_sheet.Gg * 2)
    np.testing.assert_array_equal(doubled.connections, paper_sheet.connections)
    np.testing.assert_allclose(doubled.membrane_area, paper_sheet.membrane_area)
    np.testing.assert_allclose(doubled.length, paper_sheet.length)


def test_with_conductances_checks_the_length(paper_sheet):
    with pytest.raises(ValueError, match="Gg"):
        paper_sheet.with_conductances(np.ones(3))


def test_arrays_are_float64(paper_sheet):
    for name in ["membrane_area", "delta_e", "length", "cross_section", "Gg"]:
        assert getattr(paper_sheet, name).dtype == np.float64
    assert paper_sheet.connections.dtype == np.int64


def test_repr_does_not_dump_the_arrays(paper_sheet):
    text = repr(paper_sheet)
    assert "n_cells=12" in text
    assert "n_connections=17" in text
    assert "\n" not in text


def test_a_sixteen_centimetre_cell_is_rejected():
    """Passing 16 rather than `16 * um` is the mistake this catches."""
    with pytest.raises(ValueError, match="implausible"):
        sheet(3, 3, **ISOTROPIC | {"lx": 16 * cm, "ly": 16 * cm, "lz": 16 * cm})


def test_an_implausible_connection_is_rejected_by_from_edges():
    with pytest.raises(ValueError, match="implausible"):
        from_edges(
            [Connection((0, 1), length=16 * cm, cross_section=LY * LZ, Gg=GG)],
            membrane_area=1.8e-5,
            delta_e=DELTA_E,
            sigma_i=SIGMA_I,
            sigma_e=SIGMA_E,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"delta_e": 0.0}, "delta_e"),
        ({"delta_e": 1.0}, "delta_e"),
        ({"delta_e": -0.1}, "delta_e"),
        ({"Gg": -1.0 * mS}, "Gg"),
        ({"sigma_i": 0.0}, "sigma_i"),
        ({"sigma_e": -1.0}, "sigma_e"),
        ({"lx": 0.0}, "lx"),
        ({"lz": -LY}, "lz"),
    ],
)
def test_implausible_parameters_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sheet(3, 3, **ISOTROPIC | kwargs)


def test_an_infinite_conductance_is_rejected():
    """An infinite Gg passes every positivity check and then turns Gi into a silent nan."""
    with pytest.raises(ValueError, match="Gg must be finite"):
        sheet(3, 3, **ISOTROPIC | {"Gg": np.inf})


def test_a_connection_to_a_cell_that_does_not_exist_is_rejected():
    with pytest.raises(ValueError, match="cell index"):
        from_edges(
            [Connection((0, 5), length=LX, cross_section=LY * LZ, Gg=GG)],
            n_cells=2,
            membrane_area=1.8e-5,
            delta_e=DELTA_E,
            sigma_i=SIGMA_I,
            sigma_e=SIGMA_E,
        )


def test_a_cell_cannot_connect_to_itself():
    with pytest.raises(ValueError, match="itself"):
        Connection((1, 1), length=LX, cross_section=LY * LZ, Gg=GG)


def test_mismatched_per_cell_arrays_are_rejected():
    with pytest.raises(ValueError, match="delta_e"):
        CellNetwork(
            membrane_area=np.full(3, 1.8e-5),
            delta_e=np.full(2, DELTA_E),
            connections=np.empty((0, 2), dtype=np.int64),
            length=np.empty(0),
            cross_section=np.empty(0),
            Gg=np.empty(0),
            sigma_i=SIGMA_I,
            sigma_e=SIGMA_E,
        )


def test_a_network_with_no_connections_has_an_empty_laplacian():
    network = from_edges(
        [],
        n_cells=3,
        membrane_area=1.8e-5,
        delta_e=DELTA_E,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )
    assert network.n_connections == 0
    np.testing.assert_array_equal(network.laplacian("intra").toarray(), np.zeros((3, 3)))
    np.testing.assert_array_equal(network.component_labels, [0, 1, 2])
