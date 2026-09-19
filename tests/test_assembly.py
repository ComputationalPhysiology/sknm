"""The variant operators: the matrix each model solves, and the bookkeeping around it.

Two tests carry the weight. `test_the_symmetrized_system_reproduces_the_reference_formulation`
solves the reference implementation's own nonsymmetric block form alongside ours and demands
the same membrane potential, which is what makes the row scaling a change of spelling rather
than a change of model. `test_two_cells_relax_by_the_exact_backward_euler_factor` pins each
variant's effective coupling against a closed form derived by hand, which is what stops all
three variants agreeing on something wrong.

Everything here is time-free and exact: no membrane model, no time loop, no tolerance that
depends on a convergence rate except in the one test that deliberately measures one.
"""

import dataclasses

import numpy as np
import pint
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import eigh

from sknm import Connection, Operator, Variant, assemble, chain, from_edges, sheet
from sknm.units import cm, kohm, mS, ms, uF, um

# The paper's hiPSC-CM conductivities and gap junction resistance.
DELTA_E = 0.2
SIGMA_I = 4.0 * mS / cm
SIGMA_E = 20.0 * mS / cm
GG = (1 / (5e3 * kohm)).to(mS)
MEMBRANE_AREA = 1.8e-5 * cm**2
DT = 0.02 * ms

VARIANTS = list(Variant)


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


# alpha = 1.5, so x- and y-connections differ in length, cross-section and conductance. The
# square alpha = 1 cells cannot tell the two directions apart, which hides mistakes.
ANISOTROPIC = paper_geometry(21, 14)


@pytest.fixture
def uniform_chain():
    """A short strand of identical cells. Every connection has the same Gi and the same Ge."""
    return chain(5, membrane_area=MEMBRANE_AREA, **paper_geometry(16, 16))


@pytest.fixture
def varied_sheet():
    """A small sheet with a different membrane area on every cell and one shut connection.

    The varying area is what makes `diag(dt / (Cm * Am))` a genuine scaling rather than a
    multiple of the identity, so it is the case in which the reference's operator is really
    nonsymmetric and ours is really symmetric. The shut gap junction leaves a connection that
    carries extracellular current and no intracellular current.
    """
    nx, ny = 3, 4
    geometry = dict(ANISOTROPIC)
    n_connections = (nx - 1) * ny + nx * (ny - 1)
    conductances = np.linspace(0.5, 1.5, n_connections)
    conductances[2] = 0.0
    geometry["Gg"] = conductances * GG
    areas = np.linspace(1.0, 2.5, nx * ny) * MEMBRANE_AREA
    return sheet(nx, ny, membrane_area=areas, **geometry)


def two_cells(**overrides):
    """Two cells joined by one connection, with the paper's geometry unless told otherwise."""
    return chain(2, membrane_area=MEMBRANE_AREA, **{**paper_geometry(16, 16), **overrides})


def disconnected_pairs():
    """Four cells in two components: 0-1 and 2-3."""
    connection = {
        "length": 16 * um,
        "cross_section": 16 * um * 19.2 * um,
        "Gg": GG,
    }
    return from_edges(
        [Connection((0, 1), **connection), Connection((2, 3), **connection)],
        membrane_area=MEMBRANE_AREA,
        delta_e=DELTA_E,
        sigma_i=SIGMA_I,
        sigma_e=SIGMA_E,
    )


def reference_system(network, dt_ms, variant, ground=0):
    """The reference implementation's own block form, dense and nonsymmetric.

    A transcription of the C++'s `MiI` / `Mi` / `Mib` / `Mie` construction: block row zero
    divided through by the membrane capacitance, and the Dirichlet condition imposed by
    emptying one row of the lower-left block and stamping a unit row into the lower-right one.
    That stamped row is what costs the reference its symmetry.

    Returns the matrix and the right-hand side for one step from `v_prev`, as a function of it.
    """
    n = network.n_cells
    intra = network.laplacian("intra").toarray()
    intra_extra = network.laplacian("intra_extra").toarray()
    factor = dt_ms / (network.Cm * network.membrane_area)
    if variant is Variant.SKNM:
        factor = factor * network.lam / (1.0 + network.lam)

    identity_block = np.eye(n) + factor[:, None] * intra
    if variant is not Variant.KNM:
        return identity_block, lambda v_prev: np.asarray(v_prev, dtype=np.float64)

    coupling = factor[:, None] * intra
    dirichlet_intra = intra.copy()
    dirichlet_intra[ground] = 0.0
    dirichlet_extra = intra_extra.copy()
    dirichlet_extra[ground] = 0.0
    dirichlet_extra[ground, ground] = 1.0
    matrix = np.block([[identity_block, coupling], [dirichlet_intra, dirichlet_extra]])
    return matrix, lambda v_prev: np.concatenate([v_prev, np.zeros(n)])


def step(operator, v_prev):
    """Advance the membrane potential one step through an assembled operator."""
    solution = spla.spsolve(sp.csc_array(operator.matrix), operator.rhs(v_prev))
    return operator.membrane_potential(solution)


def ramp(n_cells):
    """A deterministic non-constant starting potential, in mV."""
    return np.linspace(-80.0, 20.0, n_cells)


# --- The operator itself ------------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_operator_is_symmetric_and_positive_definite(varied_sheet, variant):
    dense = assemble(varied_sheet, dt=DT, variant=variant).matrix.toarray()
    assert np.array_equal(dense, dense.T)
    assert eigh(dense, eigvals_only=True).min() > 0.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_symmetrized_system_reproduces_the_reference_formulation(varied_sheet, variant):
    v_prev = ramp(varied_sheet.n_cells)
    matrix, rhs_of = reference_system(varied_sheet, DT.m_as("ms"), variant)
    expected = np.linalg.solve(matrix, rhs_of(v_prev))[: varied_sheet.n_cells]

    ours = step(assemble(varied_sheet, dt=DT, variant=variant), v_prev)

    np.testing.assert_allclose(ours, expected, rtol=1e-11)


def test_the_symmetrized_system_reproduces_the_reference_extracellular_potential(varied_sheet):
    """The membrane potential alone cannot see a sign error in the extracellular block.

    Negating both off-diagonal blocks is the substitution ``u_e -> -u_e``, which leaves the
    membrane potential exactly as it was. Only comparing the extracellular potential itself
    catches it -- and that potential is what the paper's own figures plot.
    """
    v_prev = ramp(varied_sheet.n_cells)
    matrix, rhs_of = reference_system(varied_sheet, DT.m_as("ms"), Variant.KNM)
    expected = np.linalg.solve(matrix, rhs_of(v_prev))[varied_sheet.n_cells :]

    operator = assemble(varied_sheet, dt=DT, variant=Variant.KNM)
    solution = spla.spsolve(sp.csc_array(operator.matrix), operator.rhs(v_prev))
    ours = operator.extracellular_potential(solution)

    assert np.abs(expected).max() > 0.0
    np.testing.assert_allclose(ours, expected, rtol=1e-9)


def test_sknm_scales_the_capacitive_diagonal_by_one_plus_lambda_over_lambda(varied_sheet):
    simplified = assemble(varied_sheet, dt=DT, variant=Variant.SKNM)
    zero_extracellular = assemble(varied_sheet, dt=DT, variant=Variant.SKNM_UE0)
    lam = varied_sheet.lam

    np.testing.assert_allclose(simplified.D, (1 + lam) / lam * zero_extracellular.D, rtol=1e-14)


def test_sknm_and_sknm_ue0_differ_only_in_that_diagonal(varied_sheet):
    simplified = assemble(varied_sheet, dt=DT, variant=Variant.SKNM)
    zero_extracellular = assemble(varied_sheet, dt=DT, variant=Variant.SKNM_UE0)

    difference = simplified.matrix.toarray() - zero_extracellular.matrix.toarray()
    np.testing.assert_allclose(
        np.diag(difference), zero_extracellular.D / varied_sheet.lam, rtol=1e-14
    )
    off_diagonal = difference - np.diag(np.diag(difference))
    assert np.array_equal(off_diagonal, np.zeros_like(off_diagonal))


def test_the_capacitive_diagonal_is_inversely_proportional_to_the_step(varied_sheet):
    coarse = assemble(varied_sheet, dt=0.02 * ms, variant=Variant.SKNM_UE0)
    fine = assemble(varied_sheet, dt=0.01 * ms, variant=Variant.SKNM_UE0)

    np.testing.assert_allclose(fine.D, 2 * coarse.D, rtol=1e-14)
    np.testing.assert_allclose(coarse.D, varied_sheet.Cm * varied_sheet.membrane_area / 0.02)


# --- Against closed forms -----------------------------------------------------------------


def effective_conductance(network, variant):
    """The conductance through which two cells exchange current, under a given variant.

    Derived by hand from each variant's system on a two-cell network: SKNM(u_e=0) couples
    through the intracellular conductance alone, KNM through the intracellular and
    extracellular conductances in series, and SKNM through the intracellular conductance
    reduced by `lam / (1 + lam)`.
    """
    intra, extra = network.Gi[0], network.Ge[0]
    if variant is Variant.SKNM_UE0:
        return intra
    if variant is Variant.KNM:
        return intra * extra / (intra + extra)
    return intra * network.lam / (1.0 + network.lam)


@pytest.mark.parametrize("variant", VARIANTS)
def test_two_cells_relax_by_the_exact_backward_euler_factor(variant):
    network = two_cells()
    dt_ms = DT.m_as("ms")
    capacitance = network.Cm * network.membrane_area[0] / dt_ms
    coupling = effective_conductance(network, variant)
    expected = capacitance / (capacitance + 2 * coupling)

    v_prev = np.array([-80.0, 20.0])
    v = step(assemble(network, dt=DT, variant=variant), v_prev)

    difference = (v[0] - v[1]) / (v_prev[0] - v_prev[1])
    assert difference == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("variant", VARIANTS)
def test_two_cells_relax_towards_the_analytic_exponential(variant):
    network = two_cells()
    coupling = effective_conductance(network, variant)
    tau = network.Cm * network.membrane_area[0] / (2 * coupling)

    dt = 1e-4 * ms
    n_steps = round(tau / dt.m_as("ms"))
    operator = assemble(network, dt=dt, variant=variant)
    v = np.array([-80.0, 20.0])
    initial_gap = v[0] - v[1]
    for _ in range(n_steps):
        v = step(operator, v)

    # Backward Euler is first order, so the gap to the exponential is O(dt / tau) over one
    # time constant -- here about 5e-4 -- and the tolerance is set just above it.
    assert (v[0] - v[1]) / initial_gap == pytest.approx(np.exp(-1.0), rel=2e-3)


def test_two_cells_have_the_analytic_extracellular_potential():
    """Intracellular current out of the leading cell returns through the extracellular space.

    On two cells grounded at the first, the lower block of the KNM system reads
    ``Gi * (v0 - v1) = (Gi + Ge) * u_e1``, so the extracellular potential is the membrane
    potential difference divided between the two conductances -- and it is positive where the
    first cell is the more depolarized one, because that is the direction the return current
    takes.
    """
    network = two_cells()
    operator = assemble(network, dt=DT, variant=Variant.KNM)
    v_prev = np.array([20.0, -80.0])

    solution = spla.spsolve(sp.csc_array(operator.matrix), operator.rhs(v_prev))
    v = operator.membrane_potential(solution)
    u = operator.extracellular_potential(solution)

    expected = network.Gi[0] * (v[0] - v[1]) / (network.Gi[0] + network.Ge[0])
    assert u[0] == 0.0
    assert expected > 0.0
    assert u[1] == pytest.approx(expected, rel=1e-11)


def test_sknm_reproduces_knm_where_the_conductance_ratio_holds_exactly(uniform_chain):
    """Every connection of a uniform network has the same Ge/Gi, which is `lam` exactly.

    The extracellular Laplacian is then `lam` times the intracellular one, which is precisely
    the assumption SKNM makes, so eliminating the extracellular potential is exact rather than
    approximate and the two variants must agree on the membrane potential.
    """
    ratio = uniform_chain.Ge / uniform_chain.Gi
    np.testing.assert_allclose(ratio, uniform_chain.lam, rtol=1e-14)

    v_prev = ramp(uniform_chain.n_cells)
    full = step(assemble(uniform_chain, dt=DT, variant=Variant.KNM), v_prev)
    simplified = step(assemble(uniform_chain, dt=DT, variant=Variant.SKNM), v_prev)

    np.testing.assert_allclose(simplified, full, rtol=1e-11)


def test_sknm_and_sknm_ue0_disagree_when_the_extracellular_space_matters(uniform_chain):
    v_prev = ramp(uniform_chain.n_cells)
    simplified = step(assemble(uniform_chain, dt=DT, variant=Variant.SKNM), v_prev)
    zero_extracellular = step(assemble(uniform_chain, dt=DT, variant=Variant.SKNM_UE0), v_prev)

    assert not np.allclose(simplified, zero_extracellular, rtol=1e-6)


# --- Grounding ----------------------------------------------------------------------------


def test_knm_removes_one_extracellular_dof(uniform_chain):
    operator = assemble(uniform_chain, dt=DT, variant=Variant.KNM)

    assert operator.n_dofs == 2 * uniform_chain.n_cells - 1
    np.testing.assert_array_equal(operator.ground, [0])


def test_knm_removes_one_extracellular_dof_per_component():
    network = disconnected_pairs()
    operator = assemble(network, dt=DT, variant=Variant.KNM)

    assert operator.n_dofs == 2 * network.n_cells - 2
    np.testing.assert_array_equal(operator.ground, [0, 2])


@pytest.mark.parametrize("variant", [Variant.SKNM, Variant.SKNM_UE0])
def test_the_simplified_variants_have_no_extracellular_dof_to_ground(uniform_chain, variant):
    operator = assemble(uniform_chain, dt=DT, variant=variant)

    assert operator.n_dofs == uniform_chain.n_cells
    assert operator.ground.size == 0


def test_an_explicit_ground_is_honoured(uniform_chain):
    operator = assemble(uniform_chain, dt=DT, variant=Variant.KNM, ground=3)

    np.testing.assert_array_equal(operator.ground, [3])
    v_prev = ramp(uniform_chain.n_cells)
    solution = spla.spsolve(sp.csc_array(operator.matrix), operator.rhs(v_prev))
    assert operator.extracellular_potential(solution)[3] == 0.0


def test_the_ground_shifts_the_extracellular_potential_by_a_constant_and_leaves_v_alone(
    uniform_chain,
):
    """The ground is a gauge on the extracellular potential; it cannot reach the membrane.

    Exactly true in algebra. Numerically, grounding a different cell deletes a different row
    and column and so factorizes a different matrix, which moves the membrane potential by a
    unit in the last place -- hence roundoff rather than equality.
    """
    v_prev = ramp(uniform_chain.n_cells)
    potentials = []
    for ground in (0, 3):
        operator = assemble(uniform_chain, dt=DT, variant=Variant.KNM, ground=ground)
        solution = spla.spsolve(sp.csc_array(operator.matrix), operator.rhs(v_prev))
        potentials.append(
            (operator.membrane_potential(solution), operator.extracellular_potential(solution))
        )

    (v_first, u_first), (v_second, u_second) = potentials
    np.testing.assert_allclose(v_second, v_first, rtol=1e-14)

    offset = u_second - u_first
    np.testing.assert_allclose(offset, offset[0], atol=1e-14 * np.abs(u_first).max())
    assert offset[0] != 0.0


def test_a_ground_outside_the_network_is_refused(uniform_chain):
    with pytest.raises(ValueError, match="cell index 9"):
        assemble(uniform_chain, dt=DT, variant=Variant.KNM, ground=9)


def test_a_ground_that_misses_a_component_is_refused():
    with pytest.raises(ValueError, match="one cell per connected component"):
        assemble(disconnected_pairs(), dt=DT, variant=Variant.KNM, ground=0)


def test_a_ground_that_pins_a_component_twice_is_refused():
    with pytest.raises(ValueError, match="one cell per connected component"):
        assemble(disconnected_pairs(), dt=DT, variant=Variant.KNM, ground=[0, 1])


# --- The seam PR 4 builds on --------------------------------------------------------------


@pytest.mark.parametrize("variant", VARIANTS)
def test_the_right_hand_side_has_one_entry_per_degree_of_freedom(varied_sheet, variant):
    operator = assemble(varied_sheet, dt=DT, variant=variant)
    rhs = operator.rhs(ramp(varied_sheet.n_cells))

    assert rhs.shape == (operator.n_dofs,)
    np.testing.assert_allclose(rhs[: varied_sheet.n_cells], operator.D * ramp(varied_sheet.n_cells))
    np.testing.assert_array_equal(rhs[varied_sheet.n_cells :], 0.0)


def test_the_right_hand_side_refuses_the_wrong_number_of_cells(uniform_chain):
    operator = assemble(uniform_chain, dt=DT, variant=Variant.SKNM)
    with pytest.raises(ValueError, match="one value per cell"):
        operator.rhs(np.zeros(3))


def test_the_extracellular_potential_is_zero_everywhere_for_sknm_ue0(uniform_chain):
    operator = assemble(uniform_chain, dt=DT, variant=Variant.SKNM_UE0)
    solution = spla.spsolve(sp.csc_array(operator.matrix), operator.rhs(ramp(5)))

    np.testing.assert_array_equal(operator.extracellular_potential(solution), np.zeros(5))


def test_the_extracellular_potential_is_unavailable_for_sknm(uniform_chain):
    operator = assemble(uniform_chain, dt=DT, variant=Variant.SKNM)
    solution = spla.spsolve(sp.csc_array(operator.matrix), operator.rhs(ramp(5)))

    assert operator.extracellular_potential(solution) is None


def test_the_matrix_is_a_sparse_csr_array(uniform_chain):
    assert isinstance(assemble(uniform_chain, dt=DT, variant=Variant.KNM).matrix, sp.csr_array)


def test_the_operator_is_frozen(uniform_chain):
    operator = assemble(uniform_chain, dt=DT, variant=Variant.SKNM)
    with pytest.raises(dataclasses.FrozenInstanceError):
        operator.variant = Variant.KNM


def test_the_operator_reports_its_variant_and_size(uniform_chain):
    operator = assemble(uniform_chain, dt=DT, variant=Variant.KNM)

    assert operator.variant is Variant.KNM
    assert operator.n_cells == 5
    assert "KNM" in repr(operator)


# --- Units and variant names --------------------------------------------------------------


def test_dt_must_carry_a_unit(uniform_chain):
    with pytest.raises(TypeError, match="dt must be a quantity with units of time"):
        assemble(uniform_chain, dt=0.02, variant=Variant.SKNM)


def test_dt_must_be_a_time(uniform_chain):
    with pytest.raises(pint.DimensionalityError):
        assemble(uniform_chain, dt=0.02 * um, variant=Variant.SKNM)


@pytest.mark.parametrize("bad", [0.0 * ms, -0.02 * ms])
def test_dt_must_be_positive(uniform_chain, bad):
    with pytest.raises(ValueError, match="dt must be positive"):
        assemble(uniform_chain, dt=bad, variant=Variant.SKNM)


def test_dt_is_converted_rather_than_assumed(uniform_chain):
    from sknm.units import s

    in_milliseconds = assemble(uniform_chain, dt=20.0 * ms, variant=Variant.SKNM)
    in_seconds = assemble(uniform_chain, dt=0.02 * s, variant=Variant.SKNM)

    np.testing.assert_allclose(in_seconds.D, in_milliseconds.D, rtol=1e-14)


@pytest.mark.parametrize("variant", VARIANTS)
def test_a_variant_can_be_named_by_its_value(uniform_chain, variant):
    named = assemble(uniform_chain, dt=DT, variant=variant.value)

    assert named.variant is variant


def test_an_unknown_variant_is_refused(uniform_chain):
    with pytest.raises(ValueError, match="variant must be one of"):
        assemble(uniform_chain, dt=DT, variant="bidomain")


def test_the_capacitance_comes_from_the_network(uniform_chain):
    doubled = assemble(
        chain(5, membrane_area=MEMBRANE_AREA, Cm=2.0 * uF / cm**2, **paper_geometry(16, 16)),
        dt=DT,
        variant=Variant.SKNM_UE0,
    )
    single = assemble(uniform_chain, dt=DT, variant=Variant.SKNM_UE0)

    np.testing.assert_allclose(doubled.D, 2 * single.D, rtol=1e-14)


def test_the_operator_type_is_exported():
    assert isinstance(
        assemble(two_cells(), dt=DT, variant=Variant.SKNM),
        Operator,
    )
