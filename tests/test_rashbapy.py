import warnings

import numpy as np
import pytest
from ase.build import bulk, molecule
from ase.spacegroup import crystal

from rashbapy import PlaneFrame, SpinSplittingModel as Model, rashba_parameter, plane_kgrid

G1, G2, G3 = [0.123, 0.217, 0.311], [0.371, 0.083, 0.647], [0.29, 0.41, 0.05]


def pna21():
    return crystal(["Bi", "In", "O"], [G1, G2, G3], spacegroup=33,
                   cellpar=[5.9, 8.4, 5.0, 90, 90, 90])


def active(model):
    return {t for t, v in zip(
        ["rashba", "dresselhaus", "modified_rashba", "weyl", "sz_kx", "sz_ky", "bx", "by", "bz"],
        np.any(np.abs(model.basis) > 1e-8, axis=0)) if v}


def grid(kmax=0.05, n=11):
    g = np.linspace(-kmax, kmax, n)
    a, b = np.meshgrid(g, g)
    return np.c_[a.ravel(), b.ravel()]


# ---------------------------------------------------------------- symmetry
def test_wurtzite_gamma_rashba_only():
    m = Model.from_symmetry(bulk("ZnO", "wurtzite", a=3.25, c=5.2))
    assert m.symmetry.little_cogroup == "6mm"
    assert active(m) == {"rashba"}


def test_zincblende_gamma_no_linear_terms():
    assert Model.from_symmetry(bulk("GaAs", "zincblende", a=5.65)).n_params == 0


def test_d2d_dresselhaus():
    a = crystal(["Si", "C"], [G1, G2], spacegroup=121, cellpar=[5, 5, 7, 90, 90, 90])
    assert active(Model.from_symmetry(a)) == {"dresselhaus"}


def test_chiral_weyl():
    a = crystal(["Si", "C"], [G1, G2], spacegroup=198, cellpar=[5, 5, 5, 90, 90, 90])
    assert active(Model.from_symmetry(a)) == {"weyl"}


def test_pna21_gives_mj1_and_rotated_rd():
    a = pna21()
    for kp in [(0, 0, 0), (0.5, 0, 0)]:
        m = Model.from_symmetry(a, kp)
        assert active(m) == {"rashba", "modified_rashba"}
        assert m.symmetry.is_trim
    m45 = Model.from_symmetry(a, (0.5, 0, 0), frame=PlaneFrame().rotated(45))
    assert active(m45) == {"rashba", "dresselhaus"}


def test_non_trim_allows_constant_term():
    m = Model.from_symmetry(pna21(), (0.2, 0, 0))
    assert "by" in active(m) and not m.symmetry.is_trim
    m2 = Model.from_symmetry(pna21(), (0.2, 0, 0), include_constant=False)
    assert not active(m2) & {"bx", "by", "bz"}


def test_molecule_c3v():
    pytest.importorskip("pymatgen")
    m = Model.from_symmetry(molecule("NH3"))
    assert m.symmetry.group_symbol == "C3v"
    assert active(m) == {"rashba"}


# ------------------------------------------------------------------ model
def test_eigen_consistency():
    m = Model.mj3(alpha=0.7, J=0.2, beta=-0.3, mass=0.4, e0=1.0)
    k = grid()
    w, v = np.linalg.eigh(m.hamiltonian(k))
    assert np.allclose(w, m.energies(k))
    sig = np.einsum("nai,xab,nbi->nix", v.conj(), np.array(
        [[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]]), v).real
    ok = np.linalg.norm(m.h_vector(k), axis=1) > 1e-10
    assert np.allclose(sig[ok], m.spins(k)[ok], atol=1e-10)


def test_pst_limits_of_mj1():
    assert np.allclose(np.abs(Model.mj1(1, 1).pst_direction()), [1, 0, 0])
    assert np.allclose(np.abs(Model.mj1(1, -1).pst_direction()), [0, 1, 0])
    assert Model.mj1(1, -1).is_persistent()
    assert not Model.mj1(1, -0.99).is_persistent()
    assert "PST" in Model.rashba_dresselhaus(1, 1).classify()


def test_rashba_dresselhaus_energy_degeneracy():
    k = grid()
    assert np.allclose(Model.rashba_dresselhaus(1, .5).energies(k),
                       Model.rashba_dresselhaus(.5, 1).energies(k))


def test_rashba_parameter_mohanta_value():
    kR, ER = 0.0457, 0.0181
    k = np.linspace(-0.15, 0.15, 3001)
    c = ER / kR ** 2  # E = c (|k|-kR)^2 - ER  -> minimum at kR, crossing at 0
    E = c * (np.abs(k) - kR) ** 2 - c * kR ** 2
    a, er, kr = rashba_parameter(k, E)
    assert np.isclose(a, 0.792, atol=2e-3)


def test_plane_kgrid_roundtrip():
    a = pna21()
    m = Model.from_symmetry(a, (0.5, 0, 0), normal=(0, 0, 1))
    kf, kp = plane_kgrid(a, (0.5, 0, 0), m.frame, 0.05, 5)
    kc = 2 * np.pi * kf @ np.linalg.inv(a.cell[:]).T
    assert np.allclose(m.data_to_plane(kc), kp)


# -------------------------------------------------------------------- fit
def _data(true, k, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    E = true.energies(k) + noise * rng.standard_normal((len(k), 2))
    S = true.spins(k) + 0.02 * noise / 1e-3 * rng.standard_normal((len(k), 2, 3))
    return E, S


def test_fit_partial_pst_bi_in_o3_like():
    """Mohanta & Jena: alpha = 1, M = -0.99 (partial PST) around X."""
    m = Model.from_symmetry(pna21(), (0.5, 0, 0))
    true = m.copy(params={"alpha": 0.8, "M": -0.792}, mass=0.35, e0=2.6)
    k = grid()
    E, S = _data(true, k, noise=1e-3)
    res = m.fit(k, E, S)
    assert res.success
    assert np.isclose(res.params["alpha"], 0.8, rtol=0.03)
    assert np.isclose(res.params["M"], -0.792, rtol=0.03)
    assert np.isclose(res.params["mass"], 0.35, rtol=0.05)
    assert "partial (bidirectional) PST" in res.model.classify()
    assert "alpha" in res.report()


def test_fit_spiral_mj2_valence_band():
    true = Model.mj2(alpha=0.69, J=0.21, mass=-0.5, e0=0.0)
    k = grid(0.08)
    E, S = _data(true, k)
    res = Model.general().fit(k, E, S)
    c = res.model.named_coefficients()
    assert np.isclose(c["rashba"], 0.69, rtol=1e-4)
    assert np.isclose(c["weyl"], 0.21, rtol=1e-4)
    assert abs(c["dresselhaus"]) < 1e-5 and abs(c["sz_kx"]) < 1e-5
    assert res.params["mass"] < 0
    assert "spiral" in res.model.classify()


def test_fit_single_band_spins_and_kmax():
    true = Model.rashba_dresselhaus(0.5, 0.2, mass=0.3, e0=1.0)
    k = grid(0.1, 21)
    E, S = _data(true, k)
    res = Model.rashba_dresselhaus().fit(k, E, S[:, 1], spin_band="upper", k_max=0.06)
    assert np.allclose([res.params["alpha"], res.params["beta"]], [0.5, 0.2], rtol=1e-5)
    assert res.n_points < len(k)


def test_fit_energy_only_warns():
    true = Model.rashba_dresselhaus(0.5, 0.2)
    k = grid()
    with pytest.warns(UserWarning, match="energies only"):
        Model.rashba_dresselhaus().fit(k, true.energies(k))


def test_fit_from_cartesian_rotated_frame():
    frame = PlaneFrame(normal=(1, 1, 0))
    true = Model.mj1(0.6, 0.3, mass=0.25, e0=0.5, frame=frame)
    kp = grid()
    kc, sc = frame.to_cartesian(kp), np.einsum("nbi,ij->nbj", true.spins(kp), frame.matrix)
    model = Model.mj1(frame=frame)
    k2, s2 = model.data_to_plane(kc, sc, kpoint_cartesian=np.zeros(3))
    res = model.fit(k2, true.energies(kp), s2)
    assert np.allclose([res.params["alpha"], res.params["M"]], [0.6, 0.3], rtol=1e-5)


def test_maurer_parameters_mj1_equals_rd():
    mp = Model.mj1(1.0, -0.99).maurer_parameters()
    assert np.isclose(abs(mp["angle"]), 45)
    assert np.isclose(mp["lambda_R"], -1.0) and np.isclose(abs(mp["lambda_D"]), 0.99)
    rd = Model.rashba_dresselhaus(0.3, 0.2).maurer_parameters()
    assert np.allclose([rd["lambda_R"], rd["lambda_D"], rd["angle"]], [-0.3, 0.2, 0])
