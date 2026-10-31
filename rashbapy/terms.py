"""Linear-in-k spin-orbit terms of a two-band (Kramers doublet) model.

A general linear spin-orbit Hamiltonian in a plane of the Brillouin zone is

    H_SO(q) = sum_ij sigma_i C_ij q_j + sum_i b_i sigma_i ,   i = x,y,z ; j = x,y

The in-plane 2x2 block of C is spanned exactly by the four terms used by
Mohanta & Jena [PRB 111, 045140 (2025)]:

    rashba          (sigma_x k_y - sigma_y k_x)   antisymmetric part   -> alpha
    dresselhaus     (sigma_x k_x - sigma_y k_y)   traceless diagonal   -> beta
    modified_rashba (sigma_x k_y + sigma_y k_x)   symmetric off-diag.  -> M
    weyl            (sigma_x k_x + sigma_y k_y)   trace                -> J

plus the out-of-plane spin terms sigma_z k_x, sigma_z k_y (H5, H6 of
Mohanta & Jena) and the k-independent terms b . sigma, which are only
symmetry-allowed away from time-reversal-invariant momenta.

Raw vector layout (9 components):
    [C_xx, C_xy, C_yx, C_yy, C_zx, C_zy, b_x, b_y, b_z]
"""
import numpy as np

#: hbar^2 / (2 m_e) in eV Angstrom^2
HBAR2_2ME = 3.80998212

TERM_NAMES = (
    "rashba", "dresselhaus", "modified_rashba", "weyl",
    "sz_kx", "sz_ky", "bx", "by", "bz",
)

TERM_STRINGS = {
    "rashba": "(σx ky − σy kx)",
    "dresselhaus": "(σx kx − σy ky)",
    "modified_rashba": "(σx ky + σy kx)",
    "weyl": "(σx kx + σy ky)",
    "sz_kx": "σz kx",
    "sz_ky": "σz ky",
    "bx": "σx",
    "by": "σy",
    "bz": "σz",
}

#: default parameter symbols (Mohanta & Jena notation where it exists)
TERM_SYMBOLS = {
    "rashba": "alpha",
    "dresselhaus": "beta",
    "modified_rashba": "M",
    "weyl": "J",
    "sz_kx": "gamma_x",
    "sz_ky": "gamma_y",
    "bx": "b_x",
    "by": "b_y",
    "bz": "b_z",
}

# columns: raw vector of each named term
TERM_MATRIX = np.zeros((9, 9))
TERM_MATRIX[[1, 2], 0] = [1, -1]   # rashba: C_xy = 1, C_yx = -1
TERM_MATRIX[[0, 3], 1] = [1, -1]   # dresselhaus
TERM_MATRIX[[1, 2], 2] = [1, 1]    # modified rashba
TERM_MATRIX[[0, 3], 3] = [1, 1]    # weyl
TERM_MATRIX[4, 4] = 1              # sigma_z k_x
TERM_MATRIX[5, 5] = 1              # sigma_z k_y
TERM_MATRIX[6, 6] = TERM_MATRIX[7, 7] = TERM_MATRIX[8, 8] = 1
TERM_MATRIX_INV = np.linalg.inv(TERM_MATRIX)

# Pauli matrices
SIGMA = np.array([
    [[0, 1], [1, 0]],
    [[0, -1j], [1j, 0]],
    [[1, 0], [0, -1]],
], dtype=complex)


def named_to_raw(named):
    """Named-term coefficients (..., 9) -> raw vector (..., 9)."""
    return np.asarray(named) @ TERM_MATRIX.T


def raw_to_named(raw):
    """Raw vector (..., 9) -> named-term coefficients (..., 9)."""
    return np.asarray(raw) @ TERM_MATRIX_INV.T


def split_raw(raw):
    """Raw vector -> (C (3x2), b (3,))."""
    raw = np.asarray(raw, float)
    return raw[..., :6].reshape(raw.shape[:-1] + (3, 2)), raw[..., 6:]


def term_vector(*names, **weights):
    """Named-coefficient vector, e.g. term_vector('rashba') or
    term_vector(rashba=1, modified_rashba=-1)."""
    v = np.zeros(9)
    for n in names:
        v[TERM_NAMES.index(n)] = 1.0
    for n, w in weights.items():
        v[TERM_NAMES.index(n)] = w
    return v
