"""Symmetry analysis of crystals (spglib) and molecules (pymatgen, optional),
and derivation of the symmetry-allowed linear spin-orbit terms.

Assumptions (the same as in Mohanta & Jena and Maurer et al.):
  * the two bands form a Kramers-like doublet whose spin transforms as an
    axial vector (sigma -> det(R) R sigma) and k as a polar vector;
  * non-magnetic system (time reversal present) unless time_reversal=False.

For a unitary operation R of the little co-group (R k0 = k0 mod G) and an
anti-unitary operation T*R (R k0 = -k0 mod G) the linear coefficient tensor
must satisfy C = det(R) R C R^T; the k-independent field obeys b = det(R) R b
(unitary) or b = -det(R) R b (anti-unitary). The allowed tensors form the
null space of these linear constraints.
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .terms import TERM_NAMES, TERM_SYMBOLS, raw_to_named

TWO_PI = 2.0 * np.pi
_INPLANE_IDX = [0, 1, 3, 4, 6, 7, 9, 10, 11]  # C[:, :2] (row-major) + b


def _get(ds, key):
    return getattr(ds, key) if hasattr(ds, key) else ds[key]


class PlaneFrame:
    """Orthonormal frame (e1, e2, n) for a plane in reciprocal space.

    Model coordinates are (k1, k2) along e1, e2; spins are expressed in the
    same frame (s1, s2, s_n)."""

    def __init__(self, normal=(0, 0, 1), x_axis=None):
        n = np.asarray(normal, float)
        n = n / np.linalg.norm(n)
        if x_axis is None:
            x_axis = np.eye(3)[np.argmin(np.abs(n))]
        e1 = np.asarray(x_axis, float)
        e1 = e1 - (e1 @ n) * n
        if np.linalg.norm(e1) < 1e-8:
            raise ValueError("x_axis must not be parallel to the normal")
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        self.matrix = np.array([e1, e2, n])

    @property
    def normal(self):
        return self.matrix[2]

    def to_plane(self, v):
        """Cartesian (...,3) -> frame coordinates (...,3)."""
        return np.asarray(v, float) @ self.matrix.T

    def to_cartesian(self, v):
        v = np.asarray(v, float)
        if v.shape[-1] == 2:
            v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], axis=-1)
        return v @ self.matrix

    def transform_operation(self, R):
        return self.matrix @ R @ self.matrix.T

    def rotated(self, angle_deg):
        """Frame rotated by angle about the normal (e.g. 45 deg turns the
        Rashba+modified-Rashba model into the Rashba+Dresselhaus model)."""
        a = np.radians(angle_deg)
        e1, e2, _ = self.matrix
        return PlaneFrame(self.normal, np.cos(a) * e1 + np.sin(a) * e2)

    def __repr__(self):
        e1, e2, n = np.round(self.matrix, 4)
        return f"PlaneFrame(e1={e1.tolist()}, e2={e2.tolist()}, n={n.tolist()})"


@dataclass
class SymmetryInfo:
    """Result of :func:`analyze_symmetry`."""
    group_symbol: str
    group_number: Optional[int]
    point_group: str
    kpoint: np.ndarray                      # reciprocal fractional coords
    k_cartesian: np.ndarray                 # 1/Angstrom (incl. 2 pi)
    unitary: List[np.ndarray] = field(repr=False)       # Cartesian R, R k = k
    antiunitary: List[np.ndarray] = field(repr=False)   # Cartesian R, R k = -k (times T)
    little_cogroup: str = ""
    is_trim: bool = False
    time_reversal: bool = True
    cell: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self):
        lines = [
            f"Group              : {self.group_symbol}"
            + (f" (No. {self.group_number})" if self.group_number else ""),
            f"Point group        : {self.point_group}",
            f"k-point (frac.)    : {np.round(self.kpoint, 5).tolist()}",
            f"TRIM               : {self.is_trim}",
            f"Little co-group    : {self.little_cogroup} "
            f"({len(self.unitary)} unitary"
            + (f", {len(self.antiunitary)} anti-unitary T·R" if self.time_reversal else "")
            + " operations)",
        ]
        return "\n".join(lines)


def _unique(mats, tol=1e-6):
    out = []
    for m in mats:
        if not any(np.allclose(m, o, atol=tol) for o in out):
            out.append(m)
    return out


def _is_lattice_vector(dk_cart, A, tol=1e-5):
    frac = dk_cart @ A.T / TWO_PI
    return np.all(np.abs(frac - np.round(frac)) < tol)


def analyze_symmetry(atoms, kpoint=(0.0, 0.0, 0.0), symprec=1e-3,
                     angle_tolerance=-1.0, time_reversal=True):
    """Analyse the symmetry of an ``ase.Atoms`` object at a k-point.

    Parameters
    ----------
    atoms : ase.Atoms
        Periodic crystal (any pbc True, also 2D slabs with vacuum) or a
        molecule (all pbc False; requires pymatgen).
    kpoint : 3 floats
        Expansion point in fractional reciprocal coordinates.
    symprec, angle_tolerance : passed to spglib.
    time_reversal : bool
        Include time-reversal symmetry (non-magnetic system).
    """
    kpoint = np.asarray(kpoint, float)
    if not np.any(atoms.pbc):
        return _analyze_molecule(atoms, symprec, time_reversal)

    import spglib

    A = np.array(atoms.cell[:], float)
    cell = (A, atoms.get_scaled_positions(wrap=True), atoms.numbers)
    ds = spglib.get_symmetry_dataset(cell, symprec=symprec,
                                     angle_tolerance=angle_tolerance)
    if ds is None:
        raise RuntimeError("spglib could not determine the symmetry")
    rots = _unique([np.array(r) for r in _get(ds, "rotations")])
    AT, ATinv = A.T, np.linalg.inv(A.T)
    k_cart = TWO_PI * kpoint @ np.linalg.inv(A).T

    unitary, anti, unitary_frac = [], [], []
    for R in rots:
        Rc = AT @ R @ ATinv
        if _is_lattice_vector(Rc @ k_cart - k_cart, A):
            unitary.append(Rc)
            unitary_frac.append(R)
        if time_reversal and _is_lattice_vector(Rc @ k_cart + k_cart, A):
            anti.append(Rc)
    pg_little = spglib.get_pointgroup(np.array(unitary_frac, dtype="intc"))[0].strip()
    is_trim = _is_lattice_vector(2 * k_cart, A)
    return SymmetryInfo(
        group_symbol=_get(ds, "international"),
        group_number=int(_get(ds, "number")),
        point_group=_get(ds, "pointgroup"),
        kpoint=kpoint, k_cartesian=k_cart,
        unitary=unitary, antiunitary=anti,
        little_cogroup=pg_little, is_trim=is_trim,
        time_reversal=time_reversal, cell=A,
    )


def _analyze_molecule(atoms, symprec, time_reversal):
    try:
        from pymatgen.core import Molecule
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Molecules (pbc all False) need pymatgen: "
                          "pip install pymatgen") from exc
    import warnings
    warnings.filterwarnings("ignore", category=np.exceptions.ComplexWarning)
    mol = Molecule(atoms.get_chemical_symbols(), atoms.positions)
    pga = PointGroupAnalyzer(mol, tolerance=max(symprec, 0.1))
    ops = _unique([op.rotation_matrix for op in pga.get_symmetry_operations()])
    return SymmetryInfo(
        group_symbol=pga.sch_symbol, group_number=None,
        point_group=pga.sch_symbol, kpoint=np.zeros(3), k_cartesian=np.zeros(3),
        unitary=ops, antiunitary=list(ops) if time_reversal else [],
        little_cogroup=pga.sch_symbol, is_trim=True, time_reversal=time_reversal,
    )


def _rref(M, tol=1e-8):
    A = np.array(M, float)
    nr, nc = A.shape
    r, pivots = 0, []
    for c in range(nc):
        if r == nr:
            break
        p = r + np.argmax(np.abs(A[r:, c]))
        if abs(A[p, c]) < tol:
            continue
        A[[r, p]] = A[[p, r]]
        A[r] /= A[r, c]
        for i in range(nr):
            if i != r:
                A[i] -= A[i, c] * A[r]
        pivots.append(c)
        r += 1
    A[np.abs(A) < tol] = 0.0
    return A[:r], pivots


def allowed_terms(unitary, antiunitary, frame=None, include_constant=True,
                  tol=1e-4):
    """Symmetry-allowed linear spin-orbit terms in the plane ``frame``.

    Returns
    -------
    basis : (p, 9) array
        Each row gives the named-term coefficients (order of
        ``terms.TERM_NAMES``) multiplying one free parameter. Rows are in
        reduced row-echelon form, so each parameter is anchored to one
        named term ("pivot") with as few ties as possible.
    names : list of str
        Parameter symbols (alpha, beta, M, J, gamma_x, ...).
    """
    frame = frame or PlaneFrame()
    blocks = []
    for ops, anti in ((unitary, False), (antiunitary, True)):
        for R in ops:
            Rp = frame.transform_operation(np.asarray(R, float))
            d = np.linalg.det(Rp)
            blk = np.zeros((12, 12))
            blk[:9, :9] = np.eye(9) - d * np.kron(Rp, Rp)
            blk[9:, 9:] = np.eye(3) - (-d if anti else d) * Rp
            blocks.append(blk)
    if not include_constant:
        blk = np.zeros((3, 12))
        blk[:, 9:] = np.eye(3)
        blocks.append(blk)
    M = np.vstack(blocks) if blocks else np.zeros((1, 12))
    _, s, vt = np.linalg.svd(M)
    rank = int(np.sum(s > tol * max(s.max(), 1.0)))
    null = vt[rank:]
    if null.shape[0] == 0:
        return np.zeros((0, 9)), []
    proj = null[:, _INPLANE_IDX]
    _, s2, vt2 = np.linalg.svd(proj)
    rowspace = vt2[: int(np.sum(s2 > tol))]
    if rowspace.shape[0] == 0:
        return np.zeros((0, 9)), []
    named = raw_to_named(rowspace)
    basis, pivots = _rref(named, tol)
    basis = np.round(basis, 10)
    names = [TERM_SYMBOLS[TERM_NAMES[p]] for p in pivots]
    return basis, names
