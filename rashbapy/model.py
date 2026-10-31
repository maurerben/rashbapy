"""Two-band linear spin-orbit model  H(q) = e0 + hbar^2 q^2 / 2m* + h(q)·σ ,
h(q) = C q + b,  q = k - k0  (in-plane coordinates of a PlaneFrame)."""
import copy

import numpy as np

from .symmetry import PlaneFrame, allowed_terms, analyze_symmetry
from .terms import (HBAR2_2ME, SIGMA, TERM_NAMES, TERM_STRINGS, TERM_SYMBOLS,
                    named_to_raw, split_raw, term_vector)


class SpinSplittingModel:
    """Linear-in-k spin-splitting model of a Kramers doublet.

    Parameters
    ----------
    terms : list of str, optional
        Named terms (see ``rashbapy.terms.TERM_NAMES``), one free parameter
        each. Alternative to ``basis``.
    basis : (p, 9) array, optional
        Named-term coefficients of each free parameter (allows tied terms).
    names : list of str, optional
        Parameter names; defaults to Mohanta-Jena symbols.
    params : dict or sequence, optional
        Parameter values in eV·Å (eV for constant terms). Default 0.
    mass : float or None
        Effective mass in units of m_e (negative for hole-like bands).
        ``None`` removes the kinetic term.
    e0 : float
        Energy at k0 (eV).
    k0 : 2 floats
        Band centre in plane coordinates (1/Å), relative to the k-point.
    frame : PlaneFrame
    symmetry : SymmetryInfo, optional
    label : str, optional
    """

    def __init__(self, terms=None, basis=None, names=None, params=None,
                 mass=1.0, e0=0.0, k0=(0.0, 0.0), frame=None, symmetry=None,
                 label=None):
        if basis is None:
            terms = list(terms or [])
            basis = np.array([term_vector(t) for t in terms]).reshape(-1, 9)
            names = names or [TERM_SYMBOLS[t] for t in terms]
        basis = np.asarray(basis, float).reshape(-1, 9)
        if names is None:
            names = [TERM_SYMBOLS[TERM_NAMES[int(np.argmax(np.abs(r) > 1e-12))]]
                     for r in basis]
        names = _deduplicate(list(names))
        if len(names) != len(basis):
            raise ValueError("names and basis have different length")
        self.basis = basis
        self.names = names
        self.params = np.zeros(len(names))
        if params is not None:
            self.set_params(params)
        self.mass = mass
        self.e0 = float(e0)
        self.k0 = np.asarray(k0, float)
        self.frame = frame or PlaneFrame()
        self.symmetry = symmetry
        self.label = label

    # ------------------------------------------------------------------ setup
    def set_params(self, params):
        if isinstance(params, dict):
            for k, v in params.items():
                self.params[self.names.index(k)] = v
        else:
            params = np.asarray(params, float)
            if params.shape != self.params.shape:
                raise ValueError(f"expected {len(self.params)} parameters")
            self.params = params.copy()
        return self

    def copy(self, **changes):
        new = copy.deepcopy(self)
        for k, v in changes.items():
            if k == "params":
                new.set_params(v)
            else:
                setattr(new, k, v)
        return new

    @property
    def n_params(self):
        return len(self.names)

    @property
    def curvature(self):
        """hbar^2 / 2m* in eV Å^2 (0 if mass is None)."""
        return 0.0 if self.mass is None else HBAR2_2ME / self.mass

    # ------------------------------------------------------- coefficient views
    def named_coefficients(self):
        """Dict term -> coefficient (alpha of rashba, beta of dresselhaus, ...)."""
        v = self.params @ self.basis
        return {t: float(c) for t, c in zip(TERM_NAMES, v)}

    def coefficient_tensor(self):
        """(C (3x2), b (3,)) with h = C q + b in the plane frame."""
        return split_raw(named_to_raw(self.params @ self.basis))

    def parameter_dict(self):
        return dict(zip(self.names, self.params.tolist()))

    # --------------------------------------------------------------- physics
    def _q(self, k):
        k = np.asarray(k, float)
        if k.shape[-1] != 2:
            raise ValueError("k must have shape (..., 2) in plane coordinates")
        return k - self.k0

    def _term_fields(self, q):
        """h contribution of each parameter: (..., p, 3)."""
        C, b = split_raw(named_to_raw(self.basis))  # (p,3,2), (p,3)
        return np.einsum("pij,...j->...pi", C, q) + b

    def h_vector(self, k):
        """Effective spin-orbit field h(k), shape (..., 3)."""
        C, b = self.coefficient_tensor()
        return self._q(k) @ C.T + b

    def band_center(self, k):
        q = self._q(k)
        return self.e0 + self.curvature * np.sum(q ** 2, axis=-1)

    def hamiltonian(self, k):
        """2x2 Hamiltonian, shape (..., 2, 2)."""
        h = self.h_vector(k)
        eps = self.band_center(k)
        return eps[..., None, None] * np.eye(2) + np.einsum("...i,ijk->...jk", h, SIGMA)

    def energies(self, k):
        """Band energies (..., 2), [lower, upper]."""
        hn = np.linalg.norm(self.h_vector(k), axis=-1)
        eps = self.band_center(k)
        return np.stack([eps - hn, eps + hn], axis=-1)

    def spins(self, k):
        """Spin expectation values <σ> (..., 2, 3) for [lower, upper] band,
        in the plane frame (s1, s2, s_n). Zero where bands are degenerate."""
        h = self.h_vector(k)
        n = np.linalg.norm(h, axis=-1, keepdims=True)
        u = np.divide(h, n, out=np.zeros_like(h), where=n > 1e-14)
        return np.stack([-u, u], axis=-2)

    spin_texture = spins

    def splitting(self, k):
        return 2.0 * np.linalg.norm(self.h_vector(k), axis=-1)

    # --------------------------------------------------------- interpretation
    def invariants(self):
        """Rotation invariants of the in-plane block: |alpha| (Rashba),
        |J| (Weyl), sqrt(beta^2+M^2) (Dresselhaus-type, Dresselhaus and
        modified Rashba differ only by a π/4 rotation), out-of-plane
        strength sqrt(gx^2+gy^2) and |b|."""
        c = self.named_coefficients()
        return {
            "rashba": abs(c["rashba"]),
            "weyl": abs(c["weyl"]),
            "dresselhaus_type": float(np.hypot(c["dresselhaus"], c["modified_rashba"])),
            "out_of_plane": float(np.hypot(c["sz_kx"], c["sz_ky"])),
            "constant": float(np.linalg.norm([c["bx"], c["by"], c["bz"]])),
        }

    def is_persistent(self, rtol=1e-3):
        """True if h(q) points along one fixed axis (persistent spin texture)."""
        C, _ = self.coefficient_tensor()
        s = np.linalg.svd(C, compute_uv=False)
        return s[0] > 0 and s[1] <= rtol * s[0]

    def pst_direction(self):
        """Spin axis of the (approximate) persistent spin texture."""
        C, _ = self.coefficient_tensor()
        return np.linalg.svd(C)[0][:, 0]

    def classify(self, term_rtol=0.02, pst_rtol=1e-3, partial_pst=0.1):
        """Short text classification of the spin texture.

        PST if s_min/s_max of the linear tensor C <= pst_rtol; an elliptic
        Rashba-Dresselhaus texture with s_min/s_max <= partial_pst is called a
        partial (bidirectional) PST, cf. alpha = 1, M = -0.99 in Mohanta & Jena."""
        inv = self.invariants()
        C, _ = self.coefficient_tensor()
        s = np.linalg.svd(C, compute_uv=False)
        if s[0] < 1e-12:
            return "no linear spin splitting"
        aniso = s[1] / s[0]
        if aniso <= pst_rtol:
            d = np.round(self.pst_direction(), 3)
            return f"persistent spin texture (PST) along {d.tolist()}"
        big = {k for k in ("rashba", "weyl", "dresselhaus_type", "out_of_plane")
               if inv[k] > term_rtol * s[0]}
        if big == {"rashba"}:
            kind = "pure Rashba (circular)"
        elif big == {"weyl"}:
            kind = "pure Weyl (radial)"
        elif big == {"dresselhaus_type"}:
            kind = "pure Dresselhaus type (hyperbolic)"
        elif big == {"rashba", "dresselhaus_type"}:
            kind = ("Rashba-Dresselhaus, partial (bidirectional) PST" if aniso <= partial_pst
                    else "Rashba-Dresselhaus (elliptic)")
        elif big == {"rashba", "weyl"}:
            kind = "Rashba-Weyl (circular spiral, H_MJ2)"
        elif {"rashba", "weyl", "dresselhaus_type"} <= big:
            kind = "Rashba-Weyl-Dresselhaus (elliptical spiral, H_MJ3/H_MJ4)"
        else:
            kind = "mixed: " + ", ".join(sorted(big))
        return f"{kind}; anisotropy s_min/s_max = {aniso:.4f}"

    def in_rotated_frame(self, angle_deg):
        """Named in-plane coefficients after rotating the in-plane axes by
        ``angle_deg`` about the normal (spin and k rotate together).
        Rashba and Weyl are invariant; (Dresselhaus, modified Rashba) rotate
        by twice the angle, so 45° maps H_MJ1 onto H_RD."""
        C, _ = self.coefficient_tensor()
        a = np.radians(angle_deg)
        R = np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])
        A = R @ C[:2, :2] @ R.T
        return {"rashba": 0.5 * (A[0, 1] - A[1, 0]), "dresselhaus": 0.5 * (A[0, 0] - A[1, 1]),
                "modified_rashba": 0.5 * (A[0, 1] + A[1, 0]), "weyl": 0.5 * (A[0, 0] + A[1, 1])}

    def maurer_parameters(self):
        """Rashba/Dresselhaus parameters in the convention of Maurer et al.,
        H = λR(kxσy − kyσx) + λD(kxσx − kyσy), evaluated in the in-plane frame
        rotated by ``angle`` (deg) so that the modified-Rashba term vanishes.
        Returns dict(lambda_R, lambda_D, angle, weyl). λR = −alpha."""
        c = self.named_coefficients()
        phi = 0.5 * np.degrees(np.arctan2(c["modified_rashba"], c["dresselhaus"]))
        best = None
        for ang in (phi, -phi):
            r = self.in_rotated_frame(ang)
            if best is None or abs(r["modified_rashba"]) < abs(best[1]["modified_rashba"]):
                best = (ang, r)
        ang, r = best
        return {"lambda_R": -r["rashba"], "lambda_D": r["dresselhaus"],
                "angle": float(ang), "weyl": r["weyl"]}

    # ---------------------------------------------------------------- output
    def formula(self):
        parts = []
        if self.mass is not None:
            parts.append("ħ²q²/2m*")
        for name, row in zip(self.names, self.basis):
            sub = []
            for t, c in zip(TERM_NAMES, row):
                if abs(c) > 1e-10:
                    coef = "" if np.isclose(c, 1) else ("−" if np.isclose(c, -1) else f"{c:.4g}·")
                    sub.append(f"{coef}{TERM_STRINGS[t]}")
            parts.append(f"{name}·" + (sub[0] if len(sub) == 1 else "[" + " + ".join(sub) + "]"))
        return "H = " + (" + ".join(parts) if parts else "0")

    def __repr__(self):
        p = ", ".join(f"{n}={v:.4g}" for n, v in zip(self.names, self.params))
        m = "None" if self.mass is None else f"{self.mass:.4g}"
        lab = f"{self.label}: " if self.label else ""
        return f"SpinSplittingModel({lab}{p}; mass={m}, e0={self.e0:.4g})"

    def summary(self):
        lines = [repr(self), self.formula(), f"Frame: {self.frame}"]
        if self.symmetry is not None:
            lines.append(self.symmetry.summary())
        if np.any(self.params):
            lines.append("Classification: " + self.classify())
        return "\n".join(lines)

    # ------------------------------------------------------------ conversion
    def data_to_plane(self, k_cartesian, spins_cartesian=None, kpoint_cartesian=None,
                      max_out_of_plane=1e-4):
        """Convert DFT data (Cartesian k in 1/Å, Cartesian spins) to plane
        coordinates relative to the expansion k-point."""
        k = np.asarray(k_cartesian, float)
        if kpoint_cartesian is None and self.symmetry is not None:
            kpoint_cartesian = self.symmetry.k_cartesian
        if kpoint_cartesian is not None:
            k = k - kpoint_cartesian
        kp = self.frame.to_plane(k)
        if np.max(np.abs(kp[..., 2]), initial=0) > max_out_of_plane:
            raise ValueError("k-points do not lie in the model plane")
        out = kp[..., :2]
        if spins_cartesian is None:
            return out
        return out, self.frame.to_plane(spins_cartesian)

    # --------------------------------------------------------- constructors
    @classmethod
    def from_symmetry(cls, atoms, kpoint=(0, 0, 0), normal=(0, 0, 1), x_axis=None,
                      frame=None, include_constant=True, symprec=1e-3,
                      time_reversal=True, mass=1.0, label=None):
        """Set up the symmetry-allowed model for ``atoms`` (ase.Atoms) around
        ``kpoint`` (fractional reciprocal coordinates) in the plane with
        Cartesian ``normal``."""
        sym = analyze_symmetry(atoms, kpoint, symprec=symprec, time_reversal=time_reversal)
        frame = frame or PlaneFrame(normal, x_axis)
        basis, names = allowed_terms(sym.unitary, sym.antiunitary, frame,
                                     include_constant=include_constant)
        return cls(basis=basis, names=names, mass=mass, frame=frame, symmetry=sym,
                   label=label or f"{sym.group_symbol} @ {np.round(sym.kpoint, 4).tolist()}")

    @classmethod
    def rashba(cls, alpha=1.0, **kw):
        return cls(["rashba"], params=[alpha], label="H_R", **kw)

    @classmethod
    def dresselhaus(cls, beta=1.0, **kw):
        return cls(["dresselhaus"], params=[beta], label="H_D", **kw)

    @classmethod
    def modified_rashba(cls, M=1.0, **kw):
        return cls(["modified_rashba"], params=[M], label="H_MR", **kw)

    @classmethod
    def weyl(cls, J=1.0, **kw):
        return cls(["weyl"], params=[J], label="H_MD (Weyl)", **kw)

    @classmethod
    def rashba_dresselhaus(cls, alpha=1.0, beta=0.0, **kw):
        return cls(["rashba", "dresselhaus"], params=[alpha, beta], label="H_RD", **kw)

    @classmethod
    def mj1(cls, alpha=1.0, M=0.0, **kw):
        """H_MJ1 = Rashba + modified Rashba (Mohanta & Jena Eq. 17)."""
        return cls(["rashba", "modified_rashba"], params=[alpha, M], label="H_MJ1", **kw)

    @classmethod
    def mj2(cls, alpha=1.0, J=0.0, **kw):
        """H_MJ2 = Rashba + Weyl (Mohanta & Jena Eq. 23)."""
        return cls(["rashba", "weyl"], params=[alpha, J], label="H_MJ2", **kw)

    @classmethod
    def mj3(cls, alpha=1.0, J=0.0, beta=0.0, **kw):
        return cls(["rashba", "weyl", "dresselhaus"], params=[alpha, J, beta], label="H_MJ3", **kw)

    @classmethod
    def mj4(cls, alpha=1.0, J=0.0, M=0.0, **kw):
        return cls(["rashba", "weyl", "modified_rashba"], params=[alpha, J, M], label="H_MJ4", **kw)

    @classmethod
    def general(cls, include_constant=False, **kw):
        terms = list(TERM_NAMES if include_constant else TERM_NAMES[:6])
        return cls(terms, label="general", **kw)

    # ----------------------------------------------------------------- fit
    def fit(self, k, energies, spins=None, **kw):
        """Fit to data; see :func:`rashbapy.fit.fit_model`."""
        from .fit import fit_model
        return fit_model(self, k, energies, spins, **kw)


def _deduplicate(names):
    seen, out = {}, []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}_{seen[n]}")
        else:
            seen[n] = 1
            out.append(n)
    return out
