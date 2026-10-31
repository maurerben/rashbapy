"""Fit a SpinSplittingModel to energies and spin textures on a k-plane."""
import warnings
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .terms import HBAR2_2ME


@dataclass
class FitResult:
    model: "SpinSplittingModel"
    params: dict
    stderr: dict
    rms_energy: float           # eV
    mean_spin_angle: float      # degrees, NaN without spin data
    success: bool
    message: str
    n_points: int
    extra: dict = field(default_factory=dict, repr=False)

    def report(self):
        m = self.model
        lines = [f"Fit {'converged' if self.success else 'FAILED'}: {self.message}",
                 f"points used           : {self.n_points}",
                 f"rms energy error      : {1e3 * self.rms_energy:.3f} meV"]
        if not np.isnan(self.mean_spin_angle):
            lines.append(f"mean spin deviation   : {self.mean_spin_angle:.2f} deg")
        lines.append("parameters (eV·Å, constants in eV):")
        for n, v in self.params.items():
            e = self.stderr.get(n, np.nan)
            lines.append(f"  {n:>10s} = {v: .5f} ± {e:.5f}")
        lines.append("named coefficients:")
        for t, v in m.named_coefficients().items():
            if abs(v) > 1e-12:
                lines.append(f"  {t:>16s} = {v: .5f}")
        mp = m.maurer_parameters()
        lines.append(f"Maurer et al. λR, λD  : {mp['lambda_R']:.5f}, {mp['lambda_D']:.5f} eV·Å "
                     f"(in-plane axes rotated by {mp['angle']:.2f}°)")
        lines.append("classification        : " + m.classify())
        return "\n".join(lines)


def _prepare_spins(spins, spin_band, n, threshold):
    """-> list of (sign, unit spins (N,3), mask)."""
    if spins is None:
        return []
    s = np.asarray(spins, float)
    if s.ndim == 2:
        if spin_band is None:
            raise ValueError("spins of shape (N,3) need spin_band='lower'/'upper'")
        band = {"lower": 0, "upper": 1, 0: 0, 1: 1}[spin_band]
        sets = [(band, s)]
    elif s.ndim == 3 and s.shape[1] == 2:
        sets = [(0, s[:, 0]), (1, s[:, 1])]
    else:
        raise ValueError("spins must have shape (N,2,3) or (N,3)")
    out = []
    for band, sb in sets:
        if sb.shape != (n, 3):
            raise ValueError("spin array does not match k")
        norm = np.linalg.norm(sb, axis=1)
        mask = norm > threshold * max(norm.max(), 1e-300)
        unit = np.zeros_like(sb)
        unit[mask] = sb[mask] / norm[mask, None]
        out.append((-1.0 if band == 0 else 1.0, unit, mask))
    return out


def fit_model(model, k, energies, spins=None, spin_band=None, fit_e0=True,
              fit_mass=True, fit_k0=False, spin_weight=None, spin_threshold=0.05,
              k_max=None, energy_weight=1.0, energy_band=None):
    """Fit spin-orbit parameters (and e0, m*, optionally k0) of ``model``.

    Parameters
    ----------
    model : SpinSplittingModel
        Defines the allowed terms (e.g. from ``SpinSplittingModel.from_symmetry``).
        Its current parameters are ignored as start values.
    k : (N, 2) array
        k-points in plane coordinates relative to the expansion point, 1/Å
        (use ``model.data_to_plane`` for Cartesian DFT data).
    energies : (N, 2) or (N,) array
        Energies (eV) of the two spin-split bands (sorted internally). A 1D
        array is a single band (``energy_band``); this needs spins to be
        identifiable and the partner band is then not constrained.
    spins : (N, 2, 3) or (N, 3) array, optional
        Spin expectation values in the plane frame (normalisation irrelevant).
        Strongly recommended: energies alone cannot distinguish e.g. Rashba
        from Dresselhaus (Maurer et al.).
    spin_band : 'lower' | 'upper'
        Band of a (N,3) spin array.
    spin_weight : float, optional
        Weight of unit-spin residuals relative to energies in eV. Default: rms
        of half the splitting, so both data types count similarly.
    spin_threshold : float
        Ignore spins with |s| below this fraction of the maximum.
    energy_band : 'lower' | 'upper'
        Which branch the 1D ``energies`` belong to (also the default
        ``spin_band``).
    k_max : float, optional
        Use only points with |k| <= k_max (linear models hold near k0).
    """
    k = np.asarray(k, float)
    if k.ndim != 2 or k.shape[1] != 2:
        raise ValueError("k must have shape (N, 2)")
    energies = np.asarray(energies, float)
    if energies.ndim == 1:
        if energy_band not in ("lower", "upper"):
            raise ValueError("1D energies need energy_band='lower'/'upper'")
        cols = [0 if energy_band == "lower" else 1]
        E = energies[:, None]
        spin_band = spin_band or energy_band
    else:
        cols = [0, 1]
        E = np.sort(energies, axis=1)
    if E.shape != (len(k), len(cols)):
        raise ValueError("energies must have shape (N, 2) or (N,)")
    two = len(cols) == 2
    sel = np.ones(len(k), bool)
    if k_max is not None:
        sel = np.linalg.norm(k - model.k0, axis=1) <= k_max
    spin_sets = _prepare_spins(spins, spin_band, len(k), spin_threshold)
    k, E = k[sel], E[sel]
    spin_sets = [(sg, u[sel], m[sel]) for sg, u, m in spin_sets]
    p = model.n_params
    if p == 0:
        warnings.warn("model has no spin-orbit parameters")
    if not spin_sets and p > 1:
        warnings.warn("fitting energies only: different spin textures (e.g. Rashba vs "
                      "Dresselhaus) give identical bands; provide spins to resolve them")

    work = model.copy()
    q = k - work.k0
    q2 = np.sum(q ** 2, axis=1)
    half = np.zeros(len(k))  # half the splitting (estimate for one band)
    if two:
        half = 0.5 * (E[:, 1] - E[:, 0])
        eps = 0.5 * (E[:, 0] + E[:, 1])

    # --- initial guess: kinetic part (linear least squares)
    if not two:
        e0, c = work.e0, work.curvature
    elif fit_mass and fit_e0:
        (e0, c), *_ = np.linalg.lstsq(np.c_[np.ones_like(q2), q2], eps, rcond=None)
    elif fit_mass:
        e0 = work.e0
        c = np.dot(q2, eps - e0) / max(np.dot(q2, q2), 1e-300)
    else:
        c = work.curvature
        e0 = np.mean(eps - c * q2) if fit_e0 else work.e0

    # --- initial guess: spin-orbit part (spins give direction, splitting the scale)
    G = work._term_fields(q)  # (N,p,3)
    if p:
        if spin_sets:
            rows = []
            for sg, u, m in spin_sets:
                rows.append(np.cross(u[m, None, :], G[m]).transpose(0, 2, 1).reshape(-1, p))
            A = np.vstack(rows)
            theta = np.linalg.svd(A)[2][-1]
            dots = sum(np.sum(sg * np.einsum("ni,npi,p->n", u[m], G[m], theta))
                       for sg, u, m in spin_sets)
            if dots < 0:
                theta = -theta
        else:
            theta = np.ones(p) / np.sqrt(p)
        hh = np.linalg.norm(np.einsum("npi,p->ni", G, theta), axis=1)
        if two:
            scale = np.dot(half, hh) / max(np.dot(hh, hh), 1e-300)
        else:  # E = e0 + c q^2 +- scale*hh, linear in (scale, e0, c)
            cols_a = [(1.0 if cols[0] else -1.0) * hh]
            y = E[:, 0] - (0 if fit_e0 else e0) - (0 if fit_mass else c * q2)
            if fit_e0:
                cols_a.append(np.ones_like(q2))
            if fit_mass:
                cols_a.append(q2)
            sol0 = np.linalg.lstsq(np.c_[tuple(cols_a)], y, rcond=None)[0]
            scale = max(sol0[0], 1e-3)
            i = 1
            if fit_e0:
                e0, i = sol0[i], i + 1
            if fit_mass:
                c = sol0[i]
            half = scale * hh
        theta = theta * scale
    else:
        theta = np.zeros(0)

    if spin_weight is None:
        spin_weight = max(np.sqrt(np.mean(half ** 2)), 1e-4)

    # --- nonlinear refinement
    layout = {"theta": slice(0, p)}
    x0 = list(theta)
    if fit_e0:
        layout["e0"] = len(x0); x0.append(e0)
    if fit_mass:
        layout["c"] = len(x0); x0.append(c)
    if fit_k0:
        layout["k0"] = slice(len(x0), len(x0) + 2); x0.extend(work.k0)
    x0 = np.array(x0, float)

    def unpack(x):
        th = x[layout["theta"]]
        e0_ = x[layout["e0"]] if fit_e0 else e0
        c_ = x[layout["c"]] if fit_mass else c
        k0_ = x[layout["k0"]] if fit_k0 else work.k0
        return th, e0_, c_, k0_

    def model_eval(x):
        th, e0_, c_, k0_ = unpack(x)
        qq = k - k0_
        h = np.einsum("npi,p->ni", work._term_fields(qq), th)
        hn = np.linalg.norm(h, axis=1)
        ep = e0_ + c_ * np.sum(qq ** 2, axis=1)
        return h, hn, np.c_[ep - hn, ep + hn]

    def residuals(x):
        h, hn, Em = model_eval(x)
        res = [energy_weight * (Em[:, cols] - E).ravel()]
        u = h / np.maximum(hn, 1e-12)[:, None]
        for sg, us, m in spin_sets:
            res.append(spin_weight * (sg * u[m] - us[m]).ravel())
        return np.concatenate(res)

    sol = least_squares(residuals, x0, method="trf", x_scale="jac")
    th, e0_f, c_f, k0_f = unpack(sol.x)

    # uncertainties
    m_res, n_par = sol.fun.size, sol.x.size
    stderr_x = np.full(n_par, np.nan)
    if m_res > n_par:
        s2 = 2 * sol.cost / (m_res - n_par)
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * s2
        stderr_x = np.sqrt(np.abs(np.diag(cov)))

    fitted = work.copy(params=th, e0=float(e0_f), k0=np.asarray(k0_f, float))
    fitted.mass = None if (not fit_mass and model.mass is None) else (
        HBAR2_2ME / c_f if abs(c_f) > 1e-12 else None)

    params = dict(zip(work.names, th.tolist()))
    stderr = dict(zip(work.names, stderr_x[layout["theta"]].tolist()))
    params["e0"] = float(e0_f)
    if fit_e0:
        stderr["e0"] = float(stderr_x[layout["e0"]])
    if fit_mass:
        params["mass"] = fitted.mass
        # propagate dc -> dm
        if fitted.mass is not None:
            stderr["mass"] = float(abs(HBAR2_2ME / c_f ** 2) * stderr_x[layout["c"]])
    if fit_k0:
        params["k0"] = fitted.k0.tolist()

    h, hn, Em = model_eval(sol.x)
    rms_e = float(np.sqrt(np.mean((Em[:, cols] - E) ** 2)))
    angles = []
    u = h / np.maximum(hn, 1e-12)[:, None]
    for sg, us, m in spin_sets:
        cosang = np.clip(np.sum(sg * u[m] * us[m], axis=1), -1, 1)
        angles.append(np.degrees(np.arccos(cosang)))
    mean_ang = float(np.mean(np.concatenate(angles))) if angles else float("nan")
    return FitResult(fitted, params, stderr, rms_e, mean_ang, bool(sol.success),
                     sol.message, int(len(k)), extra={"solution": sol})
