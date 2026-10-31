"""Helpers: k-grids for DFT input, Rashba parameter from a band, plotting."""
import numpy as np

TWO_PI = 2.0 * np.pi


def plane_kgrid(atoms, kpoint, frame, k_max=0.1, n=21):
    """Square k-mesh in the plane of ``frame`` around ``kpoint``.

    Returns
    -------
    k_frac : (n*n, 3) fractional reciprocal coordinates (for the DFT input)
    k_plane : (n*n, 2) plane coordinates relative to kpoint (1/Å, for fitting)
    """
    A = np.array(atoms.cell[:], float)
    g = np.linspace(-k_max, k_max, n)
    k1, k2 = np.meshgrid(g, g, indexing="ij")
    k_plane = np.c_[k1.ravel(), k2.ravel()]
    k_cart = TWO_PI * np.asarray(kpoint, float) @ np.linalg.inv(A).T + frame.to_cartesian(k_plane)
    k_frac = k_cart @ A.T / TWO_PI
    return k_frac, k_plane


def rashba_parameter(k, energy, kind="conduction"):
    """Rashba constant alpha = 2 E_R / k_R from a 1D band cut (Mohanta & Jena).

    k : 1D array (1/Å) along a line through the crossing point at k = 0
    energy : energies of one spin branch (eV)
    kind : 'conduction' (band minimum) or 'valence' (band maximum)
    Returns (alpha [eV·Å], E_R [eV], k_R [1/Å]).
    """
    k = np.asarray(k, float)
    E = np.asarray(energy, float)
    i0 = int(np.argmin(np.abs(k)))
    ie = int(np.argmin(E) if kind == "conduction" else np.argmax(E))
    E_R = abs(E[ie] - E[i0])
    k_R = abs(k[ie] - k[i0])
    if k_R == 0:
        raise ValueError("band extremum is at the crossing point; no Rashba offset")
    return 2 * E_R / k_R, E_R, k_R


def plot_spin_texture(model=None, band=1, k_max=0.1, n=15, ax=None,
                      k=None, spins=None, color_component=2, **quiver_kw):
    """Quiver plot of the in-plane spin, coloured by one spin component.

    Either a model (evaluated on a grid) or data (k (N,2), spins (N,3)).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    if k is None:
        g = np.linspace(-k_max, k_max, n)
        k1, k2 = np.meshgrid(g, g, indexing="ij")
        k = np.c_[k1.ravel(), k2.ravel()] + model.k0
        spins = model.spins(k)[:, band]
    k, spins = np.asarray(k), np.asarray(spins)
    col = spins[:, color_component]
    if np.max(np.abs(col)) < 0.2:  # no out-of-plane component: plain dark arrows
        q = ax.quiver(k[:, 0], k[:, 1], spins[:, 0], spins[:, 1], color="0.15",
                      pivot="mid", **quiver_kw)
    else:
        q = ax.quiver(k[:, 0], k[:, 1], spins[:, 0], spins[:, 1], col,
                      cmap="coolwarm", clim=(-1, 1), pivot="mid", **quiver_kw)
    ax.set_xlabel("k1 (1/Å)")
    ax.set_ylabel("k2 (1/Å)")
    ax.set_aspect("equal")
    return ax, q
