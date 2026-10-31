"""Fit VASP spin-texture data (+ POSCAR) and compare fit and DFT on the same grid.

data/spin_texture_band45.dat holds one band (kx ky E Sx Sy Sz, k in 1/Å incl.
2π) on the kz=0 plane. With only one band, whether it is the lower or upper
branch of the pair is not known, so both are fitted and the better one is kept.
If the file(s) hold both bands of the pair, ``tex.pair(lo, hi)`` is used instead.
"""
from pathlib import Path

import numpy as np
from ase.io import read

from rashbapy import SpinSplittingModel, plot_spin_texture, read_spin_texture

K_MAX = 0.25  # 1/Å, region around the k-point where the linear model is used

here = Path(__file__).parent / "data"
atoms = read(here / "POSCAR")
files = [here / "spin_texture_band45.dat"]
if (here / "spin_texture_band46.dat").exists():  # partner band: full two-band fit
    files.append(here / "spin_texture_band46.dat")
tex = read_spin_texture(*files)

# symmetry-allowed model around Gamma in the kx-ky plane
model = SpinSplittingModel.from_symmetry(atoms, kpoint=(0, 0, 0), normal=(0, 0, 1))
print(model.summary().splitlines()[1], "\n")

# fit region: near the k-point; the k-point itself has no defined spin direction
k_cart = tex.k_cartesian()
k2 = model.data_to_plane(k_cart)
r = np.linalg.norm(k2, axis=1)
use = (r <= K_MAX) & (r > 1e-6)

if len(tex.bands) >= 2:
    E, S = tex.pair(*tex.bands[:2])
    k2, S2 = model.data_to_plane(k_cart, S)
    result = model.fit(k2[use], E[use], S2[use])
    cols = [0, 1]
    band = 0  # branch shown in the spin plots (0 = lower, 1 = upper)
else:
    E, S = tex.band(tex.bands[0])
    k2, S2 = model.data_to_plane(k_cart, S)
    fits = {b: model.fit(k2[use], E[use], S2[use], energy_band=b) for b in ("lower", "upper")}
    cost = {b: f.extra["solution"].cost for b, f in fits.items()}
    best = min(cost, key=cost.get)
    print(f"band {tex.bands[0]}: cost lower {cost['lower']:.3g}, upper {cost['upper']:.3g} -> {best}\n")
    result = fits[best]
    cols = [0 if best == "lower" else 1]
    band = cols[0]
print(result.report())

# ---- fit vs DFT on the same k-points
fit_E = result.model.energies(k2[use])[:, cols]
fit_S = result.model.spins(k2[use])[:, band]
unit = lambda s: s / np.linalg.norm(s, axis=-1, keepdims=True)
dft_S = S2[use] if S2.ndim == 2 else S2[use][:, band]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
kw = dict(scale=18, width=0.006)
plot_spin_texture(k=k2[use], spins=unit(dft_S), ax=axes[0], **kw)[0].set_title("DFT")
plot_spin_texture(k=k2[use], spins=fit_S, ax=axes[1], **kw)[0].set_title("fit")
ax = axes[2]
cut = np.abs(k2[use][:, 1]) < 1e-6  # ky = 0
kx = np.linspace(-K_MAX, K_MAX, 101)
line = np.c_[kx, np.zeros_like(kx)]
ax.plot(k2[use][cut, 0], E[use][cut], "o", color="C0", label="DFT")
ax.plot(kx, result.model.energies(line)[:, cols], "-", color="C1")
ax.plot([], [], "-", color="C1", label="fit")
ax.set(xlabel="kx (1/Å)", ylabel="E (eV)", title="energy at ky = 0")
ax.legend()
fig.tight_layout()
fig.savefig("vasp_spin_texture_fit.png", dpi=120)
print("\nwrote vasp_spin_texture_fit.png")
