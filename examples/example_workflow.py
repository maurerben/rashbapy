"""End-to-end example: symmetry -> model -> k-grid for DFT -> fit.

The "DFT data" are synthetic here; replace them with your own energies and
spin expectation values (Cartesian) on the k-points from plane_kgrid.
"""
import numpy as np
from ase.spacegroup import crystal

from rashbapy import SpinSplittingModel, plane_kgrid, plot_spin_texture

# BiInO3-like structure, space group Pna2_1 (No. 33); generic positions
atoms = crystal(["Bi", "In", "O"],
                [[0.123, 0.217, 0.311], [0.371, 0.083, 0.647], [0.29, 0.41, 0.05]],
                spacegroup=33, cellpar=[5.9, 8.4, 5.0, 90, 90, 90])

# 1. symmetry-allowed model around X in the kx-ky plane
model = SpinSplittingModel.from_symmetry(atoms, kpoint=(0.5, 0, 0), normal=(0, 0, 1))
print(model.summary(), "\n")

# k-points for the non-self-consistent DFT run
k_frac, k_plane = plane_kgrid(atoms, (0.5, 0, 0), model.frame, k_max=0.05, n=11)

# --- synthetic stand-in for DFT output (Cartesian k, energies, spins)
truth = model.copy(params={"alpha": 0.8, "M": -0.792}, mass=0.35, e0=2.76)
k_cart = 2 * np.pi * k_frac @ np.linalg.inv(atoms.cell[:]).T
energies = truth.energies(k_plane) + 1e-3 * np.random.default_rng(1).standard_normal((len(k_plane), 2))
spins_cart = truth.frame.to_cartesian(truth.spins(k_plane).reshape(-1, 3)).reshape(-1, 2, 3)

# 2. fit
k2, s2 = model.data_to_plane(k_cart, spins_cart)
result = model.fit(k2, energies, s2)
print(result.report())

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    plot_spin_texture(k=k2, spins=s2[:, 0], ax=axes[0]); axes[0].set_title("data, lower band")
    plot_spin_texture(result.model, band=0, k_max=0.05, n=11, ax=axes[1]); axes[1].set_title("fit, lower band")
    fig.tight_layout(); fig.savefig("spin_texture_fit.png", dpi=120)
except ImportError:
    pass
