# rashbapy

Symmetry-derived linear spin-splitting models (Rashba, Dresselhaus, modified Rashba, Weyl, out-of-plane and k-independent terms) and their fit to DFT spin textures.

* **Step 1 – model from symmetry.** `SpinSplittingModel.from_symmetry(atoms, kpoint, normal)` takes an `ase.Atoms` object, finds the space group with spglib (the library ASE itself uses; molecules with `pbc=False` use pymatgen's point-group analyser), determines the little co-group of the k-point (unitary R with Rk ≡ k, anti-unitary T·R with Rk ≡ −k) and solves the invariance condition C = det(R) R C Rᵀ for the linear tensor in h(k) = C k + b. The allowed tensors are expressed in the Mohanta–Jena term basis [PRB 111, 045140 (2025)]:

  | term | operator | parameter |
  |---|---|---|
  | `rashba` | σx ky − σy kx | α |
  | `dresselhaus` | σx kx − σy ky | β |
  | `modified_rashba` | σx ky + σy kx | M |
  | `weyl` | σx kx + σy ky | J |
  | `sz_kx`, `sz_ky` | σz kx, σz ky | γx, γy |
  | `bx`, `by`, `bz` | σx, σy, σz (only off TRIM) | b |

  These four in-plane terms span every 2×2 linear coupling, so H_R, H_D, H_MR, H_MD (Weyl), H_RD, H_MJ1…H_MJ4 and H1…H6 are all special cases.
* **Step 2 – fit.** `model.fit(k, energies, spins)` fits the free parameters, e0 and m* (optionally k0) to the two band energies and the spin texture on a k-plane. Spin directions fix the type of coupling, as in Maurer et al. [PRB 105, 155149 (2022)]; the splitting fixes the magnitude.

## Install

```bash
pip install -e .            # numpy, scipy, ase, spglib
pip install -e .[molecules,plot,test]
```

## Usage

```python
from ase.io import read
from rashbapy import SpinSplittingModel, plane_kgrid

atoms = read("POSCAR")
model = SpinSplittingModel.from_symmetry(atoms, kpoint=(0.5, 0, 0), normal=(0, 0, 1))
print(model.summary())        # group, little co-group, allowed Hamiltonian

# k-points (fractional) for a non-self-consistent SOC run on the plane
k_frac, k_plane = plane_kgrid(atoms, (0.5, 0, 0), model.frame, k_max=0.05, n=11)

# ... run DFT, read energies E (N,2) and spins S (N,2,3) in Cartesian ...
k2, s2 = model.data_to_plane(k_cartesian, S)   # to plane frame, relative to k-point
result = model.fit(k2, E, s2, k_max=0.04)
print(result.report())
fitted = result.model                           # a SpinSplittingModel object
```

### Reading VASP spin textures

```python
from rashbapy import read_spin_texture

tex = read_spin_texture("spin_texture_band45.dat", "spin_texture_band46.dat")  # kx ky E Sx Sy Sz
E, S = tex.pair(45, 46)                    # (N,2), (N,2,3), sorted by energy at each k
k2, s2 = model.data_to_plane(tex.k_cartesian(kz=0.0), S)
```

`tex.band(i)` returns one band; `tex.k_cartesian()` assumes the file's kx, ky are Cartesian (1/Å, with 2π) on the kz plane you give. See `examples/example_vasp_spin_texture.py`.

Ready-made models: `SpinSplittingModel.rashba`, `.dresselhaus`, `.modified_rashba`, `.weyl`, `.rashba_dresselhaus`, `.mj1`, `.mj2`, `.mj3`, `.mj4`, `.general`, or `SpinSplittingModel(terms=[...])`.

A model object provides:

* `energies(k)`, `spins(k)`, `hamiltonian(k)`, `h_vector(k)`
* `named_coefficients()`, `invariants()`, `classify()`, `is_persistent()`, `pst_direction()`
* `maurer_parameters()`, which gives λR and λD in the frame where the modified-Rashba term vanishes, plus the rotation angle (H_MJ1 ↔ H_RD).

`rashba_parameter(k, E)` gives α = 2E_R/k_R from a 1D band cut.

Conventions: H = e0 + ħ²q²/2m* + h(q)·σ, energies in eV, k in Å⁻¹ (including 2π), parameters in eV·Å. Spins are in the plane frame (e1, e2, n). `PlaneFrame(normal, x_axis).rotated(45)` changes the in-plane axes.

## Assumptions and limits

* Two-band (Kramers doublet) model whose spin transforms as an axial vector. Bands with extra orbital degeneracy at nonsymmorphic zone-boundary points (e.g. the 4-fold τ⊗σ structure of Tao & Tsymbal) are only described effectively.
* Linear in k only, with an isotropic kinetic term. Cubic terms are not included, so fit close to the k-point (`k_max`).
* Non-magnetic by default; `time_reversal=False` drops T-based constraints.
* Without spin data the fit warns: energies alone cannot distinguish Rashba from Dresselhaus or the ±π/4-rotated pairs.

## References

If you use rashbapy, please cite the works it builds on:

1. M. C. Mohanta and P. Jena *et al.*, Phys. Rev. B **111**, 045140 (2025), [doi:10.1103/PhysRevB.111.045140](https://doi.org/10.1103/PhysRevB.111.045140). Symmetry-allowed term basis (Rashba, Dresselhaus, modified Rashba, Weyl, …) and the Rashba parameter from a band cut.
2. B. Maurer *et al.*, Phys. Rev. B **105**, 155149 (2022), [doi:10.1103/PhysRevB.105.155149](https://doi.org/10.1103/PhysRevB.105.155149). Fitting of spin-splitting models to energies and spin textures.

## Tests

```bash
pytest tests
```
