"""Readers for DFT output: VASP spin-texture files."""
from dataclasses import dataclass

from glob import glob

import numpy as np


@dataclass
class SpinTexture:
    """Spin texture of one or more bands on a common k-mesh.

    Attributes
    ----------
    k : (N, 2) in-plane k-points (kx, ky), 1/Å as written in the file
    bands : list of band indices, in file order
    energies : (nbands, N) eV
    spins : (nbands, N, 3) Sx, Sy, Sz as written in the file
    grid_size : tuple of ints from the ``#Grid-Size`` header, or None
    """
    k: np.ndarray
    bands: list
    energies: np.ndarray
    spins: np.ndarray
    grid_size: tuple = None

    def _row(self, band):
        try:
            return self.bands.index(band)
        except ValueError:
            raise KeyError(f"band {band} not in file; available: {self.bands}") from None

    def band(self, index):
        """-> energies (N,), spins (N, 3) of band ``index``."""
        i = self._row(index)
        return self.energies[i], self.spins[i]

    def pair(self, lower, upper):
        """Energies (N, 2) and spins (N, 2, 3) of two bands, as ``model.fit`` expects.

        At every k-point the two bands are sorted by energy (carrying their
        spins along), so band crossings do not mix up the branches.
        """
        e = self.energies[[self._row(lower), self._row(upper)]].T
        s = self.spins[[self._row(lower), self._row(upper)]].transpose(1, 0, 2)
        order = np.argsort(e, axis=1, kind="stable")
        return (np.take_along_axis(e, order, axis=1),
                np.take_along_axis(s, order[..., None], axis=1))

    def k_cartesian(self, kz=0.0, origin=(0.0, 0.0, 0.0)):
        """(N, 3) Cartesian k-points: the plane at height ``kz`` (1/Å), shifted
        by ``origin``. The file has no kz, so it is assumed to be the kx-ky plane."""
        k = np.c_[self.k, np.full(len(self.k), kz)]
        return k + np.asarray(origin, float)


def read_spin_texture(*paths):
    """Read VASP spin-texture file(s) (columns kx ky E Sx Sy Sz).

    Each band is a block introduced by ``#Band-Index:`` (and, optionally,
    ``#Grid-Size:``); a file may hold several blocks, and several files can be
    passed to combine bands (glob patterns such as ``"SPIN_TEXTURE_*.dat"`` are
    expanded and sorted). All bands must share the same k-points.
    """
    paths = [q for p in paths for q in (sorted(glob(str(p))) or [p])]
    blocks, grid_size = [], None
    for path in paths:
        index, rows = None, None
        with open(path) as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                if s.startswith("#"):
                    key, _, val = s[1:].partition(":")
                    key = key.strip().lower()
                    if key == "band-index":
                        index = int(val.split()[0])
                        rows = []
                        blocks.append((index, rows))
                    elif key == "grid-size":
                        grid_size = tuple(int(x) for x in val.split())
                    continue
                values = [float(x) for x in s.replace("D", "E").replace("d", "e").split()]
                if len(values) != 6:
                    raise ValueError(f"{path}: expected 6 columns, got {len(values)}: {s!r}")
                if rows is None:  # data without a #Band-Index header
                    rows = []
                    blocks.append((0, rows))
                rows.append(values)
    blocks = [(i, r) for i, r in blocks if r]
    if not blocks:
        raise ValueError("no spin-texture data found")
    indices = [i for i, _ in blocks]
    if len(set(indices)) != len(indices):
        raise ValueError(f"duplicate band indices: {indices}")
    data = [np.array(r, float) for _, r in blocks]
    for d in data[1:]:
        if d.shape != data[0].shape or not np.allclose(d[:, :2], data[0][:, :2], atol=1e-5):
            raise ValueError("bands are not given on the same k-points")
    return SpinTexture(k=data[0][:, :2], bands=indices,
                       energies=np.array([d[:, 2] for d in data]),
                       spins=np.array([d[:, 3:] for d in data]),
                       grid_size=grid_size)
