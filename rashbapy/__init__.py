"""rashbapy — symmetry-derived linear spin-splitting models and fitting.

1. ``SpinSplittingModel.from_symmetry(atoms, kpoint, normal)`` analyses the
   symmetry of an ``ase.Atoms`` object (spglib; pymatgen for molecules) and
   sets up the allowed Rashba / Dresselhaus / modified-Rashba / Weyl /
   out-of-plane terms (Mohanta & Jena, PRB 111, 045140 (2025)).
2. ``model.fit(k, energies, spins)`` fits the model to the energies and
   spin texture on a k-plane (Maurer et al., PRB 105, 155149 (2022)).
"""
from .fit import FitResult, fit_model
from .model import SpinSplittingModel
from .symmetry import PlaneFrame, SymmetryInfo, allowed_terms, analyze_symmetry
from .terms import HBAR2_2ME, TERM_NAMES, TERM_STRINGS
from .utils import plane_kgrid, plot_spin_texture, rashba_parameter

__version__ = "0.1.0"

__all__ = [
    "SpinSplittingModel", "fit_model", "FitResult", "PlaneFrame", "SymmetryInfo",
    "analyze_symmetry", "allowed_terms", "plane_kgrid", "rashba_parameter",
    "plot_spin_texture", "TERM_NAMES", "TERM_STRINGS", "HBAR2_2ME",
]
