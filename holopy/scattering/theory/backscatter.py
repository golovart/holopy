import warnings

import numpy as np

try:
    import numexpr as ne
    NUMEXPR_INSTALLED = True
except ModuleNotFoundError:
    NUMEXPR_INSTALLED = False
    from holopy.core.errors import PerformanceWarning
    _LENS_WARNING = ("numexpr not found. Falling back to using numpy only." +
                     " Note that Lens class is faster with numexpr")

from holopy.core import detector_points, update_metadata
from holopy.scattering.theory.scatteringtheory import ScatteringTheory

class Backscattered(ScatteringTheory):
    ### Usage: Lens(n, Backscattered(DDA(...)))
    def __init__(self, theory):
        self.theory = theory

    def _can_handle(self, scatterer):
        return self.theory._can_handle(scatterer)

    def _raw_scat_matrs(self, scatterer, pos, medium_wavevec, medium_index):
        r, theta, phi = pos
        theta_backscattered = np.pi - theta
        pos_backscattered = np.array([r, theta_backscattered, phi])

        args = (scatterer, pos_backscattered, medium_wavevec, medium_index)
        scat_matrices = np.array(self.theory._raw_scat_matrs(*args))

        # account for coordinate change in decomposition of incoming
        # light into theta, phi components:
        scat_matrices[:, 0, 0] *= -1
        scat_matrices[:, 0, 1] *= -1  # FIXME is it this or the transpose? NOW IT IS

        return scat_matrices