# Copyright 2011-2016, Vinothan N. Manoharan, Thomas G. Dimiduk,
# Rebecca W. Perry, Jerome Fung, Ryan McGorty, Anna Wang, Solomon Barkley
#
# This file is part of HoloPy.
#
# HoloPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# HoloPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with HoloPy.  If not, see <http://www.gnu.org/licenses/>.
"""
Compute holograms using the discrete dipole approximation (DDA).  Currently uses
ADDA (https://github.com/adda-team/adda) to do DDA calculations.
.. moduleauthor:: Thomas G. Dimiduk <tdimiduk@physics.harvard.edu>
"""

import subprocess
import tempfile
import glob
import os
import shutil
import time
import warnings

import numpy as np

from holopy.core.utils import ensure_array, SuppressOutput
from holopy.scattering.scatterer import (
    Ellipsoid, Capsule, Cylinder, Bisphere, Sphere, Scatterer, Spheroid)
from holopy.core.errors import DependencyMissing
from holopy.scattering.theory.scatteringtheory import ScatteringTheory
from holopy.scattering import calc_holo, calc_field

def bright_cen(img, w=4):
    sh = img.shape
    peak = np.array(np.unravel_index(np.argmax(img),sh))
    peak[peak<w] = w
    bright = img[peak[0]-w:peak[0]+w+1,peak[1]-w:peak[1]+w+1]
    x,y = np.meshgrid(range(-w,w+1),range(-w,w+1))
    return peak[0]+np.sum(x*bright)/np.sum(bright), peak[1]+np.sum(y*bright)/np.sum(bright)

def calc_unpolar(detector, scat, medium_index=1., illum_wavelen=0.4, theory=None):
    # X
    holo = calc_field(detector, scat, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization=(1,0), theory=theory)
    holo_x = abs(holo.sel(vector='z').expand_dims(dim={'S':['I','Q','U','V']})).drop('vector')
    holo_x.sel(S='I')[...] = np.abs(holo.sel(vector='y'))**2 + np.abs(holo.sel(vector='x'))**2
    holo_x.sel(S='Q')[...] = np.abs(holo.sel(vector='y'))**2 - np.abs(holo.sel(vector='x'))**2
    holo_x.sel(S='U')[...] = np.real(holo.sel(vector='y')*np.conj(holo.sel(vector='x')) + holo.sel(vector='x')*np.conj(holo.sel(vector='y')))
    holo_x.sel(S='V')[...] = np.imag(holo.sel(vector='x')*np.conj(holo.sel(vector='y')) - holo.sel(vector='y')*np.conj(holo.sel(vector='x')))
    # Y
    holo = calc_field(detector, scat, medium_index=medium_index, illum_wavelen=illum_wavelen, illum_polarization=(0,1), theory=theory)
    holo_y = abs(holo.sel(vector='z').expand_dims(dim={'S':['I','Q','U','V']})).drop('vector')
    holo_y.sel(S='I')[...] = np.abs(holo.sel(vector='y'))**2 + np.abs(holo.sel(vector='x'))**2
    holo_y.sel(S='Q')[...] = np.abs(holo.sel(vector='y'))**2 - np.abs(holo.sel(vector='x'))**2
    holo_y.sel(S='U')[...] = np.real(holo.sel(vector='y')*np.conj(holo.sel(vector='x')) + holo.sel(vector='x')*np.conj(holo.sel(vector='y')))
    holo_y.sel(S='V')[...] = np.imag(holo.sel(vector='x')*np.conj(holo.sel(vector='y')) - holo.sel(vector='y')*np.conj(holo.sel(vector='x')))
    return (holo_x + holo_y)/2

def apply_pol(holo_un, th):
    return holo_un.sel(S='I') + holo_un.sel(S='Q')*np.cos(2*th) + holo_un.sel(S='U')*np.sin(2*th)

def get_unpol(holo_un):
    return holo_un.sel(S='I')
