# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import pathlib as pl
from scipy.interpolate import RegularGridInterpolator
import numpy as np
from ctypes import (
    byref,
    c_double,
    c_int,
    POINTER
)

from xmipy import XmiWrapper

class HurryWaveXmi(XmiWrapper):
    def __init__(self, dll_path, working_directory):
        # if dll_path is a string, convert it to a pathlib.Path object
        if isinstance(dll_path, str):
            dll_path = pl.Path(dll_path)
        super().__init__(dll_path, working_directory=working_directory)

    def get_domain(self):
        self.get_xz_yz()
        self.get_zb()
        self.get_zs()
        self.get_h()
        self.get_wave_parameters()

    def read(self):
        pass

    def write(self):
        pass

    def find_cell(self, x, y):
        """Find the index of the cell that contains the point (x, y)"""
        indx = self.get_hurrywave_cell_index(x, y)
        return indx
    
    def get_xz_yz(self):
        """Get the grid coordinates"""
        self.xz = self.get_value_ptr("xz")
        self.yz = self.get_value_ptr("yz")

    def get_zb(self):
        """Get the bed level"""
        self.zb = self.get_value_ptr("zb")

    def get_zs(self):
        """Get the water level"""
        self.zs = self.get_value_ptr("zs")

    def get_h(self):
        """Get the water depth"""
        self.h = self.get_value_ptr("h")

    def get_uorb(self):
        """Get the orbital velocity"""
        self.uorb = self.get_value_ptr("uorb")

    def get_wave_parameters(self):
        """Get the statistical wave parameters at the current time step"""
        self.hm0    = self.get_value_ptr("hm0")
        self.tp     = self.get_value_ptr("tp")
        self.wavdir = self.get_value_ptr("wavdir")
        self.dirspr = self.get_value_ptr("dirspr")
        self.uorb   = self.get_value_ptr("uorb")

    def get_cell_indices(self, x, y):
        # Convert x and y to double arrays
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.shape[0]
        indx = np.empty(n, dtype=np.int32)
        # Convert to pointers
        x_ptr = x.ctypes.data_as(POINTER(c_double))
        y_ptr = y.ctypes.data_as(POINTER(c_double))
        indx_ptr = indx.ctypes.data_as(POINTER(c_int))

        self._execute_function(self.lib.get_cell_indices, x_ptr, y_ptr, indx_ptr, c_int(n))

        # Index is 1-based in sfincs, so we need to subtract 1 to get the 0-based index
        return indx - 1


    def get_sfincs_cell_area(self, index):
        area = c_double(0.0)
        self._execute_function(self.lib.get_sfincs_cell_area, byref(c_int(index + 1)), byref(area))
        return area.value

    def set_water_level(self, zs):
        self.zs[:] = zs

    def set_water_depth(self, h):
        self.h[:] = h

    def run_timestep(self):
        self.update()
        return self.get_current_time()


def interp2(x0, y0, z0, x1, y1, method="linear"):

    # meanx = np.mean(x0)
    # meany = np.mean(y0)
    # x0 -= meanx
    # y0 -= meany
    # x1 -= meanx
    # y1 -= meany

    f = RegularGridInterpolator(
        (y0, x0), z0, bounds_error=False, fill_value=np.nan, method=method
    )
    # reshape x1 and y1
    if x1.ndim > 1:
        sz = x1.shape
        x1 = x1.reshape(sz[0] * sz[1])
        y1 = y1.reshape(sz[0] * sz[1])
        # interpolate
        z1 = f((y1, x1)).reshape(sz)
    else:
        z1 = f((y1, x1))

    return z1
