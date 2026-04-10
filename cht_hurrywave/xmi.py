"""
HurryWave BMI/XMI wrapper.

Provides :class:`HurryWaveXmi`, which extends ``XmiWrapper`` with
HurryWave-specific helpers for retrieving grid coordinates, bed level,
water level, wave parameters, and cell indices via the shared-library
interface.

Also provides a standalone :func:`interp2` bilinear interpolation utility.
"""

import pathlib as pl
from ctypes import POINTER, byref, c_double, c_int

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from xmipy import XmiWrapper


class HurryWaveXmi(XmiWrapper):
    """
    XMI wrapper for the HurryWave shared library.

    Parameters
    ----------
    dll_path : str or pathlib.Path
        Path to the HurryWave shared library (``hurrywave.dll`` /
        ``libhurrywave.so``).
    working_directory : str or pathlib.Path
        Directory from which the model is run.
    """

    def __init__(self, dll_path, working_directory) -> None:
        if isinstance(dll_path, str):
            dll_path = pl.Path(dll_path)
        super().__init__(dll_path, working_directory=working_directory)

    def get_domain(self) -> None:
        """Retrieve all primary domain arrays from the shared library."""
        self.get_xz_yz()
        self.get_zb()
        self.get_zs()
        self.get_h()
        self.get_wave_parameters()

    def read(self) -> None:
        """No-op placeholder (reading is handled by XmiWrapper)."""
        pass

    def write(self) -> None:
        """No-op placeholder (writing is handled by XmiWrapper)."""
        pass

    def find_cell(self, x: float, y: float) -> int:
        """
        Find the grid cell index containing point ``(x, y)``.

        Parameters
        ----------
        x : float
            X-coordinate.
        y : float
            Y-coordinate.

        Returns
        -------
        int
            Zero-based cell index.
        """
        return self.get_hurrywave_cell_index(x, y)

    def get_xz_yz(self) -> None:
        """Retrieve cell-centre x and y coordinate arrays (``xz``, ``yz``)."""
        self.xz = self.get_value_ptr("xz")
        self.yz = self.get_value_ptr("yz")

    def get_zb(self) -> None:
        """Retrieve the bed-level array (``zb``)."""
        self.zb = self.get_value_ptr("zb")

    def get_zs(self) -> None:
        """Retrieve the water-level array (``zs``)."""
        self.zs = self.get_value_ptr("zs")

    def get_h(self) -> None:
        """Retrieve the water-depth array (``h``)."""
        self.h = self.get_value_ptr("h")

    def get_uorb(self) -> None:
        """Retrieve the orbital velocity array (``uorb``)."""
        self.uorb = self.get_value_ptr("uorb")

    def get_wave_parameters(self) -> None:
        """Retrieve all statistical wave-parameter arrays (Hm0, Tp, direction, spreading, uorb)."""
        self.hm0 = self.get_value_ptr("hm0")
        self.tp = self.get_value_ptr("tp")
        self.wavdir = self.get_value_ptr("wavdir")
        self.dirspr = self.get_value_ptr("dirspr")
        self.uorb = self.get_value_ptr("uorb")

    def get_cell_indices(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return zero-based cell indices for arrays of query coordinates.

        Parameters
        ----------
        x : numpy.ndarray
            1-D array of x-coordinates.
        y : numpy.ndarray
            1-D array of y-coordinates.

        Returns
        -------
        numpy.ndarray of int
            Zero-based cell indices (shape matching *x*).
        """
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n = x.shape[0]
        indx = np.empty(n, dtype=np.int32)
        x_ptr = x.ctypes.data_as(POINTER(c_double))
        y_ptr = y.ctypes.data_as(POINTER(c_double))
        indx_ptr = indx.ctypes.data_as(POINTER(c_int))

        self._execute_function(
            self.lib.get_cell_indices, x_ptr, y_ptr, indx_ptr, c_int(n)
        )

        return indx - 1

    def get_sfincs_cell_area(self, index: int) -> float:
        """
        Return the area of a cell by its zero-based index.

        Parameters
        ----------
        index : int
            Zero-based cell index.

        Returns
        -------
        float
            Cell area [m²].
        """
        area = c_double(0.0)
        self._execute_function(
            self.lib.get_sfincs_cell_area, byref(c_int(index + 1)), byref(area)
        )
        return area.value

    def set_water_level(self, zs: np.ndarray) -> None:
        """
        Overwrite the water-level array.

        Parameters
        ----------
        zs : numpy.ndarray
            New water-level values.
        """
        self.zs[:] = zs

    def set_water_depth(self, h: np.ndarray) -> None:
        """
        Overwrite the water-depth array.

        Parameters
        ----------
        h : numpy.ndarray
            New water-depth values.
        """
        self.h[:] = h

    def run_timestep(self) -> float:
        """
        Advance the model by one time step and return the new simulation time.

        Returns
        -------
        float
            Current simulation time after the update.
        """
        self.update()
        return self.get_current_time()


def interp2(
    x0: np.ndarray,
    y0: np.ndarray,
    z0: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Bilinear (or nearest-neighbour) interpolation from a regular grid.

    Parameters
    ----------
    x0 : numpy.ndarray
        1-D array of source x-coordinates.
    y0 : numpy.ndarray
        1-D array of source y-coordinates.
    z0 : numpy.ndarray
        2-D array of source values with shape ``(len(y0), len(x0))``.
    x1 : numpy.ndarray
        Target x-coordinates (1-D or 2-D).
    y1 : numpy.ndarray
        Target y-coordinates (same shape as *x1*).
    method : str, optional
        Interpolation method passed to ``RegularGridInterpolator``.
        Default is ``"linear"``.

    Returns
    -------
    numpy.ndarray
        Interpolated values at ``(x1, y1)``, same shape as *x1*.
    """
    f = RegularGridInterpolator(
        (y0, x0), z0, bounds_error=False, fill_value=np.nan, method=method
    )
    if x1.ndim > 1:
        sz = x1.shape
        x1 = x1.reshape(sz[0] * sz[1])
        y1 = y1.reshape(sz[0] * sz[1])
        z1 = f((y1, x1)).reshape(sz)
    else:
        z1 = f((y1, x1))

    return z1
