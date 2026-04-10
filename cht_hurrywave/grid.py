"""
HurryWave grid management.

Provides :class:`HurryWaveGrid`, which wraps a :class:`~.quadtree.QuadtreeMesh`
and exposes methods to build, read, write, and post-process the computational
grid (including bathymetry and exterior computation).

Also provides a standalone :func:`read_map` helper for reading binary grid
variable files.
"""

import os
import warnings

import numpy as np
import xarray as xr

np.warnings = warnings

from .quadtree import QuadtreeMesh


class HurryWaveGrid:
    """
    Manage the HurryWave computational grid.

    Parameters
    ----------
    model : HurryWave
        Parent model instance.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.type = "quadtree"
        self.data = QuadtreeMesh()

    def read(self) -> None:
        """
        Read the grid from disk.

        When no ``qtrfile`` is set in the input variables the grid is built
        in-memory from the regular-grid parameters and binary mask/depth files
        are loaded.  When ``qtrfile`` is set the netCDF quadtree file is
        loaded directly.

        Raises
        ------
        FileNotFoundError
            If the quadtree netCDF file referenced by ``qtrfile`` does not
            exist.
        """
        if not self.model.input.variables.qtrfile:
            x0 = self.model.input.variables.x0
            y0 = self.model.input.variables.y0
            nmax = self.model.input.variables.nmax
            mmax = self.model.input.variables.mmax
            dx = self.model.input.variables.dx
            dy = self.model.input.variables.dy
            rotation = self.model.input.variables.rotation
            crs = self.model.crs
            self.data = QuadtreeMesh()
            self.data.build(x0, y0, nmax, mmax, dx, dy, rotation, crs)

            if self.model.input.variables.mskfile:
                self.model.mask.initialize()
                self.data.xuds["mask"].values[:] = np.fromfile(
                    os.path.join(self.model.path, self.model.input.variables.mskfile),
                    dtype=np.int8,
                )
            if self.model.input.variables.depfile:
                self.data.xuds["z"].values[:] = np.fromfile(
                    os.path.join(self.model.path, self.model.input.variables.depfile),
                    dtype=np.float32,
                )

            self.data.get_exterior()

        else:
            file_name = os.path.join(
                self.model.path, self.model.input.variables.qtrfile
            )
            if not os.path.exists(file_name):
                raise FileNotFoundError(
                    f"Quadtree file '{file_name}' does not exist. Please build the grid first."
                )
            self.data.read(file_name)

        self.model.crs = self.data.crs
        self.model.mask.update()

    def write(self, file_name: str | None = None, version: int = 0) -> None:
        """
        Write the grid to a netCDF quadtree file.

        Parameters
        ----------
        file_name : str or None, optional
            Output path.  When ``None`` the filename is taken from
            ``input.variables.qtrfile`` (defaulting to ``"hurrywave.nc"``).
        version : int, optional
            Reserved for future file-format versioning.
        """
        if file_name is None:
            if not self.model.input.variables.qtrfile:
                self.model.input.variables.qtrfile = "hurrywave.nc"
            file_name = os.path.join(
                self.model.path, self.model.input.variables.qtrfile
            )

        self.data.write(file_name)

    def build(
        self,
        x0: float,
        y0: float,
        nmax: int,
        mmax: int,
        dx: float,
        dy: float,
        rotation: float,
        refinement_polygons=None,
        bathymetry_sets=None,
        bathymetry_database=None,
    ) -> None:
        """
        Build the quadtree mesh.

        Parameters
        ----------
        x0 : float
            X-coordinate of the grid origin.
        y0 : float
            Y-coordinate of the grid origin.
        nmax : int
            Number of cells in the n-direction at the coarsest level.
        mmax : int
            Number of cells in the m-direction at the coarsest level.
        dx : float
            Cell width at the coarsest level.
        dy : float
            Cell height at the coarsest level.
        rotation : float
            Grid rotation angle [degrees].
        refinement_polygons : geopandas.GeoDataFrame or None, optional
            Polygons used to drive local mesh refinement.
        bathymetry_sets : list or None, optional
            Bathymetry datasets passed to the refinement algorithm for
            depth-dependent cell splitting.
        bathymetry_database : object or None, optional
            Bathymetry database used to query depth values.
        """
        print("Building mesh ...")

        self.type = "quadtree"
        self.data.clear_datashader_dataframe()

        self.data.build(
            x0,
            y0,
            nmax,
            mmax,
            dx,
            dy,
            rotation,
            self.model.crs,
            refinement_polygons=refinement_polygons,
            bathymetry_sets=bathymetry_sets,
            bathymetry_database=bathymetry_database,
        )

        self.model.mask.initialize()

    def cut_inactive_cells(self) -> None:
        """Remove cells with mask value < 1 from the grid and refresh the mask."""
        self.data.clear_datashader_dataframe()
        self.data.cut_inactive_cells(mask_list=["mask"])
        self.model.mask.update()

    def set_bathymetry(self, bathymetry_list, bathymetry_database=None) -> None:
        """
        Assign bathymetry values to the grid.

        Parameters
        ----------
        bathymetry_list : list
            Ordered list of bathymetry datasets.
        bathymetry_database : object or None, optional
            Bathymetry database providing point-query functionality.
        """
        self.data.set_bathymetry(
            bathymetry_list, bathymetry_database=bathymetry_database
        )

    def set_bathymetry_mean_wet(
        self,
        bathymetry_sets,
        bathymetry_database=None,
        nr_subgrid_pixels: int = 20,
        threshold_level: float = 0.0,
        quiet: bool = True,
        progress_bar=None,
    ) -> None:
        """
        Set bathymetry using the mean depth of wet subgrid pixels.

        Parameters
        ----------
        bathymetry_sets : list
            Ordered list of bathymetry datasets.
        bathymetry_database : object or None, optional
            Bathymetry database.
        nr_subgrid_pixels : int, optional
            Number of subgrid pixels per cell side.  Default is 20.
        threshold_level : float, optional
            Elevation threshold separating wet from dry pixels.  Default 0.0.
        quiet : bool, optional
            Suppress progress output when ``True``.
        progress_bar : object or None, optional
            Optional progress-bar object with ``set_value`` / ``was_canceled``
            interface.
        """
        self.data.set_bathymetry_mean_wet(
            bathymetry_sets,
            bathymetry_database=bathymetry_database,
            nr_subgrid_pixels=nr_subgrid_pixels,
            threshold_level=threshold_level,
            quiet=quiet,
            progress_bar=progress_bar,
        )

    def map_overlay(
        self,
        file_name: str,
        xlim=None,
        ylim=None,
        color: str = "black",
        width: int = 800,
    ) -> bool:
        """
        Render a datashader overlay image of the grid lines.

        Parameters
        ----------
        file_name : str
            Output PNG path (without extension).
        xlim : list of float or None, optional
            Longitude range ``[lon_min, lon_max]``.
        ylim : list of float or None, optional
            Latitude range ``[lat_min, lat_max]``.
        color : str, optional
            Line colour.  Default is ``"black"``.
        width : int, optional
            Output image width in pixels.  Default is 800.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        okay = self.data.map_overlay(
            file_name, xlim=xlim, ylim=ylim, color=color, width=width
        )
        return okay

    def exterior(self):
        """
        Return the domain exterior as a GeoDataFrame.

        Returns
        -------
        geopandas.GeoDataFrame
            Single-row GeoDataFrame with a ``"name"`` column set to the model
            name.
        """
        gdf = self.data.exterior()
        gdf["name"] = self.model.name
        return gdf


def read_map(
    name: str,
    file_name: str,
    dtype,
    fill_value,
) -> np.ndarray:
    """
    Read a flat binary grid-variable file.

    Parameters
    ----------
    name : str
        Variable name (used for labelling purposes).
    file_name : str
        Path to the binary file.
    dtype : numpy dtype
        Data type of the stored values.
    fill_value : scalar
        Fill / no-data value used in the original file.

    Returns
    -------
    numpy.ndarray
        1-D array of values read from *file_name*.
    """
    data = np.fromfile(file_name, dtype=dtype)
    return data
