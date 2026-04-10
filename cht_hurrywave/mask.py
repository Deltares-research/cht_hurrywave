"""
HurryWave mask management.

Provides :class:`HurryWaveMask`, which wraps a
:class:`~.quadtree.QuadtreeMask` and exposes methods to initialise, build,
and query the active / boundary / inactive cell flags of the model grid.
"""

import numpy as np

from .quadtree import QuadtreeMask


class HurryWaveMask:
    """
    Manage the cell-activity mask for a HurryWave model.

    Parameters
    ----------
    model : HurryWave
        Parent model instance.
    """

    def __init__(self, model) -> None:
        self.model = model
        self.initialize()

    def initialize(self) -> None:
        """Create a fresh mask (all zeros) from the current grid dataset."""
        self.data = QuadtreeMask(self.model.grid.data.xuds)
        self.data.clear_datashader_dataframe()

    def update(self) -> None:
        """
        Synchronise the mask after the underlying grid has changed.

        Called after :meth:`~HurryWaveGrid.read` or
        :meth:`~HurryWaveGrid.cut_inactive_cells` to keep the mask's
        ``xuds`` reference and CRS consistent with the grid, and to
        rebuild the datashader dataframe.
        """
        self.data.xuds = self.model.grid.data.xuds
        self.data.crs = self.model.crs
        self.data.get_datashader_dataframe()

    def build(
        self,
        zmin: float = 99999.0,
        zmax: float = -99999.0,
        include_polygon=None,
        exclude_polygon=None,
        boundary_polygon=None,
        include_zmin: float = -99999.0,
        include_zmax: float = 99999.0,
        exclude_zmin: float = -99999.0,
        exclude_zmax: float = 99999.0,
        boundary_zmin: float = -99999.0,
        boundary_zmax: float = 99999.0,
        update_datashader_dataframe: bool = False,
        quiet: bool = True,
    ) -> None:
        """
        Build the mask from depth thresholds and optional polygons.

        Cells are first set to zero, then marked as active (1) where the bed
        level falls within [*zmin*, *zmax*].  Include / exclude / boundary
        polygons can further override the mask within their extents.

        Parameters
        ----------
        zmin : float, optional
            Global lower depth threshold for active cells.
        zmax : float, optional
            Global upper depth threshold for active cells.
        include_polygon : geopandas.GeoDataFrame or None, optional
            Polygons inside which cells are forced active.
        exclude_polygon : geopandas.GeoDataFrame or None, optional
            Polygons inside which cells are forced inactive.
        boundary_polygon : geopandas.GeoDataFrame or None, optional
            Polygons used to mark open-boundary cells (mask = 2).
        include_zmin : float, optional
            Minimum depth within include polygons.
        include_zmax : float, optional
            Maximum depth within include polygons.
        exclude_zmin : float, optional
            Minimum depth within exclude polygons.
        exclude_zmax : float, optional
            Maximum depth within exclude polygons.
        boundary_zmin : float, optional
            Minimum depth within boundary polygons.
        boundary_zmax : float, optional
            Maximum depth within boundary polygons.
        update_datashader_dataframe : bool, optional
            When ``True`` the datashader dataframe is rebuilt after the mask
            is computed (useful for live display in DelftDashboard).
        quiet : bool, optional
            Suppress progress output.  Default is ``True``.
        """
        if not quiet:
            print("Building mask ...")

        self.data.set_to_zero()
        self.data.set_global(zmin, zmax, 1)
        self.data.set_internal_polygons(include_polygon, include_zmin, include_zmax, 1)
        self.data.set_internal_polygons(exclude_polygon, exclude_zmin, exclude_zmax, 0)
        self.data.set_boundary_polygons(
            boundary_polygon, boundary_zmin, boundary_zmax, 2
        )

        if update_datashader_dataframe:
            self.data.get_datashader_dataframe()

    def has_open_boundaries(self) -> bool:
        """
        Check whether any open-boundary cells (mask == 2) exist.

        Returns
        -------
        bool
            ``True`` if at least one cell has mask value 2.
        """
        mask = self.data.xuds["mask"]
        if mask is None:
            return False
        return bool(np.any(mask == 2))

    def map_overlay(
        self,
        file_name: str,
        xlim=None,
        ylim=None,
        active_color: str = "yellow",
        boundary_color: str = "red",
        downstream_color: str = "blue",
        neumann_color: str = "purple",
        outflow_color: str = "green",
        px: int = 2,
        width: int = 800,
    ) -> bool:
        """
        Render a datashader overlay image of the mask.

        Parameters
        ----------
        file_name : str
            Output PNG path (without extension).
        xlim : list of float or None, optional
            Longitude range.
        ylim : list of float or None, optional
            Latitude range.
        active_color : str, optional
            Colour for active cells.  Default ``"yellow"``.
        boundary_color : str, optional
            Colour for open-boundary cells.  Default ``"red"``.
        downstream_color : str, optional
            Colour for downstream cells.  Default ``"blue"``.
        neumann_color : str, optional
            Colour for Neumann cells.  Default ``"purple"``.
        outflow_color : str, optional
            Colour for outflow cells.  Default ``"green"``.
        px : int, optional
            Spread radius in pixels.  Default is 2.
        width : int, optional
            Output image width in pixels.  Default is 800.

        Returns
        -------
        bool
            ``True`` on success, ``False`` on failure.
        """
        okay = self.data.map_overlay(
            file_name,
            xlim=xlim,
            ylim=ylim,
            active_color=active_color,
            boundary_color=boundary_color,
            downstream_color=downstream_color,
            neumann_color=neumann_color,
            outflow_color=outflow_color,
            px=px,
            width=width,
        )
        return okay
