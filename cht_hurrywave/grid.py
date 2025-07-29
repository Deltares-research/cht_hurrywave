# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
import os
import numpy as np
import xarray as xr
import warnings
np.warnings = warnings

from .quadtree import QuadtreeMesh

class HurryWaveGrid:
    def __init__(self, model):
        
        self.model = model
        self.type = "quadtree"
        self.data = QuadtreeMesh()

    def read(self):

        if not self.model.input.variables.qtrfile:

            # No qtr file, so assuming regular grid. There must be a mask and depth file as well.
            # This first needs to be built.
            x0 = self.model.input.variables.x0
            y0 = self.model.input.variables.y0
            nmax = self.model.input.variables.nmax
            mmax = self.model.input.variables.mmax
            dx = self.model.input.variables.dx
            dy = self.model.input.variables.dy
            rotation = self.model.input.variables.rotation
            crs = self.model.crs
            self.data = QuadtreeMesh()
            # Build the grid
            self.data.build(
                x0, y0, nmax, mmax, dx, dy, rotation, crs,
            )
            # Read mask file
            if self.model.input.variables.mskfile:
               self.model.mask.initialize() # initialize with zeros
               self.data.xuds["mask"].values[:] = np.fromfile(os.path.join(self.model.path, self.model.input.variables.mskfile), dtype=np.int8)
            # And the depth file
            if self.model.input.variables.depfile:
               self.data.xuds["z"].values[:] = np.fromfile(os.path.join(self.model.path, self.model.input.variables.depfile), dtype=np.float32)

            self.data.get_exterior()
            
            # self.write()  # Write the grid to a netCDF (quadtree) file

        else:
            # netCDF quadtree file (already includes mask and depth)
            file_name = os.path.join(self.model.path, self.model.input.variables.qtrfile)
            if not os.path.exists(file_name):
                raise FileNotFoundError(f"Quadtree file '{file_name}' does not exist. Please build the grid first.")
            self.data.read(file_name)

        self.model.crs = self.data.crs

        # And update the mask (we do not want to do initialize as this creates a new mask with all zeros)
        self.model.mask.update()

    def write(self, file_name=None, version=0):
        if file_name is None:
            if not self.model.input.variables.qtrfile: 
                self.model.input.variables.qtrfile = "hurrywave.nc"
            file_name = os.path.join(self.model.path, self.model.input.variables.qtrfile)

        self.data.write(file_name)    

    def build(self, x0, y0, nmax, mmax, dx, dy, rotation,
              refinement_polygons=None,
              bathymetry_sets=None,
              bathymetry_database=None):
        
        """Build the quadtree grid."""

        print("Building mesh ...")

        # Always quadtree !
        self.type = "quadtree"

        # Clear mask datashader dataframe (datashader dataframe is contained by the QuadtreeGrid object, i.e. self.data)
        self.data.clear_datashader_dataframe()

        self.data.build(
            x0, y0, nmax, mmax, dx, dy, rotation, self.model.crs,
            refinement_polygons=refinement_polygons,
            bathymetry_sets=bathymetry_sets,
            bathymetry_database=bathymetry_database
        )

        # Built a new grid, so should clear the mask
        self.model.mask.initialize()

    def cut_inactive_cells(self):
        # Clear datashader dataframes (new ones will be created when needed by map_overlay methods)
        self.data.clear_datashader_dataframe()

        # Cut inactive cells
        # Should either send array in, or names of mask(s) to use
        self.data.cut_inactive_cells(mask_list=["mask"])

        # Changed grid, so should update the mask
        self.model.mask.update()

    def set_bathymetry(self, bathymetry_list, bathymetry_database=None):
        """Set bathymetry for the quadtree grid object."""
        self.data.set_bathymetry(bathymetry_list, bathymetry_database=bathymetry_database)

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        okay = self.data.map_overlay(file_name,
                                     xlim=xlim,
                                     ylim=ylim,
                                     color=color,
                                     width=width)
        return okay

    def exterior(self):
        """Get the exterior of the grid as a GeoDataFrame."""
        gdf = self.data.exterior()
        gdf["name"] = self.model.name
        return gdf

def read_map(self, name, file_name, dtype, fill_value):
    """Read one of the grid variables of the HurryWave model map from a binary file."""
    data = np.fromfile(file_name, dtype=dtype)
    return data
    # data = np.reshape(data, (self.mmax, self.nmax)).transpose()
    da = xr.DataArray(
        name=name,
        data=data,
        coords=self.coordinates,
        dims=("n", "m"),
        attrs={"_FillValue": dtype(fill_value)},
    )
    return da

