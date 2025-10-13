# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:24:49 2022

@author: ormondt
"""
import numpy as np

from .quadtree import QuadtreeMask

class HurryWaveMask:
    def __init__(self, model):
        self.model = model
        self.initialize()

    def initialize(self):
        self.data = QuadtreeMask(self.model.grid.data.xuds) # Create a new mask will all zeros
        self.data.clear_datashader_dataframe()              # Clear datashader dataframe

    def update(self):
        # This is called after the grid has been read or inactive cells have been cut.
        # The grid xuds has already been updated. We just need to update the crs and datashader dataframe.
        self.data.xuds = self.model.grid.data.xuds
        self.data.crs = self.model.crs
        self.data.get_datashader_dataframe()

    def build(self,
                 zmin=99999.0,
                 zmax=-99999.0,
                 include_polygon=None,
                 exclude_polygon=None,
                 boundary_polygon=None,
                 include_zmin=-99999.0,
                 include_zmax= 99999.0,
                 exclude_zmin=-99999.0,
                 exclude_zmax= 99999.0,
                 boundary_zmin=-99999.0,
                 boundary_zmax= 99999.0,
                 update_datashader_dataframe=False,
                 quiet=True):

        if not quiet:
            print("Building mask ...")

        # First clear the mask (set to zero)
        self.data.set_to_zero()   

        # Set global based on zmin and zmax  
        self.data.set_global(zmin, zmax, 1)    
                        
        # Include polygons
        self.data.set_internal_polygons(include_polygon, include_zmin, include_zmax, 1)
    
        # Exclude polygons
        self.data.set_internal_polygons(exclude_polygon, exclude_zmin, exclude_zmax, 0)

        # Boundary polygons
        self.data.set_boundary_polygons(boundary_polygon, boundary_zmin, boundary_zmax, 2)

        if update_datashader_dataframe:
            # For use in DelftDashboard
            self.data.get_datashader_dataframe()

    def has_open_boundaries(self):
        mask = self.data.xuds["mask"]
        if mask is None:
            return False
        if np.any(mask == 2):
            return True
        else:
            return False

    def map_overlay(self,
                    file_name,
                    xlim=None,
                    ylim=None,
                    active_color="yellow",
                    boundary_color="red",
                    downstream_color="blue",
                    neumann_color="purple",
                    outflow_color="green",
                    px=2,
                    width=800):
        """"""
        okay = self.data.map_overlay(file_name,
                              xlim=xlim,
                              ylim=ylim,
                              active_color=active_color,
                              boundary_color=boundary_color,
                              downstream_color=downstream_color,
                              neumann_color=neumann_color,
                              outflow_color=outflow_color,
                              px=px,
                              width=width)
        return okay
