# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""

import os
from pyproj import CRS
import geopandas as gpd

from cht_model_builder.model_builder import ModelBuilder
from cht_hurrywave.hurrywave import HurryWave
from cht_hurrywave.waveblocking import WaveBlockingFile
import cht_utils.fileops as fo

class HurryWaveBuilder(ModelBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def build(self,
              mskfile = "hurrywave.msk",
              depfile = "hurrywave.dep",
              make_mask=True,
              get_bathymetry=True,
              make_waveblockingfile = False,
              make_tiles=True,
              quiet=False):

        crs = CRS(self.setup_config["coordinates"]["crs"])
        
        ### Mask
        inpfile = os.path.join(self.model_path, "hurrywave.inp")
        
        hw = HurryWave()
        
        hw.crs            = crs
        hw.input.variables.x0       = self.setup_config["coordinates"]["x0"]
        hw.input.variables.y0       = self.setup_config["coordinates"]["y0"]
        hw.input.variables.dx       = self.setup_config["coordinates"]["dx"]
        hw.input.variables.dy       = self.setup_config["coordinates"]["dy"]
        hw.input.variables.mmax     = self.setup_config["coordinates"]["mmax"]
        hw.input.variables.nmax     = self.setup_config["coordinates"]["nmax"]
        hw.input.variables.rotation = self.setup_config["coordinates"]["rotation"]
        hw.input.variables.crs_name = crs.name
         
        if crs.is_geographic:
            hw.input.variables.crs_type = "geographic"
            hw.input.variables.crsgeo   = 1  
        else:    
            hw.input.variables.crs_type = "projected"
            hw.input.variables.crsgeo   = 0
            if "utm" in crs.name.lower():
               hw.input.variables.crs_utmzone = crs.name[-3:]
        hw.input.variables.crs_epsg = crs.to_epsg()
                   
        for key in self.setup_config["input"]:
            setattr(hw.input.variables, key, self.setup_config["input"][key])

        # Copy hurrywave.bnd to model folder    
        if os.path.exists(os.path.join(self.data_path, "hurrywave.bnd")):
            fo.copy_file(os.path.join(self.data_path, "hurrywave.bnd"),
                        self.model_path)
            hw.input.variables.bndfile = "hurrywave.bnd"

        hw.input.variables.mskfile = mskfile
        hw.input.variables.depfile = depfile

        hw.input.model.path = self.model_path
        hw.input.write()

        ### Grid                   
        hw.grid.build()

        ### Bathymetry
        if get_bathymetry:
            hw.grid.set_bathymetry(self.bathymetry_list)
            hw.grid.write_dep_file()
        
        ### Mask
        if make_mask:
            # Initialize polygons as None
            include_polygon = None
            exclude_polygon = None
            boundary_polygon = None
            # Read polygon files and create geodataframes
            if self.setup_config["mask"]["include_polygon"]:
                include_polygon = gpd.read_file(self.setup_config["mask"]["include_polygon"])
            if self.setup_config["mask"]["exclude_polygon"]:
                exclude_polygon = gpd.read_file(self.setup_config["mask"]["exclude_polygon"])
            if self.setup_config["mask"]["open_boundary_polygon"]:
                boundary_polygon = gpd.read_file(self.setup_config["mask"]["open_boundary_polygon"])
            hw.grid.build_mask(zmin=self.setup_config["mask"]["zmin"],
                               zmax=self.setup_config["mask"]["zmax"],
                               include_polygon=include_polygon,
                               include_zmin=self.setup_config["mask"]["include_zmin"],
                               include_zmax=self.setup_config["mask"]["include_zmax"],
                               exclude_polygon=exclude_polygon,
                               exclude_zmin=self.setup_config["mask"]["exclude_zmin"],
                               exclude_zmax=self.setup_config["mask"]["exclude_zmax"],
                               boundary_polygon=boundary_polygon,
                               boundary_zmin=self.setup_config["mask"]["open_boundary_zmin"],
                               boundary_zmax=self.setup_config["mask"]["open_boundary_zmax"])
            hw.grid.write_msk_file()

        if make_waveblockingfile:
            wbl = WaveBlockingFile(model = hw)
            wblfile = 'wbl_file.nc'

            wbl.build(hw.grid,
                    bathymetry_sets= hw.bathymetry,
                    roughness_sets= [None],
                    mask = hw.mask,
                    nr_subgrid_pixels = 50,
                    file_name=os.path.join(self.model_path, wblfile),
                    nr_bins=36,
                    quiet=False,
                    showcase = False)    

            hw.input.sbgfile = wblfile          

        ### Tiles        
        if make_tiles:
            dem_names = []
            z_range   = []
            zoom_range = []
            if self.setup_config["tiling"]["zmin"]>-99990.0 or self.setup_config["tiling"]["zmax"]<99990.0:
                z_range = [self.setup_config["tiling"]["zmin"],
                           self.setup_config["tiling"]["zmax"]]
                zoom_range = [self.setup_config["tiling"]["zoom_range_min"],
                              self.setup_config["tiling"]["zoom_range_max"]]
                for dem in self.bathymetry_list:
                    dem_names.append(dem["dataset"].name)        
            hw.make_index_tiles(os.path.join(self.tile_path, "indices"),
                                zoom_range=zoom_range,
                                z_range=z_range,
                                dem_names=dem_names)
