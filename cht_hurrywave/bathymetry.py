# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import numpy as np

from cht_bathymetry.bathymetry_database import bathymetry_database
from cht_utils.misc_tools import interp3, interp2

class HurryWaveBathymetry():
    def __init__(self, hw):
        self.model = hw
        self.array = None
    
    def build(self,
              bathymetry_list,
              quiet=False):

        if not quiet:
            print("Getting bathymetry data ...")
        xz = self.model.grid.xz
        yz = self.model.grid.yz
        self.array = bathymetry_database.get_bathymetry_on_grid(xz, yz, self.model.crs, bathymetry_list)

    def set_bathymetry_from_other_source(self, grid, xb, yb, zb, rectilinearSourceData=True, fill_value=999):
        xz, yz = grid.ds.x.values, grid.ds.x.values
        zz = np.full((grid.nmax, grid.mmax), np.nan)

        if rectilinearSourceData:
            if not np.isnan(zb).all():
                zz1     = interp2_KM(xb, yb, zb, xz, yz)
                isn     = np.where(np.isnan(zz))
                zz[isn] = zz1[isn]
        else:
            zz1     = interp3(xb, yb, zb, xz, yz)
            isn     = np.where(np.isnan(zz))
            zz[isn] = zz1[isn]

        zz[np.where(np.isnan(zz))] = fill_value
        self.z = zz


    def set_bathymetry_from_other_source2(self, grid, xb, yb, zb, rectilinearSourceData=True, fill_value=999):
        xz, yz = grid.ds.x.values, grid.ds.y.values
        zz = np.full((grid.nmax, grid.mmax), np.nan)

        if rectilinearSourceData:
            if not np.isnan(zb).all():
                zz1     = interp2_KM(xb, yb, zb, xz, yz)
                isn     = np.where(np.isnan(zz))
                zz[isn] = zz1[isn]
        else:
            zz1     = interp2(xb, yb, zb, xz, yz)
            isn     = np.where(np.isnan(zz))
            zz[isn] = zz1[isn]

        zz[np.where(np.isnan(zz))] = fill_value
        self.z = zz
    

    def save(self, file_name):        
        zv = np.reshape(self.z, np.size(self.z), order='F')
        file = open(file_name, "wb")
        file.write(np.float32(zv))        
        file.close()

    def read(self):
        if not self.model.input.variables.depfile:
            return
        file_name = os.path.join(self.model.path, self.model.input.variables.depfile)
        zv   = np.fromfile(file_name, dtype=np.float32)
        nmax = self.model.input.variables.nmax
        mmax = self.model.input.variables.mmax
        self.array = np.reshape(zv, (nmax, mmax), order='F')

    def write(self):
        if not self.model.input.variables.depfile:
            self.model.input.variables.depfile = "hurrywave.dep"
        file_name = os.path.join(self.model.path, self.model.input.variables.depfile)
        zv = np.reshape(self.array, np.size(self.array), order='F')
        file = open(file_name, "wb")
        file.write(np.float32(zv))        
        file.close()
