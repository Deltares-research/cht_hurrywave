# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import numpy as np

from cht_bathymetry.bathymetry_database import bathymetry_database

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
