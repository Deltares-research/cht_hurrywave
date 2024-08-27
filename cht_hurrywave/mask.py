# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import numpy as np
from matplotlib import path
import geopandas as gpd
import shapely

class HurryWaveMask:
    def __init__(self, hw):
        self.model = hw
        self.array = None

    def build(self,
              zmin=99999.0,
              zmax=-99999.0,
              include_polygon=None,
              include_zmin=-99999.0,
              include_zmax= 99999.0,
              exclude_polygon=None,
              exclude_zmin=-99999.0,
              exclude_zmax= 99999.0,
              boundary_polygon=None,
              boundary_zmin=-99999.0,
              boundary_zmax= 99999.0,
              quiet=True):

        if not quiet:
            print("Building mask mask ...")

        grid = self.model.grid
        xz = grid.xz
        yz = grid.yz
        zz = self.model.bathymetry.array
        mask = np.zeros((grid.nmax, grid.mmax), dtype=int)

        if zmin<zmax:
            # Set initial mask based on zmin and zmax
            iok = np.where((zz>=zmin) & (zz<=zmax))
            mask[iok] = 1
                        
        # Include polygons
        if include_polygon is not None:
            for ip, polygon in include_polygon.iterrows():
                inpol = inpolygon(xz, yz, polygon["geometry"])
                iok   = np.where((inpol) & (zz>=include_zmin) & (zz<=include_zmax))
                mask[iok] = 1

        # Exclude polygons
        if exclude_polygon is not None:
            for ip, polygon in exclude_polygon.iterrows():
                inpol = inpolygon(xz, yz, polygon["geometry"])
                iok   = np.where((inpol) & (zz>=exclude_zmin) & (zz<=exclude_zmax))
                mask[iok] = 0

        # Open boundary polygons
        if boundary_polygon is not None:
            if len(boundary_polygon) > 0:
                    mskbuff = np.zeros((np.shape(mask)[0] + 2, np.shape(mask)[1] + 2), dtype=int)
                    mskbuff[1:-1, 1:-1] = mask
                    # Find cells that are next to an inactive cell
                    msk4 = np.zeros((4, np.shape(mask)[0], np.shape(mask)[1]), dtype=int)
                    msk4[0, :, :] = mskbuff[0:-2, 1:-1]
                    msk4[1, :, :] = mskbuff[2:,   1:-1]
                    msk4[2, :, :] = mskbuff[1:-1, 0:-2]
                    msk4[3, :, :] = mskbuff[1:-1, 2:  ]
                    imin = msk4.min(axis=0)
                    for ip, polygon in boundary_polygon.iterrows():
                        inpol = inpolygon(xz, yz, polygon["geometry"])
                        # Only consider points that are:
                        # 1) Inside the polygon
                        # 2) Have a mask > 0
                        # 3) z>=zmin
                        # 4) z<=zmax
                        iok   = np.where((inpol) & (imin==0) & (mask>0) & (zz>=boundary_zmin) & (zz<=boundary_zmax))
                        mask[iok] = 2
                
        self.array = mask

    def read(self):
        if not self.model.input.variables.mskfile:
            return
        file_name = os.path.join(self.model.path, self.model.input.variables.mskfile)
        nmax = self.model.input.variables.nmax
        mmax = self.model.input.variables.mmax
        mskv = np.fromfile(file_name, dtype=np.int8)
        mskv = np.reshape(mskv, (nmax, mmax), order='F')
        self.array = mskv

    def write(self):
        if not self.model.input.variables.mskfile:
            self.model.input.variables.mskfile = "hurrywave.msk"
        file_name = os.path.join(self.model.path, self.model.input.variables.mskfile)
        mskv = np.reshape(self.array, np.size(self.array), order='F')
        file = open(file_name, "wb")
        file.write(np.int8(mskv))
        file.close()

    def to_gdf(self, option="all"):
        xz = self.model.grid.xz
        yz = self.model.grid.yz
        gdf_list = []
        okay = np.zeros(self.array.shape, dtype=int)
        if option == "all":
            iok = np.where((self.array > 0))
        elif option == "include":
            iok = np.where((self.array == 1))
        elif option == "boundary":
            iok = np.where((self.array == 2))
        else:
            iok = np.where((self.array > -999))
        okay[iok] = 1
        for m in range(self.model.input.variables.mmax):
            for n in range(self.model.input.variables.nmax):
                if okay[n, m] == 1:
                    point = shapely.geometry.Point(xz[n, m], yz[n, m])
                    d = {"geometry": point}
                    gdf_list.append(d)
        if gdf_list:
            gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        else:
            # Cannot set crs of gdf with empty list
            gdf = gpd.GeoDataFrame(gdf_list)

        return gdf


def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
#    xv = xv.reshape(-1)
#    yv = yv.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
#    q = [Point(xq[i], yq[i]) for i in range(xq.shape[0])]
#    mp = MultiPoint(q)
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
#    p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
    return p.contains_points(q).reshape(shape)
#    return mp.within(p)
