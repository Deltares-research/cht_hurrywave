# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
from pyproj import Transformer
import numpy as np
from matplotlib import path

from cht_bathymetry.bathymetry_database import bathymetry_database
from cht_utils.misc_tools import interp2

class Bathymetry():
    def __init__(self):

        self.z = None
    
    def get_bathymetry(self,
                       grid,
                       bathymetry_list,
                       quiet=False):

        if not quiet:
            print("Getting bathymetry data ...")
        
        xz, yz = grid.grid_coordinates_centres()
        zz = np.full((grid.nmax, grid.mmax), np.nan)

        if grid.crs.is_geographic:            
            dx = min(111111.0*grid.dx,
                     111111.0*grid.dy*np.cos(np.pi*np.max(np.abs(yz))/180.0))
        else:    
            dx = min(grid.dx, grid.dy)
        
        # Prepare transformers
        bathymetry_transformers = []  
        for bathymetry in bathymetry_list:
            bathymetry_transformers.append(Transformer.from_crs(grid.crs,
                                                                bathymetry.crs,
                                                                always_xy=True))
        
        # Loop through bathymetry datasets
        for ibathy, bathymetry in enumerate(bathymetry_list):
            transformer = Transformer.from_crs(grid.crs,
                                               bathymetry.crs,
                                               always_xy=True)
            if bathymetry.type == "source":
                if np.isnan(zz).any():
                    xzb, yzb = transformer.transform(xz, yz)        
                    if bathymetry.type == "source":                        
                        xmin = np.nanmin(np.nanmin(xzb))
                        xmax = np.nanmax(np.nanmax(xzb))
                        ymin = np.nanmin(np.nanmin(yzb))
                        ymax = np.nanmax(np.nanmax(yzb))
                        ddx  = 0.05*(xmax - xmin)
                        ddy  = 0.05*(ymax - ymin)
                        xl   = [xmin - ddx, xmax + ddx]
                        yl   = [ymin - ddy, ymax + ddy]                        
                        # Get DEM data (ddb format for now)
                        xb, yb, zb = bathymetry_database.get_data(bathymetry.name,
                                                                  xl,
                                                                  yl,
                                                                  max_cell_size=dx)
                        zb[np.where(zb<bathymetry.zmin)] = np.nan
                        zb[np.where(zb>bathymetry.zmax)] = np.nan
                        if not np.isnan(zb).all():
                            zz1     = interp2(xb, yb, zb, xzb, yzb)
                            isn     = np.where(np.isnan(zz))
                            zz[isn] = zz1[isn]
                
            elif bathymetry.type == "array":
                # Matrix provided, interpolate to subgrid mesh
#               zz = interp2(bathymetry.x, bathymetry.y, bathymetry.z, xzb, yzb)
                pass

        self.z = zz


    def save(self, file_name):        
        zv = np.reshape(self.z, np.size(self.z), order='F')
        file = open(file_name, "wb")
        file.write(np.float32(zv))        
        file.close()
        
class Mask:
    def __init__(self, 
                 grid,
                 zz,
                 zmin=99999.0,
                 zmax=-99999.0,
                 include_polygons=None,
                 exclude_polygons=None,
                 open_boundary_polygons=None,
                 quiet=True):

        if not quiet:
            print("Building mask mask ...")

        xz, yz = grid.grid_coordinates_centres()
        mask = np.zeros((grid.nmax, grid.mmax), dtype=int)

        if zmin>=zmax:
            # Do not include any points initially
            if not include_polygons:
                print("WARNING: Entire mask set to zeros! Please ensure zmax is greater than zmin, or provide include polygon(s) !")
                return
        else:
            # Set initial mask based on zmin and zmax
            iok = np.where((zz>=zmin) & (zz<=zmax))
            mask[iok] = 1
                        
        # Include polygons
        if include_polygons:
            for polygon in include_polygons:
                inpol = inpolygon(xz, yz, polygon.geometry)
                iok   = np.where((inpol) & (zz>=polygon.zmin) & (zz<=polygon.zmax))
                mask[iok] = 1

        # Exclude polygons
        if exclude_polygons:
            for polygon in exclude_polygons:
                inpol = inpolygon(xz, yz, polygon.geometry)
                iok   = np.where((inpol) & (zz>=polygon.zmin) & (zz<=polygon.zmax))
                mask[iok] = 0

        # Open boundary polygons
        if open_boundary_polygons:

            mskbuff = np.zeros((np.shape(mask)[0] + 2, np.shape(mask)[1] + 2), dtype=int)
            mskbuff[1:-1, 1:-1] = mask
            # Find cells that are next to an inactive cell
            msk4 = np.zeros((4, np.shape(mask)[0], np.shape(mask)[1]), dtype=int)
            msk4[0, :, :] = mskbuff[0:-2, 1:-1]
            msk4[1, :, :] = mskbuff[2:,   1:-1]
            msk4[2, :, :] = mskbuff[1:-1, 0:-2]
            msk4[3, :, :] = mskbuff[1:-1, 2:  ]
            imin = msk4.min(axis=0)

            for polygon in open_boundary_polygons:
                inpol = inpolygon(xz, yz, polygon.geometry)
                # Only consider points that are:
                # 1) Inside the polygon
                # 2) Have a mask > 0
                # 3) z>=zmin
                # 4) z<=zmax
                iok   = np.where((inpol) & (imin==0) & (mask>0) & (zz>=polygon.zmin) & (zz<=polygon.zmax))
                mask[iok] = 2
                
        self.mask = mask        

    def save(self, file_name):
        
        mskv = np.reshape(self.mask, np.size(self.mask), order='F')
        file = open(file_name, "wb")
        file.write(np.int8(mskv))
        file.close()

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
