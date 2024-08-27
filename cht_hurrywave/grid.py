# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
from pyproj import Transformer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
import geopandas as gpd
import pandas as pd
import shapely
import math
import time
import xarray as xr
from affine import Affine
from cht_bathymetry.bathymetry_database import bathymetry_database
import xugrid as xu
#from .to_xugrid import xug
import warnings
np.warnings = warnings

import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image

class HurryWaveGrid:
    def __init__(self, model):
        # RegularGrid contains coordinates, mask, bed_level, obstacles
        self.model         = model
        self.xugrid        = None
        self.build()

    def build(self):

        self.x0            = self.model.input.variables.x0
        self.y0            = self.model.input.variables.y0
        self.dx            = self.model.input.variables.dx
        self.dy            = self.model.input.variables.dy
        self.nmax          = self.model.input.variables.nmax
        self.mmax          = self.model.input.variables.mmax
        self.rotation      = self.model.input.variables.rotation        

        self.xugrid        = None

        self.ds            = xr.Dataset()

        self.ds["bed_level"] = xr.DataArray(
            data=np.full((self.nmax, self.mmax), -99999.0, dtype=float),
            coords=self.coordinates,
            dims=("n", "m"),
            attrs={"_FillValue": -99999.0},
        )

        self.ds["mask"] = xr.DataArray(
            data=np.full((self.nmax, self.mmax), 0, dtype=np.int8),
            coords=self.coordinates,
            dims=("n", "m"),
            attrs={"_FillValue": 0},
        )

    @property
    def transform(self):
        """Return the affine transform of the regular grid."""
        transform = (
            Affine.translation(self.x0, self.y0)
            * Affine.rotation(self.rotation)
            * Affine.scale(self.dx, self.dy)
        )
        return transform

    @property
    def coordinates(self, x_dim="m", y_dim="n"):
        """Return the coordinates of the cell-centers the regular grid."""
        x_coords, y_coords = (
            self.transform
            * self.transform.translation(0.5, 0.5)
            * np.meshgrid(np.arange(self.mmax), np.arange(self.nmax))
        )
        coords = {
            "y": ((y_dim, x_dim), y_coords),
            "x": ((y_dim, x_dim), x_coords),
        }
        return coords

    def read(self):
        self.build()
        self.read_msk_file()
        self.read_dep_file()

    def read_msk_file(self):    
        # Read mask file
        self.ds["mask"] = self.read_map("mask", os.path.join(self.model.path, self.model.input.variables.mskfile), np.int8, 0)

    def read_dep_file(self):    
        # Read depth file
        if self.model.input.variables.depfile:
            self.ds["bed_level"] = self.read_map("bed_level",
                                                 os.path.join(self.model.path, self.model.input.variables.depfile),
                                                 np.float32,
                                                 0.0)

    def read_map(self, name, file_name, dtype, fill_value):
        """Read one of the grid variables of the SFINCS model map from a binary file."""
        data = np.fromfile(file_name, dtype=dtype)
        data = np.reshape(data, (self.mmax, self.nmax)).transpose()
        da = xr.DataArray(
            name=name,
            data=data,
            coords=self.coordinates,
            dims=("n", "m"),
            attrs={"_FillValue": dtype(fill_value)},
        )
        return da

    def write(self):
        # Write out both mask and depth files
        self.write_msk_file()
        self.write_dep_file()

    def write_dep_file(self):    
        # Write depth file
        if self.model.input.variables.depfile:
            self.write_map("bed_level",
                           os.path.join(self.model.path, self.model.input.variables.depfile),
                           np.float32)

    def write_msk_file(self):    
        # Write depth file
        if self.model.input.variables.mskfile:
            self.write_map("mask",
                           os.path.join(self.model.path, self.model.input.variables.mskfile),
                           np.int8)

    def write_map(self, name, file_name, dtype):
        array = self.ds[name].values[:]
        zv = np.reshape(array, np.size(array), order='F')
        file = open(file_name, "wb")
        file.write(zv.astype(dtype))        
        file.close()

    def get_bathymetry(self, bathymetry_list):
        z = bathymetry_database.get_bathymetry_on_grid(self.ds["x"].values[:],
                                                       self.ds["y"].values[:],
                                                       self.model.crs,
                                                       bathymetry_list)
        da = xr.DataArray(
            data=z,
            coords=self.coordinates,
            dims=("n", "m"),
            attrs={"_FillValue": -99999.0},
        )
        self.ds["bed_level"] = da

    def build_mask(self,
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

        if include_polygon is None:
            include_polygon = gpd.GeoDataFrame()
        if exclude_polygon is None:
            exclude_polygon = gpd.GeoDataFrame()
        if boundary_polygon is None:
            boundary_polygon = gpd.GeoDataFrame()

        if not quiet:
            print("Building mask mask ...")

        xz = self.ds["x"].values[:]
        yz = self.ds["y"].values[:]
        zz = self.ds["bed_level"].values[:]
        mask = np.zeros((self.nmax, self.mmax), dtype=int)

        if zmin<zmax:
            # Set initial mask based on zmin and zmax
            iok = np.where((zz>=zmin) & (zz<=zmax))
            mask[iok] = 1
                        
        # Include polygons
#        if include_polygon is not None:           
        if len(include_polygon) > 0:
            for ip, polygon in include_polygon.iterrows():
                inpol = inpolygon(xz, yz, polygon["geometry"])
                iok   = np.where((inpol) & (zz>=include_zmin) & (zz<=include_zmax))
                mask[iok] = 1

        # Exclude polygons
#        if exclude_polygon is not None:
        if len(exclude_polygon) > 0:
            for ip, polygon in exclude_polygon.iterrows():
                inpol = inpolygon(xz, yz, polygon["geometry"])
                iok   = np.where((inpol) & (zz>=exclude_zmin) & (zz<=exclude_zmax))
                mask[iok] = 0

        # Open boundary polygons
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

        self.ds["mask"].values = mask                


    def mask_to_gdf(self, option="all"):
        xz = self.ds["x"].values[:]
        yz = self.ds["y"].values[:]
        mask = self.ds["mask"].values[:]
        gdf_list = []
        okay = np.zeros(mask.shape, dtype=int)
        if option == "all":
            iok = np.where((mask > 0))
        elif option == "include":
            iok = np.where((mask == 1))
        elif option == "boundary":
            iok = np.where((mask == 2))
        else:
            iok = np.where((mask > -999))
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

    # def build(self):
    #     self.x0 = self.model.input.variables.x0
    #     self.y0 = self.model.input.variables.y0
    #     self.dx = self.model.input.variables.dx
    #     self.dy = self.model.input.variables.dy
    #     self.nmax = self.model.input.variables.nmax
    #     self.mmax = self.model.input.variables.mmax
    #     self.rotation = self.model.input.variables.rotation

    #     cosrot = np.cos(self.rotation * np.pi / 180)
    #     sinrot = np.sin(self.rotation * np.pi / 180)

    #     # Corners
    #     xx = np.linspace(0.0,
    #                      self.mmax * self.dx,
    #                      num=self.mmax + 1)
    #     yy = np.linspace(0.0,
    #                      self.nmax * self.dy,
    #                      num=self.nmax + 1)
    #     xg0, yg0 = np.meshgrid(xx, yy)
    #     self.xg = self.x0 + xg0 * cosrot - yg0 * sinrot
    #     self.yg = self.y0 + xg0 * sinrot + yg0 * cosrot

    #     xx = np.linspace(0.5 * self.dx,
    #                      self.mmax * self.dx - 0.5 * self.dx,
    #                      num=self.mmax)
    #     yy = np.linspace(0.5 * self.dy,
    #                      self.nmax * self.dy - 0.5 * self.dy,
    #                      num=self.nmax)
    #     xg0, yg0 = np.meshgrid(xx, yy)
    #     self.xz = self.x0 + xg0 * cosrot - yg0 * sinrot
    #     self.yz = self.y0 + xg0 * sinrot + yg0 * cosrot

    def to_gdf(self):
        if self.nmax == 0:
            return gpd.GeoDataFrame()
        lines = []
        cosrot = math.cos(self.rotation*math.pi/180)
        sinrot = math.sin(self.rotation*math.pi/180)
        for n in range(self.nmax):
            for m in range(self.mmax):
                xa = self.x0 + m*self.dx*cosrot - n*self.dy*sinrot
                ya = self.y0 + m*self.dx*sinrot + n*self.dy*cosrot
                xb = self.x0 + (m + 1)*self.dx*cosrot - n*self.dy*sinrot
                yb = self.y0 + (m + 1)*self.dx*sinrot + n*self.dy*cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
                xb = self.x0 + m*self.dx*cosrot - (n + 1)*self.dy*sinrot
                yb = self.y0 + m*self.dx*sinrot + (n + 1)*self.dy*cosrot
                line = shapely.geometry.LineString([[xa, ya], [xb, yb]])
                lines.append(line)
        geom = shapely.geometry.MultiLineString(lines)
        gdf = gpd.GeoDataFrame(crs=self.model.crs, geometry=[geom])
        return gdf

    def build_xugrid(self):
        tic = time.perf_counter()
        print("Building XuGrid ...")
        x0 = self.x0
        y0 = self.y0
        nmax = self.nmax
        mmax = self.mmax
        dx = self.dx
        dy = self.dy
        rotation = self.rotation
        nr_cells = nmax * mmax
        cosrot = np.cos(rotation*np.pi/180)
        sinrot = np.sin(rotation*np.pi/180)
        nm_nodes   = np.full(4*nr_cells, 1e9, dtype=int)
        face_nodes = np.full((4, nr_cells), -1, dtype=int)
        node_x     = np.full(4*nr_cells, 1e9, dtype=float)
        node_y     = np.full(4*nr_cells, 1e9, dtype=float)
        nnodes     = 0
        icel       = 0
        for m in range(mmax):
            for n in range(nmax):
                ## Lower left
                nmind = m*(nmax + 1) + n
                nm_nodes[nnodes]    = nmind
                face_nodes[0, icel] = nnodes
                node_x[nnodes]      = x0 + cosrot*(m*dx) - sinrot*(n*dy)
                node_y[nnodes]      = y0 + sinrot*(m*dx) + cosrot*(n*dy)
                nnodes              += 1
                ## Lower right
                nmind = (m + 1)*(nmax + 1) + n
                nm_nodes[nnodes]    = nmind
                face_nodes[1, icel] = nnodes
                node_x[nnodes]      = x0 + cosrot*((m + 1)*dx) - sinrot*(n*dy)
                node_y[nnodes]      = y0 + sinrot*((m + 1)*dx) + cosrot*(n*dy)
                nnodes              += 1
                ## Upper right
                nmind = (m + 1)*(nmax + 1) + (n + 1)
                nm_nodes[nnodes]    = nmind
                face_nodes[2, icel] = nnodes
                node_x[nnodes]      = x0 + cosrot*((m + 1)*dx) - sinrot*((n + 1)*dy)
                node_y[nnodes]      = y0 + sinrot*((m + 1)*dx) + cosrot*((n + 1)*dy)
                nnodes              += 1
                ## Upper left
                nmind = m*(nmax + 1) + (n + 1)
                nm_nodes[nnodes]    = nmind
                face_nodes[3, icel] = nnodes
                node_x[nnodes]      = x0 + cosrot*(m*dx) - sinrot*((n + 1)*dy)
                node_y[nnodes]      = y0 + sinrot*(m*dx) + cosrot*((n + 1)*dy)
                nnodes              += 1
                # Next cell
                icel += 1

        xxx, indx, irev = np.unique(nm_nodes, return_index=True, return_inverse=True)
        node_x = node_x[indx]
        node_y = node_y[indx]
        transformer = Transformer.from_crs(self.model.crs,
                                        3857,
                                        always_xy=True)
        node_x_1, node_y_1 = transformer.transform(node_x, node_y)
        # if self.model.crs == 4326 and node_x_0 < -180.0, subtract 20037508.34 from node_x
        if self.model.crs.is_geographic:
            node_x_1[np.where(node_x < -180.0)] -= 2 * 20037508.34
            node_x_1[np.where(node_x > 180.0)] += 2 * 20037508.34
        for icel in range(nr_cells):
            for j in range(4):
                face_nodes[j, icel] = irev[face_nodes[j, icel]]
        nodes = np.transpose(np.vstack((node_x_1, node_y_1)))
        faces = np.transpose(face_nodes)
        fill_value = -1
        self.xugrid = xu.Ugrid2d(nodes[:, 0], nodes[:, 1], fill_value, faces)

        toc = time.perf_counter()
        print(f"Done in {toc - tic:0.4f} seconds")

        # Create a dataframe with line elements
        x1 = self.xugrid.edge_node_coordinates[:,0,0]
        x2 = self.xugrid.edge_node_coordinates[:,1,0]
        y1 = self.xugrid.edge_node_coordinates[:,0,1]
        y2 = self.xugrid.edge_node_coordinates[:,1,1]
        self.df = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))



    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):
        if self.nmax == 0:
            return False
        if self.xugrid == None:
            self.build_xugrid()
        transformer = Transformer.from_crs(4326,
                                    3857,
                                    always_xy=True)
        xl0, yl0 = transformer.transform(xlim[0], ylim[0])
        xl1, yl1 = transformer.transform(xlim[1], ylim[1])
        # If xlim crosses the dateline, shift the x coordinates
        if xlim[0] < -180.0:
            xl0 -= 2 * 20037508.34
        if xlim[1] < -180.0:
            xl1 -= 2 * 20037508.34
        if xlim[0] > 180.0:
            xl0 += 2 * 20037508.34            
        if xlim[1] > 180.0:
            xl1 += 2 * 20037508.34            
        xlim = [xl0, xl1]
        ylim = [yl0, yl1]
        ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
        # print("Making datashader overlay ...")
        height = int(width * ratio)
        cvs = ds.Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)
        agg = cvs.line(self.df, x=['x1', 'x2'], y=['y1', 'y2'], axis=1)
        img = tf.shade(agg)
        path = os.path.dirname(file_name)
        name = os.path.basename(file_name)
        name = os.path.splitext(name)[0]
        export_image(img, name, export_path=path)
        return True
        # toc = time.perf_counter()
        # print(f"Done in {toc - tic:0.4f} seconds")

    def outline(self):
        # Return gdf of grid outlines

        xg = self.model.grid.coordinates["x"][1]
        yg = self.model.grid.coordinates["y"][1]
        
        # if crs:
        #     transformer = Transformer.from_crs(self.crs,
        #                                        crs,
        #                                        always_xy=True)
        #     xg, yg = transformer.transform(xg, yg)
        
        xp = list([ xg[0,0], xg[0,-1], xg[-1,-1], xg[-1,0], xg[0,0] ])
        yp = list([ yg[0,0], yg[0,-1], yg[-1,-1], yg[-1,0], yg[0,0] ])
        points = []
        for ip, point in enumerate(xp):
            points.append(shapely.geometry.Point(xp[ip], yp[ip]))
        geom = shapely.geometry.Polygon(points)
        gdf = gpd.GeoDataFrame(crs=self.model.crs, geometry=[geom])

        return gdf


def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)
