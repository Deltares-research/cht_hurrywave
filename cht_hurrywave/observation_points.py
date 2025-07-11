# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 09:03:08 2022
@author: ormondt
"""
import os
import geopandas as gpd
import shapely
import pandas as pd
import numpy as np
from matplotlib import path

class HurryWaveObservationPointsRegular:

    """
    A class to manage observation points for the HurryWave model, specifically for regular observation points.
    This class provides methods to read, write, add, delete, and list observation points, as well as clearing
    or updating the list of points.

    Key Methods:

    - __init__:
        Initializes the observation points container (GeoDataFrame).
    - read: 
        Reads observation points from a specified file and stores them as geometry points in a GeoDataFrame.
    - write:
        Writes the stored observation points to a specified file in a specific format (based on the model's CRS).
    - add_point:
         Adds a new observation point to the list.
    - delete_point:
         Deletes an observation point based on name or index.
    - clear:
         Clears all observation points.
    - list_observation_points:
         Lists the names of all current observation points.
    - add_points: 
        Adds multiple points to the observation list, checking if they are within the grid outline.

    Arguments:

    - hw: The HurryWave model object to which the observation points belong.

    """
    def __init__(self, hw):
        self.model = hw
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in all observation points

        if not self.model.input.variables.obsfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)

        # Read the bnd file
        df = pd.read_csv(file_name, index_col=False, header=None,
             sep="\s+", names=['x', 'y', 'name'])

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "long_name": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write(self):

        if not self.model.input.variables.obsfile:
            print("No name for obs file !")
            return
        if len(self.gdf.index)==0:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)
        
        if self.model.crs.is_geographic:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
                fid.write(string)
            fid.close()
        else:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                fid.write(string)
            fid.close()

    def add_point(self, x, y, name):
        point = shapely.geometry.Point(x, y)
        gdf_list = []
        d = {"name": name, "long_name": None, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index):
        if type(name_or_index) == str:
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print("Point " + name + " not found!")
        else:
            index = name_or_index
            self.gdf = self.gdf.drop(index).reset_index(drop=True)

    def clear(self):
        self.gdf  = gpd.GeoDataFrame()

    def list_observation_points(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names

    def add_points(self, gdf, name="name"):
        outline = self.model.grid.outline().loc[0]["geometry"]
        gdf = gdf.to_crs(self.model.crs)
        x = np.empty((len(gdf)))
        y = np.empty((len(gdf)))
        for index, row in gdf.iterrows():
            x[index] = row["geometry"].coords[0][0]
            y[index] = row["geometry"].coords[0][1]
        inpol = inpolygon(x, y, outline)
        gdf_list = []
        for index, row in gdf.iterrows():
            if inpol[index]:
                d = {"name": row[name], "long_name": None, "geometry": shapely.geometry.Point(x[index], y[index])}
                gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

class HurryWaveObservationPointsSpectra:

    """
    A class to manage observation points for the HurryWave model, specifically for spectra-based observation points.
    This class provides methods to read, write, add, delete, and list observation points, as well as clearing
    or updating the list of points.

    Key Methods:

    - __init__: 
        Initializes the observation points container (GeoDataFrame).
    - read:
         Reads observation points from a specified file and stores them as geometry points in a GeoDataFrame.
    - write:
         Writes the stored observation points to a specified file in a specific format (based on the model's CRS).
    - add_point:
         Adds a new observation point to the list.
    - delete_point:
         Deletes an observation point based on name or index.
    - clear:
         Clears all observation points.
    - list_observation_points: 
        Lists the names of all current observation points.

    Arguments:

    - hw: The HurryWave model object to which the observation points belong.
    """
    def __init__(self, hw):
        self.model = hw
        self.gdf  = gpd.GeoDataFrame()

    def read(self):
        # Read in observation points

        if not self.model.input.variables.ospfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.ospfile)

        # Read the bnd file
        df = pd.read_csv(file_name, index_col=False, header=None,
                         sep="\s+", names=['x', 'y', 'name'])

        gdf_list = []
        # Loop through points
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "long_name": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write(self):

        if not self.model.input.variables.ospfile:
            return
        if len(self.gdf.index)==0:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.ospfile)

        if self.model.crs.is_geographic:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.6f}{y:12.6f}  "{name}"\n'
                fid.write(string)
            fid.close()
        else:
            fid = open(file_name, "w")
            for index, row in self.gdf.iterrows():
                x = row["geometry"].coords[0][0]
                y = row["geometry"].coords[0][1]
                name = row["name"]
                string = f'{x:12.1f}{y:12.1f}  "{name}"\n'
                fid.write(string)
            fid.close()

    def add_point(self, x, y, name):
        point = shapely.geometry.Point(x, y)
        gdf_list = []
        d = {"name": name, "long_name": None, "geometry": point}
        gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index):
        if type(name_or_index) == str:
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print("Point " + name + " not found!")
        else:
            index = name_or_index
            self.gdf = self.gdf.drop(index).reset_index(drop=True)
        
    def clear(self):
        self.gdf  = gpd.GeoDataFrame()

    def list_observation_points(self):
        names = []
        for index, row in self.gdf.iterrows():
            names.append(row["name"])
        return names

def inpolygon(xq, yq, p):
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)
