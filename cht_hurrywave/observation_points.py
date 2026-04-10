"""
HurryWave observation point management.

Provides :class:`HurryWaveObservationPointsRegular` for standard time-series
output locations and :class:`HurryWaveObservationPointsSpectra` for 2-D
spectral output locations.  Both classes store points in a GeoDataFrame and
handle reading / writing the corresponding ASCII point files.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from matplotlib import path


class HurryWaveObservationPointsRegular:
    """
    Manage regular (time-series) observation points.

    Parameters
    ----------
    hw : HurryWave
        Parent model instance.
    """

    def __init__(self, hw) -> None:
        self.model = hw
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """
        Read observation points from the file referenced by ``obsfile``.

        Does nothing when ``obsfile`` is not set.
        """
        if not self.model.input.variables.obsfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)

        df = pd.read_csv(
            file_name,
            index_col=False,
            header=None,
            sep=r"\s+",
            names=["x", "y", "name"],
        )

        gdf_list = []
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "long_name": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write(self) -> None:
        """
        Write observation points to the file referenced by ``obsfile``.

        Does nothing when ``obsfile`` is not set or no points are stored.
        """
        if not self.model.input.variables.obsfile:
            print("No name for obs file !")
            return
        if len(self.gdf.index) == 0:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.obsfile)

        if self.model.crs.is_geographic:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    fid.write(f'{x:12.6f}{y:12.6f}  "{name}"\n')
        else:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    fid.write(f'{x:12.1f}{y:12.1f}  "{name}"\n')

    def add_point(self, x: float, y: float, name: str) -> None:
        """
        Add a single observation point.

        Parameters
        ----------
        x : float
            X-coordinate of the new point.
        y : float
            Y-coordinate of the new point.
        name : str
            Station name.
        """
        point = shapely.geometry.Point(x, y)
        gdf_list = [{"name": name, "long_name": None, "geometry": point}]
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index) -> None:
        """
        Delete an observation point by name or integer index.

        Parameters
        ----------
        name_or_index : str or int
            Station name or positional index to remove.
        """
        if isinstance(name_or_index, str):
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print(f"Point {name} not found!")
        else:
            index = name_or_index
            self.gdf = self.gdf.drop(index).reset_index(drop=True)

    def clear(self) -> None:
        """Remove all observation points."""
        self.gdf = gpd.GeoDataFrame()

    def list_observation_points(self) -> list:
        """
        Return the names of all stored observation points.

        Returns
        -------
        list of str
        """
        return [row["name"] for _, row in self.gdf.iterrows()]

    def add_points(self, gdf, name: str = "name") -> None:
        """
        Add multiple points, keeping only those inside the model grid outline.

        Parameters
        ----------
        gdf : geopandas.GeoDataFrame
            Source GeoDataFrame with point geometries.
        name : str, optional
            Column in *gdf* that contains station names.  Default is
            ``"name"``.
        """
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
                d = {
                    "name": row[name],
                    "long_name": None,
                    "geometry": shapely.geometry.Point(x[index], y[index]),
                }
                gdf_list.append(d)
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)


class HurryWaveObservationPointsSpectra:
    """
    Manage spectral (2-D spectrum output) observation points.

    Parameters
    ----------
    hw : HurryWave
        Parent model instance.
    """

    def __init__(self, hw) -> None:
        self.model = hw
        self.gdf = gpd.GeoDataFrame()

    def read(self) -> None:
        """
        Read spectral observation points from the file referenced by ``ospfile``.

        Does nothing when ``ospfile`` is not set.
        """
        if not self.model.input.variables.ospfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.ospfile)

        df = pd.read_csv(
            file_name,
            index_col=False,
            header=None,
            sep=r"\s+",
            names=["x", "y", "name"],
        )

        gdf_list = []
        for ind in range(len(df.x.values)):
            name = df.name.values[ind]
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "long_name": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write(self) -> None:
        """
        Write spectral observation points to the file referenced by ``ospfile``.

        Does nothing when ``ospfile`` is not set or no points are stored.
        """
        if not self.model.input.variables.ospfile:
            return
        if len(self.gdf.index) == 0:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.ospfile)

        if self.model.crs.is_geographic:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    fid.write(f'{x:12.6f}{y:12.6f}  "{name}"\n')
        else:
            with open(file_name, "w") as fid:
                for index, row in self.gdf.iterrows():
                    x = row["geometry"].coords[0][0]
                    y = row["geometry"].coords[0][1]
                    name = row["name"]
                    fid.write(f'{x:12.1f}{y:12.1f}  "{name}"\n')

    def add_point(self, x: float, y: float, name: str) -> None:
        """
        Add a single spectral observation point.

        Parameters
        ----------
        x : float
            X-coordinate of the new point.
        y : float
            Y-coordinate of the new point.
        name : str
            Station name.
        """
        point = shapely.geometry.Point(x, y)
        gdf_list = [{"name": name, "long_name": None, "geometry": point}]
        gdf_new = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)
        self.gdf = pd.concat([self.gdf, gdf_new], ignore_index=True)

    def delete_point(self, name_or_index) -> None:
        """
        Delete a spectral observation point by name or integer index.

        Parameters
        ----------
        name_or_index : str or int
            Station name or positional index to remove.
        """
        if isinstance(name_or_index, str):
            name = name_or_index
            for index, row in self.gdf.iterrows():
                if row["name"] == name:
                    self.gdf = self.gdf.drop(index).reset_index(drop=True)
                    return
            print(f"Point {name} not found!")
        else:
            index = name_or_index
            self.gdf = self.gdf.drop(index).reset_index(drop=True)

    def clear(self) -> None:
        """Remove all spectral observation points."""
        self.gdf = gpd.GeoDataFrame()

    def list_observation_points(self) -> list:
        """
        Return the names of all stored spectral observation points.

        Returns
        -------
        list of str
        """
        return [row["name"] for _, row in self.gdf.iterrows()]


def inpolygon(xq: np.ndarray, yq: np.ndarray, p) -> np.ndarray:
    """
    Test whether query points lie inside a polygon.

    Parameters
    ----------
    xq : numpy.ndarray
        X-coordinates of query points.
    yq : numpy.ndarray
        Y-coordinates of query points.
    p : shapely.geometry.Polygon
        Test polygon.

    Returns
    -------
    numpy.ndarray of bool
        Boolean array with the same shape as *xq*.
    """
    shape = xq.shape
    xq = xq.reshape(-1)
    yq = yq.reshape(-1)
    q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
    p = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
    return p.contains_points(q).reshape(shape)
