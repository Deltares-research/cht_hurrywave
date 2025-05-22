"""
HurryWave Boundary Conditions Module
====================================

This module provides functionality for handling boundary conditions
in the HurryWave wave model. It includes methods for reading, writing,
and manipulating boundary points and associated time series or spectral data.

Classes
-------
- HurryWaveBoundaryConditions: Handles boundary conditions for HurryWave.

Functions
---------
- read_timeseries_file: Reads a time series file and returns a DataFrame.
- to_fwf: Writes a DataFrame to a fixed-width formatted file.
"""

import os
import numpy as np
import geopandas as gpd
import shapely
import pandas as pd
from tabulate import tabulate
from pyproj import Transformer

class HurryWaveBoundaryConditions:
    """
    A class to manage boundary conditions for the HurryWave model.
    """
    
    def __init__(self, hw):
        """
        Initializes the boundary conditions handler.
        
        Parameters
        ----------
        hw : object
            The HurryWave model instance.
        """
        self.model = hw
        self.forcing = "timeseries"
        self.gdf = gpd.GeoDataFrame()
        self.times = []

    def read(self):
        """
        Reads all boundary data from files.
        """
        self.read_boundary_points()
        self.read_boundary_time_series()
        self.read_boundary_spectra()

    def write(self):
        """
        Writes all boundary data to files.
        """
        self.write_boundary_points()
        if self.forcing == "timeseries":
            self.write_boundary_conditions_timeseries()
        else:
            self.write_boundary_conditions_spectra()

    def read_boundary_points(self):
        """
        Reads boundary point coordinates from the HurryWave boundary file (.bnd).
        """
        if not self.model.input.variables.bndfile:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.bndfile)
        df = pd.read_csv(file_name, index_col=False, header=None, sep=r"\s+", names=['x', 'y'])

        gdf_list = []
        for ind in range(len(df.x.values)):
            name = str(ind + 1).zfill(4)
            x = df.x.values[ind]
            y = df.y.values[ind]
            point = shapely.geometry.Point(x, y)
            d = {"name": name, "timeseries": pd.DataFrame(), "spectra": None, "geometry": point}
            gdf_list.append(d)
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)

    def write_boundary_points(self):
        """
        Writes boundary point coordinates to the HurryWave boundary file (.bnd).
        """
        if len(self.gdf.index) == 0:
            return

        if not self.model.input.variables.bndfile:
            self.model.input.variables.bndfile = "hurrywave.bnd"

        file_name = os.path.join(self.model.path, self.model.input.variables.bndfile)

        with open(file_name, "w") as fid:
            for _, row in self.gdf.iterrows():
                x, y = row["geometry"].coords[0]
                if self.model.crs.is_geographic:
                    fid.write(f'{x:12.6f}{y:12.6f}\n')
                else:
                    fid.write(f'{x:12.1f}{y:12.1f}\n')
    
    def set_timeseries_uniform(self, hs, tp, wd, ds):
        """
        Sets uniform time series boundary conditions for all points.
        
        Parameters
        ----------
        hs : float
            Significant wave height.
        tp : float
            Peak wave period.
        wd : float
            Wave direction.
        ds : float
            Directional spreading.
        """
        time = [self.model.input.variables.tstart, self.model.input.variables.tstop]
        nt = len(time)
        hs, tp, wd, ds = [hs] * nt, [tp] * nt, [wd] * nt, [ds] * nt

        for index, point in self.gdf.iterrows():
            df = pd.DataFrame({"time": time, "hs": hs, "tp": tp, "wd": wd, "ds": ds})
            df.set_index("time", inplace=True)
            self.gdf.at[index, "timeseries"] = df


    def add_point(self, x, y, hs=1.0, tp=8.0, wd=0.0, ds=30.0):
        """
        Adds a new boundary point at the specified coordinates.
        
        Parameters
        ----------
        x : float
            X-coordinate of the new point.
        y : float
            Y-coordinate of the new point.
        """
        name = str(len(self.gdf.index) + 1).zfill(4)
        point = shapely.geometry.Point(x, y)
        # Create GDF for the new point
        gdf = gpd.GeoDataFrame(columns=["name", "timeseries", "spectra", "geometry"], crs=self.model.crs)
        gdf.at[0, "name"] = name
        gdf.at[0, "geometry"] = point
        if self.gdf.empty:
            df = pd.DataFrame({"time": [self.model.input.variables.tstart, self.model.input.variables.tstop],
                            "hs": [hs, hs], "tp": [tp, tp], "wd": [wd, wd], "ds": [ds, ds]}).set_index("time")
        else:
            # Copy from fist point in self.gdf
            df = self.gdf["timeseries"].iloc[0].copy()            
        gdf.at[0, "timeseries"] = df
        # Append the new point to the existing GeoDataFrame
        self.gdf = pd.concat([self.gdf, gdf], ignore_index=True)

    def delete_point(self, index):
        """
        Deletes a boundary point by index.
        
        Parameters
        ----------
        index : int
            Index of the point to be deleted.
        """
        if len(self.gdf.index) == 0 or index >= len(self.gdf.index):
            return
        
        self.gdf = self.gdf.drop(index).reset_index(drop=True)
        for idx, _ in self.gdf.iterrows():
            self.gdf.at[idx, "name"] = str(idx + 1).zfill(4)
        

    def clear(self):
        self.gdf  = gpd.GeoDataFrame()


    def read_boundary_time_series(self):
        """
        Reads boundary time series from HurryWave input files (bhs, btp, bwd, bds) and stores them in the model.
        
        The function retrieves time series for significant wave height (Hs), peak period (Tp), wave direction (Wd),
        and directional spreading (Ds) for each boundary point.
        """
        if not self.model.input.variables.bhsfile:
            return
        if len(self.gdf.index) == 0:
            return

        tref = self.model.input.variables.tref

        # Read Hs
        file_name = os.path.join(self.model.path, self.model.input.variables.bhsfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["time"] = dffile.index
            point["timeseries"]["hs"] = dffile.iloc[:, ip].values
            point["timeseries"].set_index("time", inplace=True)

        # Read Tp
        file_name = os.path.join(self.model.path, self.model.input.variables.btpfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["tp"] = dffile.iloc[:, ip].values

        # Read Wd
        file_name = os.path.join(self.model.path, self.model.input.variables.bwdfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["wd"] = dffile.iloc[:, ip].values

        # Read Ds
        file_name = os.path.join(self.model.path, self.model.input.variables.bdsfile)
        dffile = read_timeseries_file(file_name, tref)
        for ip, point in self.gdf.iterrows():
            point["timeseries"]["ds"] = dffile.iloc[:, ip].values

    def read_boundary_spectra(self):
        """
        Reads boundary spectra from the HurryWave bsp file.
        """
        if not self.model.input.variables.bspfile:
            return
        if len(self.gdf.index) == 0:
            return

        file_name = os.path.join(self.model.path, self.model.input.variables.bspfile)
        print("Reading " + file_name)

    def write_boundary_conditions_timeseries(self):
        """
        Writes boundary condition time series data to HurryWave input files.
        
        Outputs include significant wave height (Hs), peak period (Tp), wave direction (Wd),
        and directional spreading (Ds), which are saved in separate files.
        """
        if len(self.gdf.index) == 0:
            return

        time = self.gdf.loc[0]["timeseries"].index
        tref = self.model.input.variables.tref
        dt = (time - tref).total_seconds()

        def write_timeseries(var_name, file_ext):
            if not getattr(self.model.input.variables, var_name):
                setattr(self.model.input.variables, var_name, f"hurrywave.{file_ext}")
            file_name = os.path.join(self.model.path, getattr(self.model.input.variables, var_name))
            df = pd.DataFrame({ip: point["timeseries"][file_ext] for ip, point in self.gdf.iterrows()})
            df.index = dt
            to_fwf(df, file_name)

        write_timeseries("bhsfile", "hs")
        write_timeseries("btpfile", "tp")
        write_timeseries("bwdfile", "wd")
        write_timeseries("bdsfile", "ds")

    def write_boundary_conditions_spectra(self, file_name=None):
        """
        Writes boundary spectra data to a HurryWave bsp file.
        """
        import xarray as xr

        if file_name is None:
            if self.model.input.variables.bspfile is None:
                self.model.input.variables.bspfile = "hurrywave.bsp"
            file_name = os.path.join(self.model.path, self.model.input.variables.bspfile)

        sp20 = self.gdf["spectra"][0]
        times = sp20.coords["time"].values
        sigma = sp20.coords["sigma"].values
        theta = sp20.coords["theta"].values

        sp2 = np.zeros([len(times), len(self.gdf), len(theta), len(sigma)])
        xs, ys, points = np.zeros(len(self.gdf)), np.zeros(len(self.gdf)), []
        for ip, point in self.gdf.iterrows():
            points.append(point["name"])
            xs[ip], ys[ip] = point.geometry.x, point.geometry.y
            sp2[:, ip, :, :] = point["spectra"].values

        ds = xr.Dataset(
            data_vars=dict(point_spectrum2d=(["time", "stations", "theta", "sigma"], sp2),
                           station_x=(["stations"], np.single(xs)),
                           station_y=(["stations"], np.single(ys))),
            coords=dict(time=times, stations=points, theta=theta, sigma=sigma)
        )

        dstr = "seconds since " + self.model.input.variables.tref.strftime("%Y%m%d %H%M%S")
        ds.to_netcdf(path=file_name, mode='w', encoding={'time': {'units': dstr}})

    def get_boundary_points_from_mask(self, min_dist=None, bnd_dist=50000.0):
        """
        Identifies boundary points from a mask and interpolates them along boundary lines.

        This function extracts boundary points from the computational grid's mask where 
        the value is `2`. It then connects nearby points into polylines and interpolates 
        new points along these lines to ensure evenly spaced boundary points.

        Parameters
        ----------
        min_dist : float, optional
            The minimum distance between two connected boundary points. If not provided, 
            it defaults to twice the grid resolution (`2 * dx`).
        bnd_dist : float, optional, default=50000.0
            The target distance [m] for interpolated boundary points. The function will 
            generate points along boundary lines at approximately this interval (in [m]).


        Returns
        -------
        None
            Updates `self.gdf` with the identified and interpolated boundary points.
        """
        
        # Default min_dist to twice the grid spacing if not provided
        if min_dist is None:
            min_dist = self.model.grid.dx * 2

        # Extract mask data and identify boundary points (where mask == 2)
        da_mask = self.model.grid.ds["mask"]
        ibnd = np.where(da_mask.values == 2)  # Boundary indices
        xp, yp = da_mask["x"].values[ibnd], da_mask["y"].values[ibnd]  # Boundary coordinates

        # Make boolean array for points that are include in a polyline 
        used = np.full(xp.shape, False, dtype=bool)

        # Make list of polylines. Each polyline is a list of indices of boundary points.
        polylines = []

        while True:

            if np.all(used):
                # All boundary grid points have been used. We can stop now.
                break

            # Find first the unused points
            i1 = np.where(~used)[0][0]

            # Set this point to used
            used[i1] = True

            # Start new polyline with index i1
            polyline = [i1] 

            while True:
                # Compute distances to all points that have not been used
                xpunused = xp[~used]
                ypunused = yp[~used]
                # Get all indices of unused points
                unused_indices = np.where(~used)[0]

                dst = np.sqrt((xpunused - xp[i1])**2 + (ypunused - yp[i1])**2)
                if np.all(np.isnan(dst)):
                    break
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.append(inearall)
                    used[inearall] = True
                    i1 = inearall
                else:
                    # Last point found
                    break
            
            # Now work the other way
            # Start with first point of polyline
            i1 = polyline[0]
            while True:
                if np.all(used):
                    # All boundary grid points have been used. We can stop now.
                    break
                # Now we go in the other direction            
                xpunused = xp[~used]
                ypunused = yp[~used]
                unused_indices = np.where(~used)[0]
                dst = np.sqrt((xpunused - xp[i1])**2 + (ypunused - yp[i1])**2)
                inear = np.nanargmin(dst)
                inearall = unused_indices[inear]
                if dst[inear] < min_dist:
                    # Found next point along polyline
                    polyline.insert(0, inearall)
                    used[inearall] = True
                    # Set index of next point
                    i1 = inearall
                else:
                    # Last nearby point found
                    break

            if len(polyline) > 1:  
                polylines.append(polyline)

        # Transform to web mercator to get distance in metres
        if self.model.crs.is_geographic:
            transformer = Transformer.from_crs(self.model.crs,
                                            3857,
                                            always_xy=True)
        gdf_list = []
        ip = 0

        # Interpolate new points along each polyline
        for polyline in polylines:
            x = xp[polyline]
            y = yp[polyline]
            points = [(x,y) for x,y in zip(x.ravel(),y.ravel())]
            line = shapely.geometry.LineString(points)
            if self.model.crs.is_geographic:
                # Line in web mercator (to get length in metres)
                xm, ym = transformer.transform(x, y)
                pointsm = [(xm,ym) for xm,ym in zip(xm.ravel(),ym.ravel())]
                linem = shapely.geometry.LineString(pointsm)
                num_points = int(linem.length / bnd_dist) + 2
            else:
                num_points = int(line.length / bnd_dist) + 2

            # Generate evenly spaced points along the polyline
            new_points = [line.interpolate(i / float(num_points - 1), normalized=True) for i in range(num_points)]

            # Store points in GeoDataFrame format
            for point in new_points:
                gdf_list.append({
                    "name": str(ip + 1).zfill(4),
                    "timeseries": pd.DataFrame(),
                    "spectra": None,
                    "geometry": point
                })
                ip += 1  # Increment point counter

        # Store the final boundary points in a GeoDataFrame
        self.gdf = gpd.GeoDataFrame(gdf_list, crs=self.model.crs)



def read_timeseries_file(file_name, ref_date):
    """
    Reads a time series file and returns a DataFrame.
    
    Parameters
    ----------
    file_name : str
        Path to the time series file.
    ref_date : datetime
        Reference date for time indexing.
    
    Returns
    -------
    DataFrame
        DataFrame with time series indexed by time.
    """
    df = pd.read_csv(file_name, index_col=0, header=None, sep=r"\s+")
    df.index = ref_date + pd.to_timedelta(df.index, unit="s")
    return df


def to_fwf(df, fname, floatfmt=".3f"):
    """
    Writes a DataFrame to a fixed-width formatted file.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to write.
    fname : str
        Path to the output file.
    floatfmt : str, optional
        Floating point format (default is '.3f').
    """
    indx = df.index.tolist()
    vals = df.values.tolist()
    for it, t in enumerate(vals):
        t.insert(0, indx[it])
    content = tabulate(vals, [], tablefmt="plain", floatfmt=floatfmt)
    with open(fname, "w") as f:
        f.write(content)
