"""
HurryWave main domain class.

Provides :class:`HurryWave`, the top-level object that owns the model grid,
mask, boundary conditions, observation points, and wave-blocking tables, and
exposes methods to read/write the model, query outputs, and generate index
tiles.
"""

import datetime
import math
import os

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer

from .boundary_conditions import HurryWaveBoundaryConditions

# Bathymetry and mask are now part of HurryWaveGrid class
from .grid import HurryWaveGrid
from .input import HurryWaveInput
from .mask import HurryWaveMask
from .observation_points import (
    HurryWaveObservationPointsRegular,
    HurryWaveObservationPointsSpectra,
)
from .waveblocking import WaveBlocking


class HurryWave:
    """
    Top-level HurryWave model object.

    Aggregates the model's input parameters, grid, mask, boundary conditions,
    observation points, and wave-blocking tables.  Can be initialised empty,
    pointed at an existing directory, or loaded directly from an existing
    ``hurrywave.inp`` file.

    Parameters
    ----------
    load : bool, optional
        When ``True`` the input file and all attribute files are read
        immediately.  Default is ``False``.
    crs : pyproj.CRS or None, optional
        Coordinate reference system.  Defaults to WGS 84 (EPSG:4326).
    path : str or None, optional
        Model directory.  Defaults to the current working directory.
    exe_path : str or None, optional
        Path to the HurryWave executable directory.
    read_grid_data : bool, optional
        When ``True`` (default) the grid and mask binary files are read in
        addition to the input file during a ``load``.
    """

    def __init__(
        self,
        load: bool = False,
        crs=None,
        path: str | None = None,
        exe_path: str | None = None,
        read_grid_data: bool = True,
    ) -> None:
        if not crs:
            self.crs = CRS(4326)
        else:
            self.crs = crs

        if not path:
            self.path = os.getcwd()
        else:
            self.path = path

        self.exe_path = exe_path

        self.input = HurryWaveInput(self)

        if load:
            self.input.read()
            self.crs = CRS(self.input.variables.crs_name)

        self.grid = HurryWaveGrid(self)
        self.mask = HurryWaveMask(self)
        self.boundary_conditions = HurryWaveBoundaryConditions(self)
        self.observation_points_regular = HurryWaveObservationPointsRegular(self)
        self.observation_points_sp2 = HurryWaveObservationPointsSpectra(self)
        self.waveblocking = WaveBlocking(self)
        self.obstacle = []

        if load:
            self.read_attribute_files(read_grid_data=read_grid_data)

    def clear_spatial_attributes(self) -> None:
        """Reset grid, mask, boundary conditions, and observation points to empty state."""
        self.grid = HurryWaveGrid(self)
        self.mask = HurryWaveMask(self)
        self.boundary_conditions = HurryWaveBoundaryConditions(self)
        self.observation_points_regular = HurryWaveObservationPointsRegular(self)
        self.observation_points_sp2 = HurryWaveObservationPointsSpectra(self)

    def read(self, path: str | None = None, read_grid_data: bool = True) -> None:
        """
        Read the input file and all attribute files.

        Parameters
        ----------
        path : str or None, optional
            Override the model directory before reading.
        read_grid_data : bool, optional
            Whether to read grid and mask binary data.  Default is ``True``.
        """
        if path:
            self.path = path
        self.input.read()
        self.crs = CRS(self.input.variables.crs_name)
        self.read_attribute_files(read_grid_data=read_grid_data)

    def read_input_file(self) -> None:
        """Read only ``hurrywave.inp`` and update :attr:`crs`."""
        self.input.read()
        self.crs = CRS(self.input.variables.crs_name)

    def read_attribute_files(self, read_grid_data: bool = True) -> None:
        """
        Read grid, boundary conditions, and observation point files.

        Parameters
        ----------
        read_grid_data : bool, optional
            When ``True`` (default) the grid (and mask) binary data are read.
        """
        if read_grid_data:
            self.grid.read()
        self.boundary_conditions.read()
        self.observation_points_regular.read()
        self.observation_points_sp2.read()

    def write(self) -> None:
        """Write the input file and all attribute files to :attr:`path`."""
        self.input.write()
        self.write_attribute_files()

    def write_attribute_files(self) -> None:
        """Write grid, boundary conditions, and observation point files."""
        self.grid.write()
        self.boundary_conditions.write()
        self.observation_points_regular.write()
        self.observation_points_sp2.write()

    def write_batch_file(self) -> None:
        """Write a Windows batch file (``run.bat``) to launch the model."""
        bat_path = os.path.join(self.path, "run.bat")
        with open(bat_path, "w") as fid:
            fid.write("set HDF5_USE_FILE_LOCKING=FALSE\n")
            fid.write(f"{self.exe_path}\\hurrywave.exe")

    def set_path(self, path: str) -> None:
        """
        Set the model directory.

        Parameters
        ----------
        path : str
            New model directory.
        """
        self.path = path

    def list_observation_points_regular(self) -> list:
        """
        Return names of all regular observation points.

        Returns
        -------
        list of str
        """
        return self.observation_points_regular.list_observation_points()

    def list_observation_points_spectra(self) -> list:
        """
        Return names of all spectral observation points.

        Returns
        -------
        list of str
        """
        return self.observation_points_spectra.list_observation_points()

    # ------------------------------------------------------------------
    # Output readers
    # ------------------------------------------------------------------

    def read_timeseries_output(
        self,
        name_list: list | None = None,
        path: str | None = None,
        file_name: str | None = None,
        parameter: str = "hm0",
    ) -> pd.DataFrame:
        """
        Read time-series output from ``hurrywave_his.nc``.

        Parameters
        ----------
        name_list : list of str or None, optional
            Station names to extract.  When ``None`` all stations are returned.
        path : str or None, optional
            Directory containing the history file.  Defaults to :attr:`path`.
        file_name : str or None, optional
            History NetCDF filename.  Defaults to ``"hurrywave_his.nc"``.
        parameter : str, optional
            Output variable to extract (e.g. ``"hm0"``, ``"tp"``).
            Default is ``"hm0"``.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by time with one column per station.
        """
        if not path:
            path = self.path

        if not file_name:
            file_name = "hurrywave_his.nc"

        file_name = os.path.join(path, file_name)

        ddd = xr.open_dataset(file_name)
        stations = ddd.station_name.values
        all_stations = []
        for st in stations:
            all_stations.append(str(st.strip())[2:-1])

        times = ddd.point_hm0.coords["time"].values

        if not name_list:
            name_list = list(all_stations)

        df = pd.DataFrame(index=times, columns=name_list)

        data_tmp = ddd[f"point_{parameter}"]

        for station in name_list:
            for ist, st in enumerate(all_stations):
                if station == st:
                    data = data_tmp.isel(stations=ist).values
                    data[np.isnan(data)] = -999.0
                    df[st] = data
                    break

        ddd.close()

        return df

    def read_map_output_max(
        self,
        time_range: list | None = None,
        map_file: str | None = None,
        parameter: str = "hm0",
    ) -> np.ndarray:
        """
        Read maximum map output over a time range.

        Parameters
        ----------
        time_range : list of datetime, optional
            ``[t_start, t_end]`` window.  Defaults to the full output period.
        map_file : str or None, optional
            Path to the map NetCDF file.  Defaults to
            ``<path>/hurrywave_map.nc``.
        parameter : str, optional
            Variable name to extract.  Default is ``"hm0"``.

        Returns
        -------
        numpy.ndarray
            Maximum values over the requested time range.
        """
        if not map_file:
            map_file = os.path.join(self.path, "hurrywave_map.nc")

        dsin = xr.open_dataset(map_file)

        output_times = dsin.timemax.values
        if time_range is None:
            t0 = (
                pd.to_datetime(str(output_times[0]))
                .replace(tzinfo=None)
                .to_pydatetime()
            )
            t1 = (
                pd.to_datetime(str(output_times[-1]))
                .replace(tzinfo=None)
                .to_pydatetime()
            )
            time_range = [t0, t1]

        it0 = -1
        for it, time in enumerate(output_times):
            time = pd.to_datetime(str(time)).replace(tzinfo=None).to_pydatetime()
            if time >= time_range[0] and it0 < 0:
                it0 = it
            if time <= time_range[1]:
                it1 = it

        if "mesh2d_face_nodes" in dsin:
            zs_da = np.amax(dsin[parameter].values[it0:it1, :], axis=0)
        else:
            zs_da = np.amax(dsin[parameter].values[it0:it1, :, :], axis=0)
        dsin.close()

        return zs_da

    def read_hm0max(
        self,
        time_range: list | None = None,
        hm0max_file: str | None = None,
        parameter: str = "hm0max",
    ) -> np.ndarray | None:
        """
        Read maximum Hm0 values from a ``.dat`` or ``.nc`` file.

        Parameters
        ----------
        time_range : list of datetime, optional
            ``[t_start, t_end]`` window.  Defaults to the full output period.
        hm0max_file : str or None, optional
            Path to the hm0max file.  Defaults to ``<path>/hm0max.dat``.
        parameter : str, optional
            Variable name used when reading NetCDF output.  Default is
            ``"hm0max"``.

        Returns
        -------
        numpy.ndarray or None
            2-D array of maximum Hm0 values, or ``None`` for unsupported
            formats.
        """
        if not hm0max_file:
            hm0max_file = os.path.join(self.path, "hm0max.dat")

        fname, ext = os.path.splitext(hm0max_file)

        if ext == ".dat":
            ind_file = os.path.join(self.path, self.input.variables.indexfile)

            freqstr = f"{self.input.variables.dtmaxout}S"
            t00 = datetime.timedelta(
                seconds=self.input.variables.t0out + self.input.variables.dtmaxout
            )
            output_times = (
                pd.date_range(
                    start=self.input.variables.tstart + t00,
                    end=self.input.variables.tstop,
                    freq=freqstr,
                )
                .to_pydatetime()
                .tolist()
            )
            nt = len(output_times)

            if time_range is None:
                time_range = [
                    self.input.variables.tstart + t00,
                    self.input.variables.tstop,
                ]

            for it, time in enumerate(output_times):
                if time <= time_range[0]:
                    it0 = it
                if time <= time_range[1]:
                    it1 = it

            nmax = self.input.variables.nmax + 2
            mmax = self.input.variables.mmax + 2

            data_ind = np.fromfile(ind_file, dtype="i4")
            npoints = data_ind[0]
            data_ind = np.squeeze(data_ind[1:])

            data_zs = np.fromfile(hm0max_file, dtype="f4")
            data_zs = np.reshape(data_zs, [nt, npoints + 2])[it0 : it1 + 1, 1:-1]

            data_zs = np.amax(data_zs, axis=0)
            zs_da = np.full([nmax * mmax], np.nan)
            zs_da[data_ind - 1] = np.squeeze(data_zs)
            zs_da = np.where(zs_da == -999, np.nan, zs_da)
            zs_da = np.transpose(np.reshape(zs_da, [mmax, nmax]))[1:-1, 1:-1]

        elif ext == ".nc":
            dsin = xr.open_dataset(hm0max_file)

            output_times = dsin.timemax.values
            if time_range is None:
                t0 = (
                    pd.to_datetime(str(output_times[0]))
                    .replace(tzinfo=None)
                    .to_pydatetime()
                )
                t1 = (
                    pd.to_datetime(str(output_times[-1]))
                    .replace(tzinfo=None)
                    .to_pydatetime()
                )
                time_range = [t0, t1]

            it0 = -1
            for it, time in enumerate(output_times):
                time = pd.to_datetime(str(time)).replace(tzinfo=None).to_pydatetime()
                if time >= time_range[0] and it0 < 0:
                    it0 = it
                if time <= time_range[1]:
                    it1 = it

            if "mesh2d_face_nodes" in dsin:
                zs_da = np.amax(dsin[parameter].values[it0:it1, :], axis=0)
            else:
                zs_da = np.amax(dsin[parameter].values[it0:it1, :, :], axis=0)
            dsin.close()

        else:
            return None

        return zs_da

    # ------------------------------------------------------------------
    # Grid geometry helpers
    # ------------------------------------------------------------------

    def grid_coordinates(self, loc: str = "cor") -> tuple:
        """
        Compute regular grid corner or centre coordinates.

        Parameters
        ----------
        loc : str, optional
            ``"cor"`` (default) returns corner coordinates; any other value
            returns cell-centre coordinates.

        Returns
        -------
        xg : numpy.ndarray
            2-D array of x-coordinates.
        yg : numpy.ndarray
            2-D array of y-coordinates.
        """
        cosrot = math.cos(self.input.variables.rotation * math.pi / 180)
        sinrot = math.sin(self.input.variables.rotation * math.pi / 180)
        if loc == "cor":
            xx = np.linspace(
                0.0,
                self.input.variables.mmax * self.input.variables.dx,
                num=self.input.variables.mmax + 1,
            )
            yy = np.linspace(
                0.0,
                self.input.variables.nmax * self.input.variables.dy,
                num=self.input.variables.nmax + 1,
            )
        else:
            xx = np.linspace(
                0.5 * self.input.variables.dx,
                self.input.variables.mmax * self.input.variables.dx
                - 0.5 * self.input.variables.dx,
                num=self.input.variables.mmax,
            )
            yy = np.linspace(
                0.5 * self.input.variables.dy,
                self.input.variables.nmax * self.input.variables.dy
                - 0.5 * self.input.variables.dy,
                num=self.input.variables.nmax,
            )

        xg0, yg0 = np.meshgrid(xx, yy)
        xg = self.input.variables.x0 + xg0 * cosrot - yg0 * sinrot
        yg = self.input.variables.y0 + xg0 * sinrot + yg0 * cosrot

        return xg, yg

    def bounding_box(self, crs=None) -> tuple:
        """
        Return the spatial bounding box of the model grid.

        Parameters
        ----------
        crs : pyproj.CRS or None, optional
            Target CRS for the returned coordinates.  When ``None`` the model
            CRS is used.

        Returns
        -------
        x_range : list of float
            ``[x_min, x_max]``
        y_range : list of float
            ``[y_min, y_max]``
        """
        xg, yg = self.grid_coordinates(loc="cor")

        if crs:
            transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
            xg, yg = transformer.transform(xg, yg)

        x_range = [np.min(np.min(xg)), np.max(np.max(xg))]
        y_range = [np.min(np.min(yg)), np.max(np.max(yg))]

        return x_range, y_range

    def outline(self, crs=None) -> tuple:
        """
        Return the closed polygon outline of the model domain.

        Parameters
        ----------
        crs : pyproj.CRS or None, optional
            Target CRS.  Defaults to the model CRS.

        Returns
        -------
        xp : list of float
            X-coordinates of the outline (5 points, first == last).
        yp : list of float
            Y-coordinates of the outline.
        """
        xg, yg = self.grid_coordinates(loc="cor")

        if crs:
            transformer = Transformer.from_crs(self.crs, crs, always_xy=True)
            xg, yg = transformer.transform(xg, yg)

        xp = [xg[0, 0], xg[0, -1], xg[-1, -1], xg[-1, 0], xg[0, 0]]
        yp = [yg[0, 0], yg[0, -1], yg[-1, -1], yg[-1, 0], yg[0, 0]]

        return xp, yp

    def make_index_tiles(
        self,
        path: str,
        zoom_range: list | None = None,
        z_range: list | None = None,
        dem_names: list | None = None,
    ) -> None:
        """
        Generate and write binary index tiles for map tiling.

        Parameters
        ----------
        path : str
            Output directory for the tile tree.
        zoom_range : list of int, optional
            ``[min_zoom, max_zoom]``.  Defaults to ``[0, 13]``.
        z_range : list of float, optional
            Reserved for depth-range filtering (not yet re-implemented).
        dem_names : list of str, optional
            Reserved for depth-range filtering (not yet re-implemented).
        """
        import cht_utils.fileops as fo
        from cht_tiling.utils import deg2num, num2deg

        if not zoom_range:
            zoom_range = [0, 13]

        npix = 256

        lon_range, lat_range = self.bounding_box(crs=CRS.from_epsg(4326))

        cosrot = math.cos(-self.input.variables.rotation * math.pi / 180)
        sinrot = math.sin(-self.input.variables.rotation * math.pi / 180)

        transformer_a = Transformer.from_crs(
            CRS.from_epsg(4326), CRS.from_epsg(3857), always_xy=True
        )
        transformer_b = Transformer.from_crs(
            CRS.from_epsg(3857), self.crs, always_xy=True
        )

        if os.path.exists(path):
            print(f"Removing existing path {path}")
            fo.rmdir(path)

        if z_range:
            # TODO: z_range filtering needs reimplementation using hydromt data catalog
            pass

        for izoom in range(zoom_range[0], zoom_range[1] + 1):
            print(f"Processing zoom level {izoom}")

            zoom_path = os.path.join(path, str(izoom))

            dxy = (40075016.686 / npix) / 2**izoom
            xx = np.linspace(0.0, (npix - 1) * dxy, num=npix)
            yy = xx[:]
            xv, yv = np.meshgrid(xx, yy)

            ix0, iy0 = deg2num(lat_range[0], lon_range[0], izoom)
            ix1, iy1 = deg2num(lat_range[1], lon_range[1], izoom)

            for i in range(ix0, ix1 + 1):
                path_okay = False
                zoom_path_i = os.path.join(zoom_path, str(i))

                for j in range(iy0, iy1 + 1):
                    file_name = os.path.join(zoom_path_i, f"{j}.dat")

                    lat, lon = num2deg(i, j, izoom)

                    xo, yo = transformer_a.transform(lon, lat)

                    xm = xv[:] + xo + 0.5 * dxy
                    ym = yv[:] + yo + 0.5 * dxy

                    x, y = transformer_b.transform(xm, ym)

                    x00 = x - self.input.variables.x0
                    y00 = y - self.input.variables.y0
                    xg = x00 * cosrot - y00 * sinrot
                    yg = x00 * sinrot + y00 * cosrot

                    iind = np.floor(xg / self.input.variables.dx).astype(int)
                    jind = np.floor(yg / self.input.variables.dy).astype(int)
                    ind = iind * self.input.variables.nmax + jind

                    ind[iind < 0] = -999
                    ind[jind < 0] = -999
                    ind[iind >= self.input.variables.mmax] = -999
                    ind[jind >= self.input.variables.nmax] = -999

                    if ind.max() >= 0:
                        ingrid = np.where(ind >= 0)
                        msk = np.zeros((256, 256), dtype=int) + 1
                        msk[ingrid] = self.grid.ds["mask"].values[
                            jind[ingrid], iind[ingrid]
                        ]
                        iex = np.where(msk < 1)
                        ind[iex] = -999

                    if np.any(ind >= 0):
                        if not path_okay:
                            if not os.path.exists(zoom_path_i):
                                fo.mkdir(zoom_path_i)
                                path_okay = True
                        with open(file_name, "wb") as fid:
                            fid.write(ind)

    def setup_wind_uniform_forcing(
        self,
        timeseries=None,
        magnitude: float | None = None,
        direction: float | None = None,
    ) -> None:
        """
        Set up spatially uniform wind forcing.

        Parameters
        ----------
        timeseries : str or None, optional
            Path to a CSV file with columns ``[time, magnitude, direction]``.
        magnitude : float or None, optional
            Constant wind speed [m/s].  Used together with *direction*.
        direction : float or None, optional
            Constant wind direction [deg from north].  Used together with
            *magnitude*.

        Raises
        ------
        ValueError
            If *timeseries* is not a string path, or if neither *timeseries*
            nor both *magnitude* and *direction* are supplied.
        """
        tstart, tstop = self.input.variables.tstart, self.input.variables.tstop
        if timeseries is not None:
            if isinstance(timeseries, str):
                df_ts = pd.read_csv(timeseries, index=[0], header=None)
            else:
                raise ValueError("Timeseries should be path to csv file")

        elif magnitude is not None and direction is not None:
            df_ts = pd.DataFrame(
                index=pd.date_range(tstart, tstop, periods=2),
                data=np.array([[magnitude, direction], [magnitude, direction]]),
                columns=["mag", "dir"],
            )
        else:
            raise ValueError(
                "Either timeseries or magnitude and direction must be provided"
            )

        df_ts.name = "wnd"
        df_ts.index.name = "time"
        df_ts.columns.name = "index"

        df_ts.index = df_ts.index - self.input.variables.tref
        df_ts.index = df_ts.index.total_seconds()

        df_ts.to_csv(os.path.join(self.path, "hurrywave.wnd"), sep=" ", header=False)


def read_timeseries_file(file_name: str, ref_date) -> pd.DataFrame:
    """
    Read a whitespace-delimited time-series file.

    The first column contains time offsets in seconds relative to
    *ref_date*.

    Parameters
    ----------
    file_name : str
        Path to the time-series file.
    ref_date : datetime.datetime
        Reference date used to convert the time offset column.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by absolute datetime.
    """
    df = pd.read_csv(file_name, index_col=0, header=None, sep=r"\s+")
    ts = ref_date + pd.to_timedelta(df.index, unit="s")
    df.index = ts

    return df
