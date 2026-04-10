"""
Wave blocking coefficient computation.

Provides :class:`WaveBlocking`, which calculates directional blocking
coefficients from subgrid bathymetry for use in HurryWave.  Also provides
the internal helper classes :class:`Cell` and :class:`Cell2` that perform
the obstacle-projection geometry.
"""

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from cht_utils.interpolation import interp2


class WaveBlocking:
    """
    Compute and store wave blocking coefficients for a HurryWave grid.

    Parameters
    ----------
    model : HurryWave
        Parent model instance.
    version : int, optional
        Algorithm version flag.  Default is 0.
    """

    def __init__(self, model, version: int = 0) -> None:
        self.model = model
        self.version = version

    def read(self) -> None:
        """Load pre-computed blocking coefficients from the file named in ``wblfile``."""
        file_name = os.path.join(self.model.path, self.model.input.variables.wblfile)
        self.load(file_name)

    def build(
        self,
        bathymetry_sets,
        bathymetry_database=None,
        file_name: str = "hurrywave.wbl",
        nr_dirs: int = 36,
        nr_subgrid_pixels: int = 20,
        threshold_level: float = -5.0,
        quiet: bool = True,
        progress_bar=None,
        showcase: bool = False,
    ):
        """
        Compute wave blocking coefficients from subgrid bathymetry.

        Iterates over all grid levels and blocks, evaluates subgrid
        topography, and for each cell computes the fraction of the
        directional projection plane that is blocked by obstacles above
        *threshold_level*.

        Parameters
        ----------
        bathymetry_sets : list
            Ordered list of bathymetry datasets (tif or database entries).
        bathymetry_database : object or None, optional
            Bathymetry database object with a
            ``get_bathymetry_on_grid`` method.  When ``None`` tif-file
            datasets in *bathymetry_sets* are used instead.
        file_name : str, optional
            Output NetCDF filename.  Default is ``"hurrywave.wbl"``.
        nr_dirs : int, optional
            Number of directional bins.  Default is 36.
        nr_subgrid_pixels : int, optional
            Number of subgrid pixels per coarse cell side.  Default is 20.
        threshold_level : float, optional
            Elevation [m] above which pixels are treated as obstacles.
            Default is -5.0.
        quiet : bool, optional
            Suppress verbose output.  Default is ``True``.
        progress_bar : object or None, optional
            Optional progress-bar object with ``set_value``,
            ``set_maximum``, and ``was_canceled`` methods.
        showcase : bool, optional
            When ``True`` interactive elevation-map plots are shown for
            each processed cell.  Default is ``False``.

        Returns
        -------
        xarray.Dataset
            Dataset containing the ``blocking_coefficient`` variable saved
            to *file_name*.
        """
        refi = nr_subgrid_pixels
        self.nbins = nr_dirs
        nrmax = 2000

        grid = self.model.grid.data
        nr_cells = grid.nr_cells
        nr_ref_levs = grid.nr_refinement_levels
        x0 = grid.xuds.attrs["x0"]
        y0 = grid.xuds.attrs["y0"]
        nmax = grid.xuds.attrs["nmax"]
        mmax = grid.xuds.attrs["mmax"]
        dx = grid.xuds.attrs["dx"]
        dy = grid.xuds.attrs["dy"]
        rotation = grid.xuds.attrs["rotation"]
        cosrot = grid.cosrot
        sinrot = grid.sinrot
        level = grid.xuds["level"].values[:] - 1
        n = grid.xuds["n"].values[:] - 1
        m = grid.xuds["m"].values[:] - 1

        counter = 0

        shape = (nr_dirs, nr_cells)
        self.block_coefficient = np.zeros(shape, dtype=float)

        ifirst = np.zeros(nr_ref_levs, dtype=int)
        ilast = np.zeros(nr_ref_levs, dtype=int)
        nr_cells_per_level = np.zeros(nr_ref_levs, dtype=int)
        ireflast = -1
        for ic in range(nr_cells):
            if level[ic] > ireflast:
                ifirst[level[ic]] = ic
                ireflast = level[ic]
        for ilev in range(nr_ref_levs - 1):
            ilast[ilev] = ifirst[ilev + 1] - 1
        ilast[nr_ref_levs - 1] = nr_cells - 1
        for ilev in range(nr_ref_levs):
            nr_cells_per_level[ilev] = ilast[ilev] - ifirst[ilev] + 1

        for ilev in range(nr_ref_levs):
            cell_indices_in_level = np.arange(ifirst[ilev], ilast[ilev] + 1, dtype=int)
            nr_cells_in_level = np.size(cell_indices_in_level)

            if nr_cells_in_level == 0:
                continue

            n0 = np.min(n[ifirst[ilev] : ilast[ilev] + 1])
            n1 = np.max(n[ifirst[ilev] : ilast[ilev] + 1])
            m0 = np.min(m[ifirst[ilev] : ilast[ilev] + 1])
            m1 = np.max(m[ifirst[ilev] : ilast[ilev] + 1])

            dxi = dx / 2**ilev
            dyi = dy / 2**ilev
            dxp = dxi / refi
            dyp = dyi / refi

            nrcb = int(np.floor(nrmax / refi))
            nrbn = int(np.ceil((n1 - n0 + 1) / nrcb))
            nrbm = int(np.ceil((m1 - m0 + 1) / nrcb))

            if not quiet:
                print(f"Number of regular cells in a block : {nrcb}")
                print(f"Number of grid cells    : {grid.nmax}")
                print(f"Grid size of HW grid             : dx= {dx}, dy= {dy}")
                print(
                    f"Grid size of waveblocking pixels        : dx= {dxp}, dy= {dyp}"
                )
                print(f"Grid size of directional bins        : directions = {nr_dirs}")

            if progress_bar:
                progress_bar.set_text(
                    "               Computing wave blocking coefficients ...                "
                )
                progress_bar.set_minimum(0)
                progress_bar.set_maximum(nrbm * nrbn)
                progress_bar.set_value(0)

            ib = 0
            for ii in range(nrbm):
                for jj in range(nrbn):
                    if progress_bar:
                        progress_bar.set_value(ib)
                        if progress_bar.was_canceled():
                            return False

                    ib += 1

                    if not quiet:
                        print("--------------------------------------------------------------")
                        print(f"Processing block {ib} of {nrbn * nrbm} ...")
                        print("Getting bathymetry data ...")

                    bn0 = n0 + jj * nrcb
                    bn1 = min(bn0 + nrcb - 1, n1) + 1
                    bm0 = m0 + ii * nrcb
                    bm1 = min(bm0 + nrcb - 1, m1) + 1

                    x00 = 0.5 * dxp + bm0 * refi * dyp
                    x01 = x00 + (bm1 - bm0 + 1) * refi * dxp
                    y00 = 0.5 * dyp + bn0 * refi * dyp
                    y01 = y00 + (bn1 - bn0 + 1) * refi * dyp

                    x0v = np.arange(x00, x01, dxp)
                    y0v = np.arange(y00, y01, dyp)
                    xg0, yg0 = np.meshgrid(x0v, y0v)

                    xg = x0 + cosrot * xg0 - sinrot * yg0
                    yg = y0 + sinrot * xg0 + cosrot * yg0

                    del x0v, y0v, xg0, yg0

                    zg = np.full(np.shape(xg), np.nan)

                    if bathymetry_database is not None:
                        try:
                            zg = bathymetry_database.get_bathymetry_on_grid(
                                xg, yg, self.model.crs, bathymetry_sets
                            )
                        except Exception as e:
                            print(e)
                            return

                    else:
                        for ibathy, bathymetry in enumerate(bathymetry_sets):
                            if np.isnan(zg).any():
                                try:
                                    if bathymetry.attrs["type"] == "tif_file":
                                        xgb, ygb = xg, yg
                                        xb, yb, zb = (
                                            bathymetry.x,
                                            bathymetry.y,
                                            bathymetry[0].data,
                                        )
                                        if zb is not np.nan:
                                            if not np.isnan(zb).all():
                                                zg1 = interp2(xb, yb, zb, xgb, ygb)
                                                isn = np.where(np.isnan(zg))
                                                zg[isn] = zg1[isn]
                                        del xb, yb, zb
                                except Exception as e:
                                    print(e)
                                    return

                    if not quiet:
                        print("Computing blocking coefficients ...")

                    nvec = int(nr_dirs / 2)
                    dtheta = 360.0 / nr_dirs
                    angles = np.linspace(
                        0.5 * dtheta, 180.0 + 0.5 * dtheta, nvec, endpoint=False
                    )
                    radians = np.deg2rad(angles)
                    vectors = np.array(
                        [[np.cos(angle), np.sin(angle), 0] for angle in radians]
                    )

                    index_cells_in_block = np.zeros(nrcb * nrcb, dtype=int)
                    nr_cells_in_block = 0
                    for ic in range(nr_cells_in_level):
                        indx = cell_indices_in_level[ic]
                        if (
                            n[indx] >= bn0
                            and n[indx] < bn1
                            and m[indx] >= bm0
                            and m[indx] < bm1
                        ):
                            index_cells_in_block[nr_cells_in_block] = indx
                            nr_cells_in_block += 1

                    if nr_cells_in_block == 0:
                        continue

                    index_cells_in_block = index_cells_in_block[0:nr_cells_in_block]

                    for index in index_cells_in_block:
                        nn = (n[index] - bn0) * refi
                        mm = (m[index] - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]

                        if np.nanmax(zgc) < threshold_level:
                            continue
                        counter += 1

                        if showcase:
                            plt.figure()
                            plt.pcolor(zgc)
                            plt.axis("equal")
                            plt.title(f"Elevation map {m} x {n}")
                            plt.show()

                            elevation_map_mask = np.where(zgc > threshold_level, 1, 0)

                            plt.figure()
                            plt.pcolor(elevation_map_mask)
                            plt.title(f"Elevation map mask {m} x {n}")
                            plt.axis("equal")
                            plt.show()

                            input("Press Enter to continue...")

                        cell = Cell2(elevation_map=zgc, threshold_level=threshold_level)

                        for idx in range(nvec):
                            covered_ratio = cell.project_on_plane(vectors[idx])
                            self.block_coefficient[idx, index] = covered_ratio
                            self.block_coefficient[idx + nvec, index] = covered_ratio

                        if showcase:
                            print(
                                "blocking coefficient:\n",
                                self.block_coefficient[:, index],
                            )
                            print(
                                "For angles:\n",
                                np.concatenate((angles, angles + 180.0)),
                            )

        if not quiet:
            print(f"Total number of cells processed: {counter}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

        if file_name:
            block_coefficient = self.block_coefficient

            dims = ("directions", "cells")
            coords = {
                "cells": np.arange(0, nr_cells),
                "directions": np.concatenate((angles, angles + 180.0)),
            }

            data_array = xr.DataArray(
                block_coefficient, dims=dims, coords=coords, name="blocking_coefficient"
            )
            data_array.attrs["description"] = "Blocking coefficient of cell"

            dataset = xr.Dataset({"blocking_coefficient": data_array})
            dataset.attrs["title"] = "Hurrywave wave blocking file"
            dataset.attrs["institution"] = "Deltares"
            dataset.attrs["source"] = "Deltares"
            dataset.attrs["history"] = f"Created {pd.Timestamp.now()}"

            dataset.to_netcdf(os.path.join(self.model.path, file_name))
            print(f"Dataset saved as '{file_name}'")

        return dataset


class Cell:
    """
    Obstacle-projection geometry for a single subgrid cell (legacy version).

    Parameters
    ----------
    elevation_map : array-like
        2-D elevation array for the cell.
    threshold_wl : float, optional
        Elevation threshold above which pixels are treated as obstacles.
        Default is 0.
    """

    def __init__(self, elevation_map, threshold_wl: float = 0) -> None:
        self.width = len(elevation_map[0])
        self.height = len(elevation_map)
        self.dx = 1
        self.dy = 1
        self.obstacles = self.extract_obstacles(elevation_map, threshold_wl)
        self.midpoint, self.circle_radius = self.circle_around_cell(
            self.width, self.height
        )

    def circle_around_cell(self, cell_width: int, cell_height: int) -> tuple:
        """
        Calculate the bounding circle of a cell.

        Parameters
        ----------
        cell_width : int
            Width of the cell in pixels.
        cell_height : int
            Height of the cell in pixels.

        Returns
        -------
        center : tuple of float
            ``(center_x, center_y)``
        radius : float
            Radius of the bounding circle.
        """
        diagonal_length = math.sqrt(cell_width**2 + cell_height**2)
        center_x = cell_width / 2
        center_y = cell_height / 2
        radius = diagonal_length / 2
        return (center_x, center_y), radius

    def extract_obstacles(self, elevation_map, threshold_wl: float) -> list:
        """
        Extract obstacle pixel coordinates from an elevation map.

        Parameters
        ----------
        elevation_map : array-like
            2-D elevation values.
        threshold_wl : float
            Pixels with elevation above this value are obstacles.

        Returns
        -------
        list of tuple
            List of ``(x, y)`` pixel coordinates that are obstacles.
        """
        obstacles = []
        for y in range(self.height):
            for x in range(self.width):
                if elevation_map[y][x] > threshold_wl:
                    obstacles.append((x, y))
        return obstacles

    def project_point_on_line(
        self, point: tuple, line_start: tuple, line_end: tuple
    ) -> tuple:
        """
        Project a point orthogonally onto a line segment.

        Parameters
        ----------
        point : tuple of float
            Point to project.
        line_start : tuple of float
            Start of the line.
        line_end : tuple of float
            End of the line.

        Returns
        -------
        projection : tuple of float
            Projected point coordinates.
        projection_length : float
            Parametric distance along the line (0 = start, 1 = end).
        """
        line_vector = np.array(line_end) - np.array(line_start)
        point_vector = np.array(point) - np.array(line_start)
        projection_length = np.dot(point_vector, line_vector) / np.dot(
            line_vector, line_vector
        )
        projection = np.array(line_start) + projection_length * line_vector
        return tuple(projection), projection_length

    def project_on_plane(self, incoming_direction: np.ndarray) -> float:
        """
        Compute the fraction of the projection plane covered by obstacles.

        Parameters
        ----------
        incoming_direction : numpy.ndarray
            Unit vector ``[cos, sin, 0]`` of the wave direction.

        Returns
        -------
        float
            Covered ratio in [0, 1].
        """
        incoming_direction_rad = np.arctan2(
            incoming_direction[1], incoming_direction[0]
        )

        projection_plane_midpoint = (
            self.midpoint[0]
            + self.circle_radius * np.cos(incoming_direction_rad + np.pi),
            self.midpoint[1]
            - self.circle_radius * np.sin(incoming_direction_rad + np.pi),
        )

        orthonal_vector = (-incoming_direction[1], incoming_direction[0])
        x0, y0 = (
            projection_plane_midpoint[0] - (self.width * orthonal_vector[0]),
            projection_plane_midpoint[1] + orthonal_vector[1] * self.height,
        )
        x1, y1 = (
            projection_plane_midpoint[0] + (self.width * orthonal_vector[0]),
            projection_plane_midpoint[1] - orthonal_vector[1] * self.height,
        )

        corners_cell = [
            (0, 0),
            (self.width, 0),
            (self.width, self.height),
            (0, self.height),
        ]

        projected_corners = []
        projected_lengths_ratio = []

        for corner in corners_cell:
            projected_corner, projected_length_ratio = self.project_point_on_line(
                corner, (x0, y0), (x1, y1)
            )
            projected_corners.append(projected_corner)
            projected_lengths_ratio.append(projected_length_ratio)

        min_idx, max_idx = (
            np.argmin(projected_lengths_ratio),
            np.argmax(projected_lengths_ratio),
        )
        x0_cut, y0_cut = projected_corners[min_idx]
        x1_cut, y1_cut = projected_corners[max_idx]

        obstacles_idx = self.obstacles

        nrp = 100
        pnts = np.zeros(nrp).astype(int)

        for obstacle in obstacles_idx:
            x, y = obstacle
            corners_obstacle = [
                (x, y),
                (x + self.dx, y),
                (x + self.dx, y + self.dy),
                (x, y + self.dy),
            ]

            projected_lengths_ratio_obs = [
                self.project_point_on_line(
                    corner_obs, (x0_cut, y0_cut), (x1_cut, y1_cut)
                )[1]
                for corner_obs in corners_obstacle
            ]
            obs_projection_ratio = (
                np.min(projected_lengths_ratio_obs),
                np.max(projected_lengths_ratio_obs),
            )

            i0 = int(obs_projection_ratio[0] * nrp)
            i1 = int(obs_projection_ratio[1] * nrp)
            pnts[i0:i1] = 1

        covered_ratio = np.sum(pnts) / nrp
        return covered_ratio


class Cell2:
    """
    Vectorised obstacle-projection geometry for a single subgrid cell.

    Parameters
    ----------
    elevation_map : numpy.ndarray
        2-D elevation array for the cell.
    threshold_level : float, optional
        Elevation threshold above which pixels are treated as obstacles.
        Default is 0.
    """

    def __init__(self, elevation_map: np.ndarray, threshold_level: float = 0) -> None:
        self.height = np.shape(elevation_map)[0]
        self.width = np.shape(elevation_map)[1]
        self.dx = 1.0
        self.dy = 1.0
        self.extract_obstacles(elevation_map, threshold_level)
        self.midpoint, self.circle_radius = self.circle_around_cell(
            self.width, self.height
        )

    def circle_around_cell(self, cell_width: int, cell_height: int) -> tuple:
        """
        Calculate the bounding circle of a cell.

        Parameters
        ----------
        cell_width : int
            Width of the cell in pixels.
        cell_height : int
            Height of the cell in pixels.

        Returns
        -------
        center : tuple of float
        radius : float
        """
        diagonal_length = math.sqrt(cell_width**2 + cell_height**2)
        center_x = cell_width / 2
        center_y = cell_height / 2
        radius = diagonal_length / 2
        return (center_x, center_y), radius

    def extract_obstacles(
        self, elevation_map: np.ndarray, threshold_wl: float
    ) -> None:
        """
        Populate ``obscor_x`` and ``obscor_y`` from suprathreshold pixels.

        Parameters
        ----------
        elevation_map : numpy.ndarray
            2-D elevation values.
        threshold_wl : float
            Pixels above this elevation are treated as obstacles.
        """
        n, m = np.where(elevation_map > threshold_wl)
        nobs = np.size(m)
        self.obscor_x = np.zeros((nobs, 4))
        self.obscor_y = np.zeros((nobs, 4))
        self.obscor_x[:, 0] = m
        self.obscor_x[:, 1] = m + 1
        self.obscor_x[:, 2] = m + 1
        self.obscor_x[:, 3] = m
        self.obscor_y[:, 0] = n
        self.obscor_y[:, 1] = n
        self.obscor_y[:, 2] = n + 1
        self.obscor_y[:, 3] = n + 1

    def project_points_on_line(
        self,
        point_x: np.ndarray,
        point_y: np.ndarray,
        line: np.ndarray,
    ) -> np.ndarray:
        """
        Project obstacle corner points onto a line segment (vectorised).

        Parameters
        ----------
        point_x : numpy.ndarray
            Shape ``(nobs, 4)`` x-coordinates of obstacle corners.
        point_y : numpy.ndarray
            Shape ``(nobs, 4)`` y-coordinates of obstacle corners.
        line : numpy.ndarray
            Shape ``(2, 2)`` array with start and end points of the line.

        Returns
        -------
        numpy.ndarray
            Shape ``(nobs, 4)`` parametric projection distances.
        """
        nobs = np.shape(point_x)[0]
        point_x = point_x.reshape(nobs * 4, 1)
        point_y = point_y.reshape(nobs * 4, 1)

        line_vector = line[1, :] - line[0, :]
        point_vector = np.squeeze(np.array([point_x, point_y])).T - line[0, :]

        projection_length = np.reshape(
            np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector),
            (nobs, 4),
        )
        return projection_length

    def project_on_plane(self, incoming_direction: np.ndarray) -> float:
        """
        Compute the fraction of the projection plane covered by obstacles.

        Parameters
        ----------
        incoming_direction : numpy.ndarray
            Unit vector ``[cos, sin, 0]`` of the wave direction.

        Returns
        -------
        float
            Covered ratio in [0, 1].
        """
        incoming_direction_rad = np.arctan2(
            incoming_direction[1], incoming_direction[0]
        )

        projection_plane_midpoint = (
            self.midpoint[0]
            + self.circle_radius * np.cos(incoming_direction_rad + np.pi),
            self.midpoint[1]
            - self.circle_radius * np.sin(incoming_direction_rad + np.pi),
        )

        orthonal_vector = (-incoming_direction[1], incoming_direction[0])
        x0, y0 = (
            projection_plane_midpoint[0] - self.width * orthonal_vector[0],
            projection_plane_midpoint[1] - orthonal_vector[1] * self.height,
        )
        x1, y1 = (
            projection_plane_midpoint[0] + self.width * orthonal_vector[0],
            projection_plane_midpoint[1] + orthonal_vector[1] * self.height,
        )

        corners_cell_x = np.array([[0.0, self.width, self.width, 0.0]])
        corners_cell_y = np.array([[0.0, 0.0, self.height, self.height]])
        line_vector = np.array([[x0, y0], [x1, y1]])

        projected_length_ratio = self.project_points_on_line(
            corners_cell_x, corners_cell_y, line_vector
        )

        min_idx, max_idx = (
            np.argmin(projected_length_ratio),
            np.argmax(projected_length_ratio),
        )
        x0_cut = corners_cell_x[0, min_idx]
        y0_cut = corners_cell_y[0, min_idx]
        x1_cut = corners_cell_x[0, max_idx]
        y1_cut = corners_cell_y[0, max_idx]

        line_vector = np.array([[x0_cut, y0_cut], [x1_cut, y1_cut]])

        nrp = 100
        pnts = np.zeros(nrp).astype(int)
        projected_lengths_ratio_obs = self.project_points_on_line(
            self.obscor_x, self.obscor_y, line_vector
        )
        min_ratio = np.min(projected_lengths_ratio_obs, axis=1)
        max_ratio = np.max(projected_lengths_ratio_obs, axis=1)
        for iobs in range(np.size(min_ratio)):
            i0 = int(min_ratio[iobs] * nrp)
            i1 = int(max_ratio[iobs] * nrp)
            pnts[i0:i1] = 1

        covered_ratio = np.sum(pnts) / nrp
        return covered_ratio
