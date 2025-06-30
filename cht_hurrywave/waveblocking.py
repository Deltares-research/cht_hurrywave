import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import xarray as xr
from cht_utils.misc_tools import interp2

# from cht_bathymetry.bathymetry_database import bathymetry_database

class WaveBlocking:

    """
    Class to compute wave blocking coefficients for a given model grid.

    This class is used to calculate wave blocking coefficients based on bathymetry data. 
    The blocking coefficients are calculated for various directional bins and stored 
    for further use in the model. The calculations involve subgrid processing for each block 
    and can use either a bathymetry database or bathymetry data from a file (e.g., tif files).
    
    Attributes:

    model : Model object
        The model to which this WaveBlocking instance belongs.
    version : int
        The version of the wave blocking calculation (default is 0).
    block_coefficient : ndarray
        Array to store the calculated wave blocking coefficients.
    nbins : int
        Number of directional bins for the calculation.
    
    Methods:
    
    read():
        Reads wave blocking data from a file.
    build(bathymetry_sets, bathymetry_database=None, file_name="hurrywave.wbl", 
          nr_dirs=36, nr_subgrid_pixels=20, threshold_level=-5.0, quiet=True, 
          progress_bar=None, showcase=False):
        Builds the wave blocking coefficients by iterating over grid blocks and calculating 
        the blocking coefficients for each cell based on bathymetry data.
    """

    def __init__(self, model, version=0):
        # A regular subgrid table contains only for cells with msk>0
        self.model = model
        self.version = version

    def read(self):
        file_name = os.path.join(self.model.path, self.model.input.variables.wblfile)
        self.load(file_name)
    
    def build(self,
              bathymetry_sets,
              bathymetry_database=None,
              file_name="hurrywave.wbl",
              nr_dirs=36,
              nr_subgrid_pixels=20,
              threshold_level=-5.0,
              quiet=True,
              progress_bar=None,
              showcase=False):  
       
        refi        = nr_subgrid_pixels
        self.nbins  = nr_dirs
        nrmax       = 2000

        # Get some information from the grid
        grid        = self.model.grid.data
        nr_cells    = grid.nr_cells
        nr_ref_levs = grid.nr_refinement_levels
        x0          = grid.xuds.attrs["x0"]
        y0          = grid.xuds.attrs["y0"]
        nmax        = grid.xuds.attrs["nmax"]
        mmax        = grid.xuds.attrs["mmax"]
        dx          = grid.xuds.attrs["dx"]
        dy          = grid.xuds.attrs["dy"]
        rotation    = grid.xuds.attrs["rotation"]        
        cosrot      = grid.cosrot
        sinrot      = grid.sinrot
        level       = grid.xuds["level"].values[:] - 1
        n           = grid.xuds["n"].values[:] - 1
        m           = grid.xuds["m"].values[:] - 1

        counter = 0

        # Step 1: Create a dataset of zeros
        dims = ('directional_bins', 'cells')
        shape = (nr_dirs, nr_cells)  # Example dimensions
        self.block_coefficient  = np.zeros(shape, dtype=float)

        # This stuff is already stored in the grid!                

        # Grid neighbors (subtract 1 from indices to get zero-based indices)

        # Determine first indices and number of cells per refinement level
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


        # Loop through all levels
        for ilev in range(nr_ref_levs):

            # Make blocks off cells in this level only
            cell_indices_in_level = np.arange(ifirst[ilev], ilast[ilev] + 1, dtype=int)
            nr_cells_in_level = np.size(cell_indices_in_level)

            if nr_cells_in_level == 0:
                continue

            n0 = np.min(n[ifirst[ilev] : ilast[ilev] + 1])
            n1 = np.max(
                n[ifirst[ilev] : ilast[ilev] + 1]
            )  # + 1 # add extra cell to compute u and v in the last row/column
            m0 = np.min(m[ifirst[ilev] : ilast[ilev] + 1])
            m1 = np.max(
                m[ifirst[ilev] : ilast[ilev] + 1]
            )  # + 1 # add extra cell to compute u and v in the last row/column

            dxi = dx / 2**ilev  # cell size at this level
            dyi = dy / 2**ilev  # cell size at this level
            dxp = dxi / refi    # size of subgrid pixel
            dyp = dyi / refi    # size of subgrid pixel

            nrcb = int(np.floor(nrmax / refi))         # nr of regular cells in a block
            nrbn = int(np.ceil((n1 - n0 + 1) / nrcb))  # nr of blocks in n direction
            nrbm = int(np.ceil((m1 - m0 + 1) / nrcb))  # nr of blocks in m direction

        # n0 = 0
        # n1 = grid.nmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
        # m0 = 0
        # m1 = grid.mmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
        
        # dx   = grid.dx      # cell size
        # dy   = grid.dy      # cell size
        # dxp  = dx / refi      # size of subgrid pixel
        # dyp  = dy / refi      # size of subgrid pixel
        
        # nrcb = int(np.floor(nrmax/refi))         # nr of regular cells in a block            
        # nrbn = int(np.ceil((n1 - n0 + 1)/nrcb))  # nr of blocks in n direction
        # nrbm = int(np.ceil((m1 - m0 + 1)/nrcb))  # nr of blocks in m direction

            if not quiet:
                print("Number of regular cells in a block : " + str(nrcb))
                print("Number of grid cells    : " + str(grid.nmax))
                print("Number of grid cells    : " + str(grid.mmax))
            
            if not quiet:
                print("Grid size of HW grid             : dx= " + str(dx) + ", dy= " + str(dy))
                print("Grid size of waveblocking pixels        : dx= " + str(dxp) + ", dy= " + str(dyp))
                print("Grid size of directional bins        : directions = " + str(nr_dirs))

            if progress_bar:
                progress_bar.set_text("               Computing wave blocking coefficients ...                ")
                progress_bar.set_minimum(0)
                progress_bar.set_maximum(nrbm * nrbn)
                progress_bar.set_value(0)
                                
            ## Loop through blocks
            ib = 0
            for ii in range(nrbm):
                for jj in range(nrbn):

                    if progress_bar:
                        # perc_ready = int(100*ib/(nrbn*nrbm))
                        progress_bar.set_value(ib)
                        if progress_bar.was_canceled():
                            return False

                    ib += 1

                    if not quiet:
                        print("--------------------------------------------------------------")
                        print("Processing block " + str(ib) + " of " + str(nrbn*nrbm) + " ...")
                        print("Getting bathymetry data ...")

                    bn0 = n0  + jj*nrcb               # Index of first n in block
                    bn1 = min(bn0 + nrcb - 1, n1) + 1 # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                    bm0 = m0  + ii*nrcb               # Index of first m in block
                    bm1 = min(bm0 + nrcb - 1, m1) + 1 # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                    # Now build the pixel matrix
                    x00 = 0.5 * dxp + bm0 * refi * dyp
                    x01 = x00 + (bm1 - bm0 + 1) * refi * dxp
                    y00 = 0.5 * dyp + bn0 * refi * dyp
                    y01 = y00 + (bn1 - bn0 + 1) * refi * dyp

                    x0v = np.arange(x00, x01, dxp)
                    y0v = np.arange(y00, y01, dyp)
                    xg0, yg0 = np.meshgrid(x0v, y0v)

                    # Rotate and translate
                    xg = x0 + cosrot * xg0 - sinrot * yg0
                    yg = y0 + sinrot * xg0 + cosrot * yg0

                    # Clear temporary variables
                    del x0v, y0v, xg0, yg0
                    
                    # Initialize depth of subgrid at NaN
                    zg = np.full(np.shape(xg), np.nan)

                    # Get bathy on refined grid
                    # Start using HydroMT for this at some point
                    if bathymetry_database is not None:
                        # Getting bathymetry array zg from database
                        try: 
                            zg = bathymetry_database.get_bathymetry_on_grid(xg, yg,
                                                                            self.model.crs,
                                                                            bathymetry_sets)
                        except Exception as e:
                            print(e)
                            pass
                            return

                    else:
                        # Getting bathymetry array zg from tif file(s)                      
                        # Loop through bathymetry datasets
                        for ibathy, bathymetry in enumerate(bathymetry_sets):
                            # Check if there are NaNs left in this block 
                            if np.isnan(zg).any():
                                try:
                                    if bathymetry.attrs["type"] == "tif_file":
                                        xgb, ygb = xg, yg
                                        # Get DEM data (ddb format for now)
                                        xb, yb, zb = bathymetry.x, bathymetry.y, bathymetry[0].data
                                        if zb is not np.nan:
                                            if not np.isnan(zb).all():
                                                zg1 = interp2(xb, yb, zb, xgb, ygb)
                                                isn = np.where(np.isnan(zg))
                                                zg[isn] = zg1[isn]
                                    # clear temp variables
                                    del xb, yb, zb
                                except Exception as e:
                                    print(e)
                                    return
                    
                    # Now compute subgrid properties
                    if not quiet:
                        print("Computing blocking coefficients ...")

                    # Calculate angles
                    nvec = int(nr_dirs / 2)
                    dtheta = 360.0 / nr_dirs    
                    angles = np.linspace(0.5 * dtheta, 180.0 + 0.5 * dtheta, nvec, endpoint=False) 
                    radians = np.deg2rad(angles)
                    vectors = np.array([[np.cos(angle), np.sin(angle), 0] for angle in radians])

                    # Now find available cells in this block

                    # First we loop through all the possible cells in this block
                    index_cells_in_block = np.zeros(nrcb * nrcb, dtype=int)

                    # Loop through all cells in this level
                    nr_cells_in_block = 0
                    for ic in range(nr_cells_in_level):
                        indx = cell_indices_in_level[ic]  # index of the whole quadtree
                        if (
                            n[indx] >= bn0
                            and n[indx] < bn1
                            and m[indx] >= bm0
                            and m[indx] < bm1
                        ):
                            # Cell falls inside block
                            index_cells_in_block[nr_cells_in_block] = indx
                            nr_cells_in_block += 1

                    if nr_cells_in_block == 0: 
                        # No cells in this block
                        continue

                    index_cells_in_block = index_cells_in_block[0:nr_cells_in_block]

                    for index in index_cells_in_block:

                        # Get elevation in cells
                        nn  = (n[index] - bn0) * refi
                        mm  = (m[index] - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]
    
                        if np.nanmax(zgc) < threshold_level:
                            # Check if any cells above threshold for island
                            continue
                        counter += 1

                        # Plot elevation map
                        if showcase:   
                            plt.figure()
                            plt.pcolor(zgc)
                            plt.axis('equal')
                            plt.title(f'Elevation map {m} x {n}')
                            plt.show()

                            # pause the loop for show case
                            elevation_map_mask = np.where(zgc > threshold_level, 1, 0)

                            plt.figure()

                            plt.pcolor(elevation_map_mask)
                            plt.title(f'Elevation map mask {m} x {n}')
                            plt.axis('equal')
                            plt.show()

                            input("Press Enter to continue...")

                        # Should maybe parallelize this loop

                        # Create a cell object
                        # cell = Cell(elevation_map=zgc, threshold_wl=threshold_level)
                        cell = Cell2(elevation_map=zgc, threshold_level=threshold_level)
                        
                        # To-Do: Add loop for secerla directions

                        # for idx, incoming_direction in enumerate(vectors):
                        #     covered_ratio = cell.project_on_plane(incoming_direction)
                        #     self.block_coefficient[len(vectors) - idx, n, m]  = covered_ratio
                        #     self.block_coefficient[len(vectors) - (idx + int(nr_dirs/2)), n, m]  = covered_ratio

                        for idx in range(nvec):
                            covered_ratio = cell.project_on_plane(vectors[idx])
                            self.block_coefficient[idx, index] = covered_ratio
                            self.block_coefficient[idx + nvec, index] = covered_ratio
                            
                        if showcase:
                            print("blocking coefficient:\n", self.block_coefficient[:, index])
                            print("For angles:\n", np.concatenate((angles, angles + 180.0)))

        if not quiet:
            print(f"Total number of cells processed: {counter}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        if file_name:

            # Example data
            block_coefficient = self.block_coefficient
            #additional_variable = self.additional_variable  # Another variable to include

            # Define dimensions and coordinates
            dims = ('directions', 'cells')  # Example dimensions
            coords = {
                'cells': np.arange(0, nr_cells),
                'directions': np.concatenate((angles, angles + 180.0))
            }

            # Create DataArray with attributes
            data_array = xr.DataArray(block_coefficient, dims=dims, coords=coords, name='blocking_coefficient')
            data_array.attrs['description'] = 'Blocking coefficient of cell'

            # Create Dataset and add global attributes
            dataset = xr.Dataset({'blocking_coefficient': data_array})
            dataset.attrs['title'] = 'Hurrywave wave blocking file'
            dataset.attrs['institution'] = 'Deltares'
            dataset.attrs['source'] = 'Deltares'
            dataset.attrs['history'] = 'Created ' + str(pd.Timestamp.now())

            # Save the dataset to a NetCDF file
            dataset.to_netcdf(os.path.join(self.model.path, file_name))
            print(f"Dataset saved as '{file_name}")

            # # And now rewrite the file, with dimension nr_points, with flattened data
            # # lon and lat are also written to the file, but must also be flattened and have nr_points as dimension
            # dataset = xr.Dataset()
            # dims = ('nr_points', 'directions')
            # nr_points = block_coefficient.shape[1] * block_coefficient.shape[2]
            # coords = {
            #     'nr_points': np.arange(nr_points),
            #     'directions': np.concatenate((angles, angles + 180.0))
            # }
            # data_array = xr.DataArray(block_coefficient.reshape((nr_points, -1)), dims=dims, coords=coords, name='blocking_coefficient')
            # dataset['blocking_coefficient'] = data_array
            # dataset.attrs['title'] = 'Hurrywave wave blocking file'
            # dataset.attrs['institution'] = 'Deltares'
            # dataset.attrs['source'] = 'Deltares'
            # dataset.attrs['history'] = 'Created ' + str(pd.Timestamp.now())
            # dataset.to_netcdf(os.path.join(self.model.path, file_name.replace('.wbl', '_flat.nc')))
            # print(f"Dataset saved as '{file_name.replace('.wbl', '_flat.nc')}'")


        return dataset

class Cell:
    def __init__(self, elevation_map, threshold_wl=0):
        self.width = len(elevation_map[0]) # width of the cell
        self.height = len(elevation_map) # height of the cell 
        self.dx = 1 # dx
        self.dy = 1 # dy
        self.obstacles = self.extract_obstacles(elevation_map, threshold_wl) # obstacles
        self.midpoint, self.circle_radius = self.circle_around_cell(self.width, self.height) # circle around cell

    
    def circle_around_cell(self, cell_width, cell_height):

        '''Function to calculate cricle around a cell based on height and width'''

        # Calculate diagonal length of the cell
        diagonal_length = math.sqrt(cell_width**2 + cell_height**2)
        
        # Calculate center of the circle (midpoint of the diagonal)
        center_x = cell_width / 2
        center_y = cell_height / 2
        
        # Radius of the circle (half of the diagonal length)
        radius = diagonal_length / 2
        
        return (center_x, center_y), radius

    def extract_obstacles(self, elevation_map, threshold_wl):

        '''Extract obstacles based on elevation within cell'''

        obstacles = []
        for y in range(self.height):
            for x in range(self.width):
                if elevation_map[y][x] > threshold_wl:
                    obstacles.append((x, y))
        return obstacles

    
    def project_point_on_line(self, point, line_start, line_end):

        '''Project a point on a line'''

        # Calculate the orthogonal projection of the point on the line

        line_vector = np.array(line_end) - np.array(line_start)
        point_vector = np.array(point) - np.array(line_start)
        
        # Projection length of the point vector alonmg the line vector is calculated by the dot product of the two vectors devided by the dot product of the line vector with itself

        projection_length = np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector)

        # The actual coordinates of the projection are calculated by adding the projection length to the line start

        projection = np.array(line_start) + projection_length * line_vector
        return tuple(projection), projection_length
    
    def project_on_plane(self, incoming_direction):
        '''Project the corners of the objects on a plane'''

        # Define plane based on incoming direction
        incoming_direction_rad = np.arctan2(incoming_direction[1], incoming_direction[0])

        # Plane midpoints
        projection_plane_midpoint = (
            self.midpoint[0] + self.circle_radius * np.cos(incoming_direction_rad + np.pi),
            self.midpoint[1] - self.circle_radius * np.sin(incoming_direction_rad + np.pi)
        )

        # End and begin of plane (infinite length) which is orthogonal to the incoming direction and goes through projection_plane_midpoint
        orthonal_vector = (-incoming_direction[1], incoming_direction[0])  # Orthogonal vector
        x0, y0 = projection_plane_midpoint[0] - (self.width*orthonal_vector[0]), projection_plane_midpoint[1] + orthonal_vector[1]  * self.height
        x1, y1 = projection_plane_midpoint[0] + (self.width*orthonal_vector[0]), projection_plane_midpoint[1] - orthonal_vector[1]  * self.height

        # Define the corners of the cell
        corners_cell = [(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)]

        # Project these corners on the plane defined by (x0, y0) and (x1, y1)
        projected_corners = []
        projected_lengths_ratio = []

        for corner in corners_cell:
            # Calculate the projection of the corner on the plane
            projected_corner, projected_length_ratio = self.project_point_on_line(corner, (x0, y0), (x1, y1))
            projected_corners.append(projected_corner)
            projected_lengths_ratio.append(projected_length_ratio)

        min_idx, max_idx = np.argmin(projected_lengths_ratio), np.argmax(projected_lengths_ratio)
        x0_cut, y0_cut = projected_corners[min_idx]
        x1_cut, y1_cut = projected_corners[max_idx]

        # Project the corners of the objects on the plane
        obstacles_idx = self.obstacles

        # projected_obs_ratios = []

        nrp = 100
        pnts = np.zeros(nrp).astype(int)

        for obstacle in obstacles_idx:
            x, y = obstacle
            corners_obstacle = [(x, y), (x + self.dx, y), (x + self.dx, y + self.dy), (x, y + self.dy)]

            # Calculate the projection of the corner on the plane
            projected_lengths_ratio_obs = [self.project_point_on_line(corner_obs, (x0_cut, y0_cut), (x1_cut, y1_cut))[1] for corner_obs in corners_obstacle]
            obs_projection_ratio = np.min(projected_lengths_ratio_obs), np.max(projected_lengths_ratio_obs)
            # projected_obs_ratios.append(obs_projection_ratio)

            i0 = int(obs_projection_ratio[0] * nrp)
            i1 = int(obs_projection_ratio[1] * nrp)
            pnts[i0:i1] = 1

        covered_ratio = np.sum(pnts) / nrp    

        # # Calculate the total length of the plane that is covered by the objects
        # merged_intervals = []
        # for interval in sorted(projected_obs_ratios):
        #     if not merged_intervals or interval[0] > merged_intervals[-1][1]:
        #         # If the current interval does not overlap with the previous one, add it
        #         merged_intervals.append(interval)
        #     else:
        #         # If the current interval overlaps with the previous one, merge them
        #         merged_intervals[-1] = merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1])

        # # Calculate the total length covered by the objects
        # covered_ratio = sum(max_val - min_val for min_val, max_val in merged_intervals)

        # print("Combined region covered:", covered_ratio)
        return covered_ratio   

class Cell2:
    def __init__(self, elevation_map, threshold_level=0):
        self.height = np.shape(elevation_map)[0] # height of the cell 
        self.width = np.shape(elevation_map)[1] # width of the cell
        self.dx = 1.0 # dx
        self.dy = 1.0 # dy
        self.extract_obstacles(elevation_map, threshold_level) # obstacles
        self.midpoint, self.circle_radius = self.circle_around_cell(self.width, self.height) # circle around cell
    
    def circle_around_cell(self, cell_width, cell_height):

        '''Function to calculate circle around a cell based on height and width'''

        # Calculate diagonal length of the cell
        diagonal_length = math.sqrt(cell_width**2 + cell_height**2)
        
        # Calculate center of the circle (midpoint of the diagonal)
        center_x = cell_width / 2
        center_y = cell_height / 2
        
        # Radius of the circle (half of the diagonal length)
        radius = diagonal_length / 2
        
        return (center_x, center_y), radius

    def extract_obstacles(self, elevation_map, threshold_wl):

        '''Extract obstacles based on elevation within cell'''

        # Find subgrid pixels that are above threshold
        n, m = np.where(elevation_map > threshold_wl)
        nobs = np.size(m)
        # Create empty arrays for the x and y coordinates of the obstacles
        self.obscor_x = np.zeros((nobs, 4))
        self.obscor_y = np.zeros((nobs, 4))
        self.obscor_x[:,0] = m
        self.obscor_x[:,1] = m + 1
        self.obscor_x[:,2] = m + 1
        self.obscor_x[:,3] = m
        self.obscor_y[:,0] = n
        self.obscor_y[:,1] = n
        self.obscor_y[:,2] = n + 1
        self.obscor_y[:,3] = n + 1
   
    def project_points_on_line(self, point_x, point_y, line):

        """Project a point on a line"""

        # point_x and point_y are originally 4 by n arrays with the x and y coordinates of the corners of the objects
        # First, reshape the arrays to 4nx1 arrays
        nobs = np.shape(point_x)[0]
        point_x = point_x.reshape(nobs*4, 1)
        point_y = point_y.reshape(nobs*4, 1)

        # Calculate the orthogonal projection of the point on the line
        line_vector = line[1, :] - line[0, :]
        point_vector = np.squeeze(np.array([point_x, point_y])).T - line[0, :]
        
        # Projection length of the point vector alonmg the line vector is calculated by the dot product of the two vectors devided by the dot product of the line vector with itself
        projection_length = np.reshape(np.dot(point_vector, line_vector) / np.dot(line_vector, line_vector), (nobs, 4))

        # The actual coordinates of the projection are calculated by adding the projection length to the line start
        # I don't think this is necessary, only the projection length is needed
        # projection = line[0,:] + projection_length * line_vector
        # return tuple(projection), projection_length
        return projection_length
    
    def project_on_plane(self, incoming_direction):
        '''Project the corners of the objects on a plane'''

        # Define plane based on incoming direction
        incoming_direction_rad = np.arctan2(incoming_direction[1], incoming_direction[0])

        # Plane midpoints
        projection_plane_midpoint = (
            self.midpoint[0] + self.circle_radius * np.cos(incoming_direction_rad + np.pi),
            self.midpoint[1] - self.circle_radius * np.sin(incoming_direction_rad + np.pi)
        )

        # End and begin of plane (infinite length) which is orthogonal to the incoming direction and goes through projection_plane_midpoint
        orthonal_vector = (-incoming_direction[1], incoming_direction[0])  # Orthogonal vector
        # x0, y0 = projection_plane_midpoint[0] - self.width * orthonal_vector[0], projection_plane_midpoint[1] + orthonal_vector[1] * self.height
        # x1, y1 = projection_plane_midpoint[0] + self.width * orthonal_vector[0], projection_plane_midpoint[1] - orthonal_vector[1] * self.height
        # Should this not be this ?:
        x0, y0 = projection_plane_midpoint[0] - self.width * orthonal_vector[0], projection_plane_midpoint[1] - orthonal_vector[1] * self.height
        x1, y1 = projection_plane_midpoint[0] + self.width * orthonal_vector[0], projection_plane_midpoint[1] + orthonal_vector[1] * self.height

        # Define the corners of the cell
        corners_cell_x = np.array([[0.0, self.width, self.width, 0.0]])
        corners_cell_y = np.array([[0.0, 0.0, self.height, self.height]])
        line_vector = np.array([[x0, y0] , [x1, y1]])

        # Calculate the projection of the corners on the plane
        projected_length_ratio = self.project_points_on_line(corners_cell_x, corners_cell_y, line_vector)

        min_idx, max_idx = np.argmin(projected_length_ratio), np.argmax(projected_length_ratio)
        x0_cut = corners_cell_x[0, min_idx]
        y0_cut = corners_cell_y[0, min_idx]
        x1_cut = corners_cell_x[0, max_idx]
        y1_cut = corners_cell_y[0, max_idx]

        line_vector = np.array([[x0_cut, y0_cut] , [x1_cut, y1_cut]]) 

        # Project the corners of the objects on the plane
        # Use 100 bins (Koen's method probably more accurate this discrete method)
        nrp = 100
        pnts = np.zeros(nrp).astype(int)
        # Get the projected lengths for the 4 corner points  
        projected_lengths_ratio_obs = self.project_points_on_line(self.obscor_x, self.obscor_y, line_vector)
        # projected_lengths_ratio_obs is a (nobs, 4) array with the projected lengths of the objects
        # Get min and max relative position for each obstacle
        min_ratio = np.min(projected_lengths_ratio_obs, axis=1)
        max_ratio = np.max(projected_lengths_ratio_obs, axis=1)
        # Loop through the obstacles and set the bins to 1 where the obstacle is projected on the plane
        for iobs in range(np.size(min_ratio)):
            i0 = int(min_ratio[iobs] * nrp)
            i1 = int(max_ratio[iobs] * nrp)
            pnts[i0:i1] = 1

        # Compute ratio
        covered_ratio = np.sum(pnts) / nrp    

        # Koen's method probably a bit more accurate than discrete method above. Can still revert to his method.
        # # Calculate the total length of the plane that is covered by the objects
        # merged_intervals = []
        # for interval in sorted(projected_obs_ratios):
        #     if not merged_intervals or interval[0] > merged_intervals[-1][1]:
        #         # If the current interval does not overlap with the previous one, add it
        #         merged_intervals.append(interval)
        #     else:
        #         # If the current interval overlaps with the previous one, merge them
        #         merged_intervals[-1] = merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1])

        # # Calculate the total length covered by the objects
        # covered_ratio = sum(max_val - min_val for min_val, max_val in merged_intervals)

        # print("Combined region covered:", covered_ratio)
        return covered_ratio   
