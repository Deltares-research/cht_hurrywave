import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import xarray as xr
from cht.misc.misc_tools import interp2
from cht_bathymetry.bathymetry_database import bathymetry_database

class WaveBlockingFile:

    def __init__(self, model, version=0):
        # A regular subgrid table contains only for cells with msk>0
        self.model = model
        self.version = version

    def read(self):
        file_name = os.path.join(self.model.path, self.model.input.variables.sbgfile)
        self.load(file_name)

    
    def build(self,
              bathymetry_sets,
              file_name="hurrywave.wbl",
              mask=None,
              nr_bins=36,
              nr_subgrid_pixels=20,
              quiet=True,
              progress_bar = False,
              showcase=False,):  

        grid = self.model.grid

        refi  = nr_subgrid_pixels
        self.nbins = nr_bins

        counter = 0

        # Step 1: Create a dataset of zeros
        dims = ('directional_bins', 'latitude', 'longitude')
        shape = (nr_bins, grid.nmax, grid.mmax)  # Example dimensions
        self.block_coefficient  = np.zeros(shape, dtype=float)
                
        cosrot = np.cos(grid.rotation*np.pi/180)
        sinrot = np.sin(grid.rotation*np.pi/180)
        nrmax  = 2000
       
        n0 = 0
        n1 = grid.nmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
        m0 = 0
        m1 = grid.mmax - 1 # + 1 # add extra cell to compute u and v in the last row/column
        
        dx   = grid.dx      # cell size
        dy   = grid.dy      # cell size
        dxp  = dx/refi      # size of subgrid pixel
        dyp  = dy/refi      # size of subgrid pixel
        
        nrcb = int(np.floor(nrmax/refi))         # nr of regular cells in a block            
        nrbn = int(np.ceil((n1 - n0 + 1)/nrcb))  # nr of blocks in n direction
        nrbm = int(np.ceil((m1 - m0 + 1)/nrcb))  # nr of blocks in m direction

        if not quiet:
            print("Number of regular cells in a block : " + str(nrcb))
            print("Number of grid cells    : " + str(grid.nmax))
            print("Number of grid cells    : " + str(grid.mmax))
        
        if not quiet:
            print("Grid size of HW grid             : dx= " + str(dx) + ", dy= " + str(dy))
            print("Grid size of waveblocking pixels        : dx= " + str(dxp) + ", dy= " + str(dyp))
            print("Grid size of directional bins        : directions = " + str(nr_bins))

        ## Loop through blocks
        ib = 0
        for ii in range(nrbm):
            for jj in range(nrbn):
            
                
                bn0 = n0  + jj*nrcb               # Index of first n in block
                bn1 = min(bn0 + nrcb - 1, n1) + 1 # Index of last n in block (cut off excess above, but add extra cell to compute u and v in the last row)
                bm0 = m0  + ii*nrcb               # Index of first m in block
                bm1 = min(bm0 + nrcb - 1, m1) + 1 # Index of last m in block (cut off excess to the right, but add extra cell to compute u and v in the last column)

                if not quiet:
                    print("--------------------------------------------------------------")
                    print("Processing block " + str(ib + 1) + " of " + str(nrbn*nrbm) + " ...")

                if progress_bar:
                    progress_bar.set_text("               Generating Wave blocking file ...                ")
                    progress_bar.set_maximum(nrbm * nrbn)
                    
                # Now build the pixel matrix
                x00 = 0.5*dxp + bm0*refi*dyp
                x01 = x00 + (bm1 - bm0 + 1)*refi*dxp
                y00 = 0.5*dyp + bn0*refi*dyp
                y01 = y00 + (bn1 - bn0 + 1)*refi*dyp
                
                x0 = np.arange(x00, x01, dxp)
                y0 = np.arange(y00, y01, dyp)
                xg0, yg0 = np.meshgrid(x0, y0)
                # Rotate and translate
                xg = grid.x0 + cosrot*xg0 - sinrot*yg0
                yg = grid.y0 + sinrot*xg0 + cosrot*yg0                    

                # Clear variables
                del x0, y0, xg0, yg0
                
                # Initialize depth of subgrid at NaN
                zg = np.full(np.shape(xg), np.nan)

                # Get bathy on refined grid

                try: 
                    zg = bathymetry_database.get_bathymetry_on_grid(xg, yg,
                                                                        self.model.crs,
                                                                        bathymetry_sets)
                except:
                    pass   
                    
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
                                    # zb[np.where(zb<bathymetry.zmin)] = np.nan
                                    # zb[np.where(zb>bathymetry.zmax)] = np.nan
                                    if not np.isnan(zb).all():
                                        zg1 = interp2(xb, yb, zb, xgb, ygb)
                                        isn = np.where(np.isnan(zg))
                                        zg[isn] = zg1[isn]

                                         # clear temp variables
                            del xb, yb, zb
            
                        except:
                            pass
                            
               
                
                # Now compute subgrid properties

                # Loop through all active cells in this block
                for m in range(bm0, bm1):
                    for n in range(bn0, bn1):
                        
                        # if self.mask[n, m]<1:
                        #     # Check if computational cell is active
                        #     continue

                        
                        # Get elevation in cells
                        nn  = (n - bn0) * refi
                        mm  = (m - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]
    
                        z_thres = -20
            
                        if np.nanmax(zgc) < z_thres:
                            # Check if any cells above threshold for island
                            continue
                        counter += 1
                        # Plot elevetaion map

                        if showcase:   
                            plt.figure()
                            plt.pcolor(zgc)
                            plt.axis('equal')
                            plt.title(f'Elevation map {m} x {n}')
                            plt.show()

                            # pause the loop for show case
                            elevation_map_mask = np.where(zgc > z_thres, 1, 0)

                            plt.figure()

                            plt.pcolor(elevation_map_mask)
                            plt.title(f'Elevation map mask {m} x {n}')
                            plt.axis('equal')
                            plt.show()

                            input("Press Enter to continue...")


                        # Create a cell object
                        cell = Cell(elevation_map=zgc, threshold_wl=z_thres)
                        angles = np.linspace(0 + 5, 180 + 5, int(nr_bins/2), endpoint=False) 
                        radians = np.deg2rad(angles)

                        # Calculate x, y, z coordinates
                        vectors = np.array([[np.cos(angle), np.sin(angle), 0] for angle in radians])
                        
                        # To-Do: Add loop for secerla directions
                        for idx, incoming_direction in enumerate(vectors):
                            covered_ratio = cell.project_on_plane(incoming_direction)
                            self.block_coefficient[len(vectors) - idx, n, m]  = covered_ratio
                            self.block_coefficient[len(vectors) - (idx + int(nr_bins/2)), n, m]  = covered_ratio

                            
                        if showcase:
                            print("blocking coefficient:\n", self.block_coefficient[:, n, m])
                            
                            print("For angles:\n", np.concatenate((angles, angles + 180)))

                        if progress_bar:
                            ib += 1
                            progress_bar.set_value(ib)
                            if progress_bar.was_canceled():
                                return
                    
        if not quiet:
            print(f"Total number of cells processed: {counter}")
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        
        if file_name:


            # Example data
            block_coefficient = self.block_coefficient
            #additional_variable = self.additional_variable  # Another variable to include

            # Define dimensions and coordinates
            dims = ('directions', 'lon', 'lat')  # Example dimensions
            coords = {
                'lon': np.arange(0, block_coefficient.shape[1]),
                'lat': np.arange(0, block_coefficient.shape[2]),
                'directions': np.concatenate((angles, angles + 180))

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

        projected_obs_ratios = []

        for obstacle in obstacles_idx:
            x, y = obstacle
            corners_obstacle = [(x, y), (x + self.dx, y), (x + self.dx, y + self.dy), (x, y + self.dy)]

            # Calculate the projection of the corner on the plane
            projected_lengths_ratio_obs = [self.project_point_on_line(corner_obs, (x0_cut, y0_cut), (x1_cut, y1_cut))[1] for corner_obs in corners_obstacle]
            obs_projection_ratio = np.min(projected_lengths_ratio_obs), np.max(projected_lengths_ratio_obs)
            projected_obs_ratios.append(obs_projection_ratio)

        # Calculate the total length of the plane that is covered by the objects
        merged_intervals = []
        for interval in sorted(projected_obs_ratios):
            if not merged_intervals or interval[0] > merged_intervals[-1][1]:
                # If the current interval does not overlap with the previous one, add it
                merged_intervals.append(interval)
            else:
                # If the current interval overlaps with the previous one, merge them
                merged_intervals[-1] = merged_intervals[-1][0], max(merged_intervals[-1][1], interval[1])

        # Calculate the total length covered by the objects
        covered_ratio = sum(max_val - min_val for min_val, max_val in merged_intervals)
        
        # print("Combined region covered:", covered_ratio)
        return covered_ratio   
