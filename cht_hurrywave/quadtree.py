import time
import os
import numpy as np
import xarray as xr
import pandas as pd
from matplotlib import path
from pyproj import CRS, Transformer
import shapely
from scipy.interpolate import RegularGridInterpolator, griddata
from shapely.geometry import Polygon
from shapely.prepared import prep
import xugrid as xu
import warnings
import geopandas as gpd
import datashader as ds
import datashader.transfer_functions as tf
from datashader.utils import export_image
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling

np.warnings = warnings


class QuadtreeMesh:
    def __init__(self):

        # self.xuds = xu.UgridDataset()
        self.xuds = None
        self.exterior = gpd.GeoDataFrame()
        self.datashader_dataframe = pd.DataFrame()

    def read(self, file_name):
        """
        Read a quadtree grid from a file. We cannot manipulate a grid that has been loaded in! I.e. no cut_inactive_cells!
        In order to do this, we'd need to set all the helper arrays.
        
        Parameters
        ----------
        file_name : str
            Path to the file containing the quadtree grid.
        """
        self.xuds = xu.load_dataset(file_name)

        crd_dict = self.xuds["crs"].attrs

        crs = CRS.from_epsg(4326)  # Default to WGS84
        if "projected_crs_name" in crd_dict:
            crs = CRS(crd_dict["projected_crs_name"])
        elif "geographic_crs_name" in crd_dict:
            crs = CRS(crd_dict["geographic_crs_name"])
        else:
            print("Could not find CRS in quadtree netcdf file. Assuming WGS84.")

        self.crs = crs

        self.get_exterior()

        # Now set the data arrays that are needed for finding neighbors etc.
        self.level = self.xuds.level.values - 1
        self.n = self.xuds.n.values - 1
        self.m = self.xuds.m.values -1
        self.mu = self.xuds.mu.values
        self.mu1 = self.xuds.mu1.values - 1
        self.mu2 = self.xuds.mu2.values - 1
        self.md = self.xuds.md.values
        self.md1 = self.xuds.md1.values - 1
        self.md2 = self.xuds.md2.values - 1
        self.nu = self.xuds.nu.values
        self.nu1 = self.xuds.nu1.values - 1
        self.nu2 = self.xuds.nu2.values - 1
        self.nd = self.xuds.nd.values
        self.nd1 = self.xuds.nd1.values - 1
        self.nd2 = self.xuds.nd2.values - 1
        self.z = self.xuds.z.values

        self.nr_cells = len(self.n)
        self.nr_refinement_levels = np.max(self.level) + 1

        self.x0 = self.xuds.attrs.get("x0", 0.0)
        self.y0 = self.xuds.attrs.get("y0", 0.0)
        self.nmax = self.xuds.attrs.get("nmax", 0)
        self.mmax = self.xuds.attrs.get("mmax", 0)
        self.dx = self.xuds.attrs.get("dx", 0.0)
        self.dy = self.xuds.attrs.get("dy", 0.0)
        self.rotation = self.xuds.attrs.get("rotation", 0.0)
        self.cosrot = np.cos(self.rotation * np.pi / 180)
        self.sinrot = np.sin(self.rotation * np.pi / 180)

        # Do we need any more?

    def write(self, file_name):
        """
        Write the quadtree grid to a file.
        
        Parameters
        ----------
        file_name : str
            Path to the file where the quadtree grid will be saved.
        """
        ds = self.xuds.ugrid.to_dataset()
        ds.attrs = self.xuds.attrs
        ds.to_netcdf(file_name)
        ds.close()

    def build(self, x0, y0, nmax, mmax, dx, dy, rotation, crs,
              refinement_polygons=None,
              bathymetry_sets=None,
              bathymetry_database=None):

        self.x0 = x0
        self.y0 = y0
        self.nmax = nmax
        self.mmax = mmax
        self.dx = dx
        self.dy = dy
        self.rotation = rotation
        self.cosrot = np.cos(rotation*np.pi/180)
        self.sinrot = np.sin(rotation*np.pi/180)
        self.crs = crs

        self.nr_cells = 0
        self.nr_refinement_levels = 1
        self.version = 0

        self.refinement_polygons = refinement_polygons
        self.bathymetry_sets = bathymetry_sets
        self.bathymetry_database = bathymetry_database

        # Make regular grid
        self.get_regular_grid()
 
        # Initialize data arrays 
        self.initialize_data_arrays()

        # Refine all cells 
        if self.refinement_polygons is not None:
            self.refine_mesh()

        # Initialize data arrays
        self.initialize_data_arrays()

        # Get all neighbor arrays (mu, mu1, mu2, nu, nu1, nu2)
        self.get_neighbors()

        # Get uv points
        self.get_uv_points()

        # Create xugrid dataset 
        self.to_xugrid()

        self.get_exterior()

    def get_regular_grid(self):
        # Build initial grid with one level
        ns = np.linspace(0, self.nmax - 1, self.nmax, dtype=int)
        ms = np.linspace(0, self.mmax - 1, self.mmax, dtype=int)
        self.m, self.n = np.meshgrid(ms, ns)
        self.n = np.transpose(self.n).flatten()
        self.m = np.transpose(self.m).flatten()
        self.nr_cells = self.nmax * self.mmax
        self.level = np.zeros(self.nr_cells, dtype=int)
        self.nr_refinement_levels = 1
        # Determine ifirst and ilast for each level
        self.find_first_cells_in_level()
        # Compute cell center coordinates self.x and self.y
        self.compute_cell_center_coordinates()

    def initialize_data_arrays(self):
        # Set indices of neighbors to -1
        self.mu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.mu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.mu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md  = np.zeros(self.nr_cells, dtype=np.int8)
        self.md1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nd1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd2 = np.zeros(self.nr_cells, dtype=int) - 1

        # Set initial depth
        self.z = np.zeros(self.nr_cells, dtype=float)

        # Let's not set the masks here!

        # # Set initial SFINCS mask to zeros
        # self.mask = np.zeros(self.nr_cells, dtype=np.int8)

        # # Set initial SnapWave mask to zeros
        # self.snapwave_mask = np.zeros(self.nr_cells, dtype=np.int8)

    def refine_mesh(self): 
        # Loop through rows in gdf and create list of polygons
        # Determine maximum refinement level

        start = time.time()
        print("Refining ...")

        self.ref_pols = []
        for irow, row in self.refinement_polygons.iterrows():
            iref = row["refinement_level"]
            polygon = {"geometry": row["geometry"], "refinement_level": iref}
            if "zmin" in row:
                polygon["zmin"] = row["zmin"]
            else:
                polygon["zmin"] = -np.inf
            if "zmax" in row:
                polygon["zmax"] = row["zmax"]
            else:
                polygon["zmax"] = np.inf
            self.ref_pols.append(polygon)

        # Loop through refinement polygons and start refining
        for polygon in self.ref_pols:
            # Refine, reorder, find first cells in level
            self.refine_in_polygon(polygon)

        print("Time elapsed : " + str(time.time() - start) + " s")

    def refine_in_polygon(self, polygon):
        # Finds cell to refine and calls refine_cells

        # Loop through refinement levels for this polygon
        for ilev in range(polygon["refinement_level"]):

            # Refine cells in refinement polygons
            # Compute grid spacing for this level
            dx = self.dx/2**ilev
            dy = self.dy/2**ilev
            nmax = self.nmax * 2**ilev
            mmax = self.mmax * 2**ilev
            # Add buffer of 0.5*dx around polygon
            polbuf = polygon["geometry"]

            # Now get the exterior coords of polbuf
            if polbuf.geom_type == "MultiPolygon":
                # Get the exterior coords of all polygons in the MultiPolygon
                all_coords = []
                for poly in polbuf.geoms:
                    all_coords.extend(list(poly.exterior.coords))
                polbuf = Polygon(all_coords)

            coords = polbuf.exterior.coords[:]
            npoints = len(coords)
            polx = np.zeros(npoints)
            poly = np.zeros(npoints)

            for ipoint, point in enumerate(polbuf.exterior.coords[:]):
                # Cell centres
                polx[ipoint] =   self.cosrot*(point[0] - self.x0) + self.sinrot*(point[1] - self.y0)
                poly[ipoint] = - self.sinrot*(point[0] - self.x0) + self.cosrot*(point[1] - self.y0)

            # Find cells cells in grid that could fall within polbuf 
            n0 = int(np.floor(np.min(poly) / dy)) - 1
            n1 = int(np.ceil(np.max(poly) / dy)) + 1
            m0 = int(np.floor(np.min(polx) / dx)) - 1
            m1 = int(np.ceil(np.max(polx) / dx)) + 1

            n0 = min(max(n0, 0), nmax - 1)
            n1 = min(max(n1, 0), nmax - 1)
            m0 = min(max(m0, 0), mmax - 1)
            m1 = min(max(m1, 0), mmax - 1)

            # Generate grid (corners)
            nn, mm = np.meshgrid(np.arange(n0, n1 + 2), np.arange(m0, m1 + 2))
            xx = self.x0 + self.cosrot * mm * dx - self.sinrot * nn * dy
            yy = self.y0 + self.sinrot * mm * dx + self.cosrot * nn * dy
            in_polygon = grid_in_polygon(xx, yy, polygon["geometry"])
            in_polygon = np.transpose(in_polygon).flatten()

            nn, mm = np.meshgrid(np.arange(n0, n1 + 1), np.arange(m0, m1 + 1))
            nn = np.transpose(nn).flatten()
            mm = np.transpose(mm).flatten()

            # Indices of cells in level within polbuf
            nn_in = nn[in_polygon]
            mm_in = mm[in_polygon]
            nm_in = nmax * mm_in + nn_in

            # Find existing cells of this level in nmi array
            n_level = self.n[self.ifirst[ilev]:self.ilast[ilev] + 1]
            m_level = self.m[self.ifirst[ilev]:self.ilast[ilev] + 1]
            nm_level = m_level * nmax + n_level

            # Find indices all cells to be refined
            ind_ref = binary_search(nm_level, nm_in)

            ind_ref = ind_ref[ind_ref>=0]

            # ind_ref = ind_ref[ind_ref < np.size(nm_level)]
            if not np.any(ind_ref):
                continue
            # Index of cells to refine
            ind_ref += self.ifirst[ilev]

            # But only where elevation is between zmin and zmax
            if self.bathymetry_sets is not None and (polygon["zmin"] > -20000.0 or polygon["zmax"] < 20000.0):
                # self.to_xugrid()
                # self.compute_cell_center_coordinates()
                zmin, zmax = self.get_bathymetry_min_max(ind_ref, ilev, self.bathymetry_sets, self.bathymetry_database)
                # z = self.data["z"][ind_ref]
                ind_ref = ind_ref[np.logical_and(zmax > polygon["zmin"], zmin < polygon["zmax"] )]

            if not np.any(ind_ref):
                continue

            self.refine_cells(ind_ref, ilev)

    def refine_cells(self, ind_ref, ilev):
        # Refine cells with index ind_ref

        # First find lower-level neighbors (these will be refined in the next iteration)
        if ilev > 0:
            ind_nbr = self.find_lower_level_neighbors(ind_ref, ilev)
        else:
            ind_nbr = np.empty(0, dtype=int)    

        # n and m indices of cells to be refined
        n = self.n[ind_ref]
        m = self.m[ind_ref]

        # New cells
        nnew = np.zeros(4 * len(ind_ref), dtype=int)
        mnew = np.zeros(4 * len(ind_ref), dtype=int)
        lnew = np.zeros(4 * len(ind_ref), dtype=int) + ilev + 1
        nnew[0::4] = n*2      # lower left
        nnew[1::4] = n*2 + 1  # upper left
        nnew[2::4] = n*2      # lower right
        nnew[3::4] = n*2 + 1  # upper right
        mnew[0::4] = m*2      # lower left
        mnew[1::4] = m*2      # upper left
        mnew[2::4] = m*2 + 1  # lower right
        mnew[3::4] = m*2 + 1  # upper right
        # Add new cells to grid
        self.n = np.append(self.n, nnew)
        self.m = np.append(self.m, mnew)
        self.level = np.append(self.level, lnew)
        # Remove old cells from grid
        self.n = np.delete(self.n, ind_ref)
        self.m = np.delete(self.m, ind_ref)
        self.level = np.delete(self.level, ind_ref)        

        self.nr_cells = len(self.n)
        self.initialize_data_arrays()

        # Update nr_refinement_levels at max of ilev + 2 and self.nr_refinement_levels
        self.nr_refinement_levels = np.maximum(self.nr_refinement_levels, ilev + 2)
        # Reorder cells
        self.reorder()
        # Update ifirst and ilast
        self.find_first_cells_in_level()
        # Compute cell center coordinates self.x and self.y
        self.compute_cell_center_coordinates()

        if np.any(ind_nbr):
            self.refine_cells(ind_nbr, ilev - 1)

    def get_neighbors(self):
        # Get mu, mu1, mu2, nu, nu1, nu2 for all cells   

        start = time.time()

        print("Finding neighbors ...")

        # Get nm indices for all cells
        nm_all = np.zeros(self.nr_cells, dtype=int)
        for ilev in range(self.nr_refinement_levels):
            nmax = self.nmax * 2**ilev + 1
            i0 = self.ifirst[ilev]
            i1 = self.ilast[ilev] + 1
            n = self.n[i0:i1]
            m = self.m[i0:i1]
            nm_all[i0:i1] = m * nmax + n

        # Loop over levels
        for ilev in range(self.nr_refinement_levels):

            nmax = self.nmax * 2**ilev + 1

            # First and last cell in this level
            i0 = self.ifirst[ilev]
            i1 = self.ilast[ilev] + 1

            # Initialize arrays for this level
            mu = np.zeros(i1 - i0, dtype=int)
            mu1 = np.zeros(i1 - i0, dtype=int) - 1
            mu2 = np.zeros(i1 - i0, dtype=int) - 1
            nu = np.zeros(i1 - i0, dtype=int)
            nu1 = np.zeros(i1 - i0, dtype=int) - 1
            nu2 = np.zeros(i1 - i0, dtype=int) - 1

            # Get n and m indices for this level
            n = self.n[i0:i1]
            m = self.m[i0:i1]
            nm = nm_all[i0:i1]

            # Now look for neighbors 
                           
            # Same level

            # Right
            nm_to_find = nm + nmax
            inb = binary_search(nm, nm_to_find)
            mu1[inb>=0] = inb[inb>=0] + i0

            # Above
            nm_to_find = nm + 1
            inb = binary_search(nm, nm_to_find)
            nu1[inb>=0] = inb[inb>=0] + i0

            ## Coarser level neighbors
            if ilev>0:

                nmaxc = self.nmax * 2**(ilev - 1) + 1   # Number of cells in coarser level in n direction 

                i0c = self.ifirst[ilev - 1]  # First cell in coarser level                
                i1c = self.ilast[ilev - 1] + 1 # Last cell in coarser level

                nmc = nm_all[i0c:i1c] # Coarser level nm indices
                nc = n // 2 # Coarser level n index of this cells in this level
                mc = m // 2 # Coarser level m index of this cells in this level 

                # Right
                nmc_to_find = (mc + 1) * nmaxc + nc
                inb = binary_search(nmc, nmc_to_find)
                inb[np.where(even(m))[0]] = -1
                # Set mu and mu1 for inb>=0
                mu1[inb>=0] = inb[inb>=0] + i0c
                mu[inb>=0] = -1

                # Above
                nmc_to_find = mc * nmaxc + nc + 1
                inb = binary_search(nmc, nmc_to_find)
                inb[np.where(even(n))[0]] = -1
                # Set nu and nu1 for inb>=0
                nu1[inb>=0] = inb[inb>=0] + i0c
                nu[inb>=0] = -1

            # Finer level neighbors
            if ilev<self.nr_refinement_levels - 1:

                nmaxf = self.nmax * 2**(ilev + 1) + 1 # Number of cells in finer level in n direction

                i0f = self.ifirst[ilev + 1]  # First cell in finer level
                i1f = self.ilast[ilev + 1] + 1 # Last cell in finer level
                nmf = nm_all[i0f:i1f] # Finer level nm indices

                # Right

                # Lower row
                nf = n * 2 # Finer level n index of this cells in this level
                mf = m * 2 + 1 # Finer level m index of this cells in this level
                nmf_to_find = (mf + 1) * nmaxf + nf
                inb = binary_search(nmf, nmf_to_find)
                mu1[inb>=0] = inb[inb>=0] + i0f
                mu[inb>=0] = 1

                # Upper row
                nf = n * 2 + 1# Finer level n index of this cells in this level
                mf = m * 2 + 1 # Finer level m index of this cells in this level
                nmf_to_find = (mf + 1) * nmaxf + nf
                inb = binary_search(nmf, nmf_to_find)
                mu2[inb>=0] = inb[inb>=0] + i0f
                mu[inb>=0] = 1

                # Above

                # Left column
                nf = n * 2 + 1 # Finer level n index of this cells in this level
                mf = m * 2 # Finer level m index of this cells in this level
                nmf_to_find = mf * nmaxf + nf + 1
                inb = binary_search(nmf, nmf_to_find)
                nu1[inb>=0] = inb[inb>=0] + i0f
                nu[inb>=0] = 1

                # Right column
                nf = n * 2 + 1 # Finer level n index of this cells in this level
                mf = m * 2 + 1 # Finer level m index of this cells in this level
                nmf_to_find = mf * nmaxf + nf + 1
                inb = binary_search(nmf, nmf_to_find)
                nu2[inb>=0] = inb[inb>=0] + i0f
                nu[inb>=0] = 1

            # Fill in mu, mu1, mu2, nu, nu1, nu2 for this level
            self.mu[i0:i1] = mu
            self.mu1[i0:i1] = mu1
            self.mu2[i0:i1] = mu2
            self.nu[i0:i1] = nu
            self.nu1[i0:i1] = nu1
            self.nu2[i0:i1] = nu2

        print("Time elapsed : " + str(time.time() - start) + " s")

        # Making global model
        # Check if CRS is geographic
        if self.crs.is_geographic:
            # Now check if mmax * dx is 360
            if self.mmax * self.dx > 359 and self.mmax * self.dx < 361:
                # We have a global model
                # Loop through all points
                for ilev in range(self.nr_refinement_levels):                    
                    i0 = self.ifirst[ilev]
                    i1 = self.ilast[ilev] + 1
                    nmaxf = self.nmax * 2**ilev + 1
                    mf = self.mmax * 2**ilev - 1
                    nmf = nm_all[i0:i1]
                    # Loop through all cells
                    for i in range(i0, i1):
                        if self.m[i] > 0:
                            # This cell is not on the left of the model
                            break
                        # Now find matching cell on the right
                        # nm index of cell on RHS of grid
                        nmf_to_find = mf * nmaxf + self.n[i]
                        iright = np.where(nmf==nmf_to_find)[0]
                        if iright.size > 0:
                            iright = iright + i0
                            self.mu[iright] = 0
                            self.mu1[iright] = i
                            self.mu2[iright] = -1

        print("Setting neighbors left and below ...")

        # Right
       
        iok1 = np.where(self.mu1>=0)[0]
        # Same level
        iok2 = np.where(self.mu==0)[0]
        # Indices of cells that have a same level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Indices of neighbors
        imu = self.mu1[iok]
        self.md[imu] = 0
        self.md1[imu] = iok

        # Coarser
        iok2 = np.where(self.mu==-1)[0]
        # Indices of cells that have a coarse level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Odd
        iok_odd  = iok[np.where(odd(self.n[iok]))]
        iok_even = iok[np.where(even(self.n[iok]))]
        imu = self.mu1[iok_odd]
        self.md[imu] = 1
        self.md1[imu] = iok_odd
        imu = self.mu1[iok_even]
        self.md[imu] = 1
        self.md2[imu] = iok_even

        # Finer
        # Lower
        iok1 = np.where(self.mu1>=0)[0]
        # Same level
        iok2 = np.where(self.mu==1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        imu = self.mu1[iok]
        self.md[imu] = -1
        self.md1[imu] = iok
        # Upper
        iok1 = np.where(self.mu2>=0)[0]
        # Same level
        iok2 = np.where(self.mu==1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        imu = self.mu2[iok]
        self.md[imu] = -1
        self.md1[imu] = iok

        # Above
        iok1 = np.where(self.nu1>=0)[0]
        # Same level
        iok2 = np.where(self.nu==0)[0]
        # Indices of cells that have a same level neighbor above
        iok = np.intersect1d(iok1, iok2)
        # Indices of neighbors
        inu = self.nu1[iok]
        self.nd[inu] = 0
        self.nd1[inu] = iok

        # Coarser
        iok2 = np.where(self.nu==-1)[0]
        # Indices of cells that have a coarse level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        # Odd
        iok_odd  = iok[np.where(odd(self.m[iok]))]
        iok_even = iok[np.where(even(self.m[iok]))]
        inu = self.nu1[iok_odd]
        self.nd[inu] = 1
        self.nd1[inu] = iok_odd
        inu = self.nu1[iok_even]
        self.nd[inu] = 1
        self.nd2[inu] = iok_even

        # Finer
        # Left
        iok1 = np.where(self.nu1>=0)[0]
        # Same level
        iok2 = np.where(self.nu==1)[0]
        # Indices of cells that have finer level neighbor above
        iok = np.intersect1d(iok1, iok2)
        inu = self.nu1[iok]
        self.nd[inu] = -1
        self.nd1[inu] = iok
        # Upper
        iok1 = np.where(self.nu2>=0)[0]
        # Same level
        iok2 = np.where(self.nu==1)[0]
        # Indices of cells that have finer level neighbor to the right
        iok = np.intersect1d(iok1, iok2)
        inu = self.nu2[iok]
        self.nd[inu] = -1
        self.nd1[inu] = iok

        print("Time elapsed : " + str(time.time() - start) + " s")

    def get_uv_points(self):

        start = time.time()
        print("Getting uv points ...")

        # Get uv points (do we actually need to do this?)
        self.uv_index_z_nm  = np.zeros((self.nr_cells*4), dtype=int)
        self.uv_index_z_nmu = np.zeros((self.nr_cells*4), dtype=int)
        self.uv_dir         = np.zeros((self.nr_cells*4), dtype=int)
        # Loop through points (SHOULD TRY TO VECTORIZE THIS, but try to keep same order of uv points
        nuv = 0
        for ip in range(self.nr_cells):
            if self.mu1[ip]>=0:
                self.uv_index_z_nm[nuv] = ip        
                self.uv_index_z_nmu[nuv] = self.mu1[ip]     
                self.uv_dir[nuv] = 0
                nuv += 1
            if self.mu2[ip]>=0:
                self.uv_index_z_nm[nuv] = ip        
                self.uv_index_z_nmu[nuv] = self.mu2[ip]     
                self.uv_dir[nuv] = 0     
                nuv += 1
            if self.nu1[ip]>=0:
                self.uv_index_z_nm[nuv] = ip        
                self.uv_index_z_nmu[nuv] = self.nu1[ip]     
                self.uv_dir[nuv] = 1     
                nuv += 1
            if self.nu2[ip]>=0:
                self.uv_index_z_nm[nuv] = ip
                self.uv_index_z_nmu[nuv] = self.nu2[ip]
                self.uv_dir[nuv] = 1
                nuv += 1
        self.uv_index_z_nm  = self.uv_index_z_nm[0 : nuv]
        self.uv_index_z_nmu = self.uv_index_z_nmu[0 : nuv]
        self.uv_dir         = self.uv_dir[0 : nuv]
        self.nr_uv_points   = nuv

        print("Time elapsed : " + str(time.time() - start) + " s")

    def reorder(self):
        # Reorder cells
        # Sort cells by level, then m, then n
        i = np.lexsort((self.n, self.m, self.level))
        self.n = self.n[i]
        self.m = self.m[i]
        self.level = self.level[i]

    def find_first_cells_in_level(self):
        # Find first cell in each level
        self.ifirst = np.zeros(self.nr_refinement_levels, dtype=int)
        self.ilast = np.zeros(self.nr_refinement_levels, dtype=int)
        for ilev in range(0, self.nr_refinement_levels):
            # Find index of first cell with this level
            ifirst = np.where(self.level == ilev)[0]
            if ifirst.size == 0:
                self.ifirst[ilev] = -1
            else:
                self.ifirst[ilev] = ifirst[0]
            #self.ifirst[ilev] = np.where(self.level == ilev)[0][0]
            # Find index of last cell with this level
            if ilev<self.nr_refinement_levels - 1:
                # self.ilast[ilev] = np.where(self.level == ilev + 1)[0][0] - 1
                self.ilast[ilev] = np.where(self.level > ilev)[0][0] - 1
            else:
                self.ilast[ilev] = self.nr_cells - 1

    def find_lower_level_neighbors(self, ind_ref, ilev):

        # ind_ref are the indices of the cells that need to be refined

        n = self.n[ind_ref]
        m = self.m[ind_ref]

        n_odd = np.where(odd(n))
        m_odd = np.where(odd(m))
        n_even = np.where(even(n))
        m_even = np.where(even(m))
        
        ill   = np.intersect1d(n_even, m_even)
        iul   = np.intersect1d(n_odd, m_even)
        ilr   = np.intersect1d(n_even, m_odd)
        iur   = np.intersect1d(n_odd, m_odd)

        n_nbr = np.zeros((2, np.size(n)), dtype=int)        
        m_nbr = np.zeros((2, np.size(n)), dtype=int)

        # LL
        n0 = np.int32(n[ill] / 2)
        m0 = np.int32(m[ill] / 2)
        n_nbr[0, ill] = n0 - 1
        m_nbr[0, ill] = m0
        n_nbr[1, ill] = n0
        m_nbr[1, ill] = m0 - 1
        # UL
        n0 = np.int32((n[iul] - 1) / 2)
        m0 = np.int32(m[iul] / 2)
        n_nbr[0, iul] = n0 + 1
        m_nbr[0, iul] = m0
        n_nbr[1, iul] = n0
        m_nbr[1, iul] = m0 - 1
        # LR
        n0 = np.int32(n[ilr] / 2)
        m0 = np.int32((m[ilr] - 1) / 2)
        n_nbr[0, ilr] = n0 - 1
        m_nbr[0, ilr] = m0
        n_nbr[1, ilr] = n0
        m_nbr[1, ilr] = m0 + 1
        # UR
        n0 = np.int32((n[iur] - 1) / 2)
        m0 = np.int32((m[iur] - 1) / 2)
        n_nbr[0, iur] = n0 + 1
        m_nbr[0, iur] = m0
        n_nbr[1, iur] = n0
        m_nbr[1, iur] = m0 + 1

        nmax = self.nmax * 2**(ilev - 1) + 1

        n_nbr = n_nbr.flatten()
        m_nbr = m_nbr.flatten()
        nm_nbr = m_nbr * nmax + n_nbr
        nm_nbr = np.sort(np.unique(nm_nbr, return_index=False))

        # Actual cells in the coarser level 
        n_level = self.n[self.ifirst[ilev - 1]:self.ilast[ilev - 1] + 1]
        m_level = self.m[self.ifirst[ilev - 1]:self.ilast[ilev - 1] + 1]
        nm_level = m_level * nmax + n_level

        # Find  
        ind_nbr = binary_search(nm_level, nm_nbr)
        ind_nbr = ind_nbr[ind_nbr>=0]

        if np.any(ind_nbr):
            ind_nbr += self.ifirst[ilev - 1]

        return ind_nbr

    def compute_cell_center_coordinates(self):
        # Compute cell center coordinates
        # Loop through refinement levels
        dx = self.dx / 2**self.level
        dy = self.dy  /2**self.level
        self.x = self.x0 + self.cosrot * (self.m + 0.5) * dx - self.sinrot * (self.n + 0.5) * dy
        self.y = self.y0 + self.sinrot * (self.m + 0.5) * dx + self.cosrot * (self.n + 0.5) * dy

    def get_ugrid2d(self):

        tic = time.perf_counter()

        n = self.n
        m = self.m
        level = self.level

        nmax = self.nmax * 2**(self.nr_refinement_levels - 1) + 1

        face_nodes_n = np.full((8,self.nr_cells), -1, dtype=int)
        face_nodes_m = np.full((8,self.nr_cells), -1, dtype=int)
        face_nodes_nm = np.full((8,self.nr_cells), -1, dtype=int)

        # Highest refinement level 
        ifac = 2**(self.nr_refinement_levels - level - 1)
        dxf = self.dx / 2**(self.nr_refinement_levels - 1)
        dyf = self.dy / 2**(self.nr_refinement_levels - 1)

        face_n = n * ifac
        face_m = m * ifac

        # First do the 4 corner points
        face_nodes_n[0, :] = face_n
        face_nodes_m[0, :] = face_m
        face_nodes_n[2, :] = face_n
        face_nodes_m[2, :] = face_m + ifac
        face_nodes_n[4, :] = face_n + ifac
        face_nodes_m[4, :] = face_m + ifac
        face_nodes_n[6, :] = face_n + ifac
        face_nodes_m[6, :] = face_m

        # Find cells with refinement below
        i = np.where(self.nd==1)
        face_nodes_n[1, i] = face_n[i]
        face_nodes_m[1, i] = face_m[i] + ifac[i]/2
        # Find cells with refinement to the right
        i = np.where(self.mu==1)
        face_nodes_n[3, i] = face_n[i] + ifac[i]/2
        face_nodes_m[3, i] = face_m[i] + ifac[i]
        # Find cells with refinement above
        i = np.where(self.nu==1)
        face_nodes_n[5, i] = face_n[i] + ifac[i]
        face_nodes_m[5, i] = face_m[i] + ifac[i]/2
        # Find cells with refinement to the left
        i = np.where(self.md==1)
        face_nodes_n[7, i] = face_n[i] + ifac[i]/2
        face_nodes_m[7, i] = face_m[i]

        # Flatten
        face_nodes_n = face_nodes_n.transpose().flatten()
        face_nodes_m = face_nodes_m.transpose().flatten()

        # Compute nm value of nodes        
        face_nodes_nm = nmax * face_nodes_m + face_nodes_n
        nopoint = max(face_nodes_nm) + 1
        # Set missing points to very high number
        face_nodes_nm[np.where(face_nodes_n==-1)] = nopoint

        # Get the unique nm values
        xxx, index, irev = np.unique(face_nodes_nm, return_index=True, return_inverse=True)
        j = np.where(xxx==nopoint)[0][0] # Index of very high number
        # irev2 = np.reshape(irev, (self.nr_cells, 8))
        # face_nodes_all = irev2.transpose()
        face_nodes_all = np.reshape(irev, (self.nr_cells, 8)).transpose()
        face_nodes_all[np.where(face_nodes_all==j)] = -1

        face_nodes = np.full(face_nodes_all.shape, -1)  # Create a new array filled with -1
        for i in range(face_nodes.shape[1]):
            idx = np.where(face_nodes_all[:,i] != -1)[0]
            face_nodes[:len(idx), i] = face_nodes_all[idx, i]  

        # Now get rid of all the rows where all values are -1
        # Create a mask where each row is True if not all elements in the row are -1
        mask = (face_nodes != -1).any(axis=1)

        # Use this mask to index face_nodes
        face_nodes = face_nodes[mask]

        node_n = face_nodes_n[index[:j]]
        node_m = face_nodes_m[index[:j]]
        node_x = self.x0 + self.cosrot*(node_m*dxf) - self.sinrot*(node_n*dyf)
        node_y = self.y0 + self.sinrot*(node_m*dxf) + self.cosrot*(node_n*dyf)

        toc = time.perf_counter()

        print(f"Got rid of duplicates in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()

        nodes = np.transpose(np.vstack((node_x, node_y)))
        faces = np.transpose(face_nodes)
        fill_value = -1

        ugrid2d = xu.Ugrid2d(nodes[:, 0], nodes[:, 1], fill_value, faces)

        ugrid2d.set_crs(self.crs)

        # Set datashader df to None
        self.df = None 

        toc = time.perf_counter()

        print(f"Made XUGrid in {toc - tic:0.4f} seconds")

        return ugrid2d

    def to_xugrid(self):    

        print("Making XUGrid ...")

        # Create the grid
        ugrid2d = self.get_ugrid2d()

        # Create the dataset
        self.xuds = xu.UgridDataset(grids=ugrid2d)

        # Add attributes
        attrs = {"x0": self.x0,
                 "y0": self.y0,
                 "nmax": self.nmax,
                 "mmax": self.mmax,
                 "dx": self.dx,
                 "dy": self.dy,
                 "rotation": self.rotation,
                 "nr_levels": self.nr_refinement_levels}
        self.xuds.attrs = attrs

        # Now add the data arrays

        self.xuds["crs"] = self.crs.to_epsg()
        self.xuds["crs"].attrs = self.crs.to_cf()
        self.xuds["level"] = xu.UgridDataArray(xr.DataArray(data=self.level + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["z"] = xu.UgridDataArray(xr.DataArray(data=self.z, dims=[ugrid2d.face_dimension]), ugrid2d)
        # self.xuds["mask"] = xu.UgridDataArray(xr.DataArray(data=self.mask, dims=[ugrid2d.face_dimension]), ugrid2d)
        # self.xuds["snapwave_mask"] = xu.UgridDataArray(xr.DataArray(data=self.snapwave_mask, dims=[ugrid2d.face_dimension]), ugrid2d)

        self.xuds["n"] = xu.UgridDataArray(xr.DataArray(data=self.n + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["m"] = xu.UgridDataArray(xr.DataArray(data=self.m + 1, dims=[ugrid2d.face_dimension]), ugrid2d)

        self.xuds["mu"]  = xu.UgridDataArray(xr.DataArray(data=self.mu, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["mu1"] = xu.UgridDataArray(xr.DataArray(data=self.mu1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["mu2"] = xu.UgridDataArray(xr.DataArray(data=self.mu2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["md"]  = xu.UgridDataArray(xr.DataArray(data=self.md, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["md1"] = xu.UgridDataArray(xr.DataArray(data=self.md1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["md2"] = xu.UgridDataArray(xr.DataArray(data=self.md2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)

        self.xuds["nu"]  = xu.UgridDataArray(xr.DataArray(data=self.nu, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["nu1"] = xu.UgridDataArray(xr.DataArray(data=self.nu1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["nu2"] = xu.UgridDataArray(xr.DataArray(data=self.nu2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["nd"]  = xu.UgridDataArray(xr.DataArray(data=self.nd, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["nd1"] = xu.UgridDataArray(xr.DataArray(data=self.nd1 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)
        self.xuds["nd2"] = xu.UgridDataArray(xr.DataArray(data=self.nd2 + 1, dims=[ugrid2d.face_dimension]), ugrid2d)

    def get_bathymetry_min_max(self, ind_ref, ilev, bathymetry_sets, bathymetry_database=None, quiet=True):
        """"Used to determine min and max bathymetry of a cell (used for refinement)"""
        
        if bathymetry_database is None:
            print("Error! No bathymetry database provided!")
            return

        if not quiet:
            print("Getting bathymetry data ...")

        dx = self.dx / 2**ilev
        dy = self.dy / 2**ilev
        xz = self.x0 + self.cosrot * (self.m[ind_ref] + 0.5) * dx - self.sinrot * (self.n[ind_ref] + 0.5) * dy
        yz = self.y0 + self.sinrot * (self.m[ind_ref] + 0.5) * dx + self.cosrot * (self.n[ind_ref] + 0.5) * dy

        # Compute the four corner coordinates of the cell, given that the cosine of the rotation is cosrot and the sine is sinrot and the cell center is xz, yz
        xcor = np.zeros((4, np.size(xz)))
        ycor = np.zeros((4, np.size(xz)))
        xcor[0, :] = xz - 0.5 * self.cosrot * dx - 0.5 * self.sinrot * dy
        ycor[0, :] = yz - 0.5 * self.sinrot * dx + 0.5 * self.cosrot * dy
        xcor[1, :] = xz + 0.5 * self.cosrot * dx - 0.5 * self.sinrot * dy
        ycor[1, :] = yz + 0.5 * self.sinrot * dx + 0.5 * self.cosrot * dy
        xcor[2, :] = xz + 0.5 * self.cosrot * dx + 0.5 * self.sinrot * dy
        ycor[2, :] = yz + 0.5 * self.sinrot * dx - 0.5 * self.cosrot * dy
        xcor[3, :] = xz - 0.5 * self.cosrot * dx + 0.5 * self.sinrot * dy
        ycor[3, :] = yz - 0.5 * self.sinrot * dx - 0.5 * self.cosrot * dy 

        if self.crs.is_geographic:
            dx = dx * 111000.0

        # Now loop through the 4 corners and get the minimum and maximum bathymetry
        for i in range(4):
            zgl = bathymetry_database.get_bathymetry_on_points(xcor[i, :],
                                                               ycor[i, :],
                                                               dx,
                                                               self.crs,
                                                               bathymetry_sets)
            if i == 0:
                zmin = zgl
                zmax = zgl
            else:
                zmin = np.minimum(zmin, zgl)
                zmax = np.maximum(zmax, zgl)    

        return zmin, zmax

    def cut_inactive_cells(self, mask_list=[]):

        if len(mask_list) == 0:
            print("No mask list provided, skipping inactive cell removal!")
            return

        print("Removing inactive cells ...")

        # Set crs
        # Why?
        # self.crs = CRS.from_epsg(self.xuds["crs"].values)

        # In the xugrid data, the indices are 1-based, so we need to subtract 1 
        n = self.xuds["n"].values[:] - 1
        m = self.xuds["m"].values[:] - 1
        level = self.xuds["level"].values[:] - 1
        z = self.xuds["z"].values[:]

        # Loop over mask names
        mask_sum = np.zeros_like(n, dtype=int)
        for mask_name in mask_list:
            if mask_name in self.xuds:
                mask_sum += self.xuds[mask_name].values[:]

        indx = np.where((mask_sum)>0)

        self.nr_refinement_levels = self.xuds.attrs["nr_levels"]
        self.nmax = self.xuds.attrs["nmax"]
        self.mmax = self.xuds.attrs["mmax"]
        self.x0 = self.xuds.attrs["x0"]
        self.y0 = self.xuds.attrs["y0"]
        self.rotation = self.xuds.attrs["rotation"]
        self.cosrot = np.cos(np.radians(self.rotation))
        self.sinrot = np.sin(np.radians(self.rotation))
        self.dx = self.xuds.attrs["dx"]
        self.dy = self.xuds.attrs["dy"]
        self.nr_cells = np.size(indx)
        self.n        = n[indx]
        self.m        = m[indx]
        self.level    = level[indx]
        self.z        = z[indx]

        mask_arrays = {}
        for mask_name in mask_list:
            if mask_name in self.xuds:
                mask_arrays[mask_name] = self.xuds[mask_name].values[indx]

        # Set indices of neighbors to -1
        self.mu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.mu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.mu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md  = np.zeros(self.nr_cells, dtype=np.int8)
        self.md1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.md2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nu1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nu2 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd  = np.zeros(self.nr_cells, dtype=np.int8)
        self.nd1 = np.zeros(self.nr_cells, dtype=int) - 1
        self.nd2 = np.zeros(self.nr_cells, dtype=int) - 1

        self.find_first_cells_in_level()
        self.get_neighbors() 
        self.get_uv_points()
        self.to_xugrid()

        # And set back the mask arrays (surely there must be a more elegant way to do this)
        for mask_name in mask_list:
            self.xuds[mask_name] = xu.UgridDataArray(xr.DataArray(data=mask_arrays[mask_name], dims=[self.xuds.grid.face_dimension]), self.xuds.grid)

    def interpolate_bathymetry(self, x, y, z, method="linear"):
        """x, y, and z are numpy arrays with coordinates and bathymetry values"""
        xy = self.data.grid.face_coordinates
        xz = xy[:, 0]
        yz = xy[:, 1]
        zz = interp2(x, y, z, xz, yz, method=method)
        self.xuds["z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[self.xuds.grid.face_dimension]), self.xuds.grid)

    def set_uniform_bathymetry(self, zb):
        self.xuds["z"][:] = zb

    def set_bathymetry(self, bathymetry_sets, bathymetry_database=None, zmin=-1.0e9, zmax=1.0e9, quiet=True):
        
        if bathymetry_database is None:
            print("Error! No bathymetry database provided!")
            return

        if not quiet:
            print("Getting bathymetry data ...")

        # Number of refinement levels
        nlev = self.xuds.attrs["nr_levels"]
        # Cell centre coordinates
        xy = self.xuds.grid.face_coordinates
        # Get number of cells
        nr_cells = len(xy)
        # Initialize bathymetry array
        zz = np.full(nr_cells, np.nan)
        # cell size of coarsest level
        dx = self.xuds.attrs["dx"]

        # Determine first indices and number of cells per refinement level
        # This is also done when the grid is built, but that information is not stored
        ifirst = np.zeros(nlev, dtype=int)
        ilast = np.zeros(nlev, dtype=int)
        level = self.xuds["level"].values[:] - 1 # 0-based
        for ilev in range(0, nlev):
            # Find index of first cell with this level
            ifirst[ilev] = np.where(level == ilev)[0][0]
            # Find index of last cell with this level
            if ilev < nlev - 1:
                ilast[ilev] = np.where(level == ilev + 1)[0][0] - 1
            else:
                ilast[ilev] = nr_cells - 1

        # Loop through all levels
        for ilev in range(nlev):

            if not quiet:
                print("Processing bathymetry level " + str(ilev + 1) + " of " + str(nlev) + " ...")

            # First and last cell indices in this level            
            i0 = ifirst[ilev]
            i1 = ilast[ilev]
            
            # Make blocks of cells in this level only
            cell_indices_in_level = np.arange(i0, i1 + 1, dtype=int)
                  
            xz  = xy[cell_indices_in_level, 0]
            yz  = xy[cell_indices_in_level, 1]
            dxmin = dx / 2**ilev

            if self.crs.is_geographic:
                dxmin = dxmin * 111000.0

            zgl = bathymetry_database.get_bathymetry_on_points(xz,
                                                               yz,
                                                               dxmin,
                                                               self.crs,
                                                               bathymetry_sets)
            
            # Limit zgl to zmin and zmax
            zgl = np.maximum(zgl, zmin)
            zgl = np.minimum(zgl, zmax)

            zz[cell_indices_in_level] = zgl

        self.xuds["z"] = xu.UgridDataArray(xr.DataArray(data=zz, dims=[self.xuds.grid.face_dimension]), self.xuds.grid)

    def set_bathymetry_mean_wet(self,
              bathymetry_sets,
              bathymetry_database=None,
              nr_subgrid_pixels=20,
              threshold_level=0.0,
              quiet=True,
              progress_bar=None):  
       
        refi        = nr_subgrid_pixels
        nrmax       = 2000

        # Get some information from the grid
        nr_cells    = self.nr_cells
        nr_ref_levs = self.nr_refinement_levels
        x0          = self.xuds.attrs["x0"]
        y0          = self.xuds.attrs["y0"]
        dx          = self.xuds.attrs["dx"]
        dy          = self.xuds.attrs["dy"]
        rot         = self.xuds.attrs["rotation"]
        cosrot      = np.cos(np.radians(rot))
        sinrot      = np.sin(np.radians(rot))
        level       = self.xuds["level"].values[:] - 1
        n           = self.xuds["n"].values[:] - 1
        m           = self.xuds["m"].values[:] - 1

        counter = 0

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

            if progress_bar:
                progress_bar.set_text("               Computing bathymetry ...                ")
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
                                                                            self.crs,
                                                                            bathymetry_sets)
                        except Exception as e:
                            print(e)
                            pass
                            return

                    if not quiet:
                        print("Processing cells ...")

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

                    # Should really parallelize this loop !

                    for index in index_cells_in_block:

                        # Get elevation in cells
                        nn  = (n[index] - bn0) * refi
                        mm  = (m[index] - bm0) * refi
                        zgc = zg[nn : nn + refi, mm : mm + refi]
    
                        # if np.nanmax(zgc) < threshold_level:
                        #     # Check if any cells above threshold for island
                        #     continue

                        counter += 1

                        # Compute the mean depth of wet pixels
                        indwet = np.where(zgc <= threshold_level)                        
                        if len(indwet[0]) > 0:
                            zbmean = np.nanmean(zgc[indwet])
                        else:
                            zbmean = np.nanmean(zgc)

                        # Set bed level in grid
                        self.xuds["z"].values[index] = zbmean

    def snap_to_grid(self, polyline):
        if len(polyline) == 0:
            return gpd.GeoDataFrame()
        # If geographic coordinates, set max_snap_distance to 0.1 degrees
        if self.crs.is_geographic:
            max_snap_distance = 1.0e-6
        else:
            max_snap_distance = 0.1

        geom_list = []
        for iline, line in polyline.iterrows():
            geom = line["geometry"]
            if geom.geom_type == 'LineString':
                geom_list.append(geom)
        gdf = gpd.GeoDataFrame({'geometry': geom_list})    
        print("Snapping to grid ...")
        snapped_uds, snapped_gdf = xu.snap_to_grid(gdf, self.xuds.grid, max_snap_distance=max_snap_distance)
        print("Snapping to grid done.")
        snapped_gdf = snapped_gdf.set_crs(self.crs)
        return snapped_gdf

    def face_coordinates(self):
        # if self.data is None:
        #     return None, None
        xy = self.xuds.grid.face_coordinates
        return xy[:, 0], xy[:,1]

    def get_exterior(self):
        try:
            indx = self.xuds.grid.edge_node_connectivity[self.xuds.grid.exterior_edges, :]
            x = self.xuds.grid.node_x[indx]
            y = self.xuds.grid.node_y[indx]
            # Make linestrings from numpy arrays x and y
            linestrings = [shapely.LineString(np.column_stack((x[i], y[i]))) for i in range(len(x))]
            # Merge linestrings
            merged = shapely.ops.linemerge(linestrings)
            # Merge polygons
            polygons = shapely.ops.polygonize(merged)
    #        polygons = shapely.simplify(polygons, self.dx)
            self.exterior = gpd.GeoDataFrame(geometry=list(polygons), crs=self.crs)
        except:
            self.exterior = gpd.GeoDataFrame()    

    def bounds(self, crs=None, buffer=0.0):
        """Returns list with bounds (lon1, lat1, lon2, lat2), with buffer (default 0.0) and in any CRS (default : same CRS as model)"""
        if crs is None:
            crs = self.crs
        # Convert exterior gdf to WGS 84
        lst = self.exterior.to_crs(crs=crs).total_bounds.tolist()
        dx = lst[2] - lst[0]
        dy = lst[3] - lst[1]
        lst[0] = lst[0] - buffer * dx
        lst[1] = lst[1] - buffer * dy
        lst[2] = lst[2] + buffer * dx
        lst[3] = lst[3] + buffer * dy
        return lst

    def map_overlay(self, file_name, xlim=None, ylim=None, color="black", width=800):

        if self.xuds is None:
            # No grid (yet)
            return False

        try:
            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            transformer = Transformer.from_crs(4326,
                                        3857,
                                        always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)
            cvs = ds.Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)
            agg = cvs.line(self.datashader_dataframe,
                           x=['x1', 'x2'],
                           y=['y1', 'y2'],
                           axis=1)
            img = tf.shade(agg)
            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True
        except Exception as e:
            return False

    def get_datashader_dataframe(self):
        """Creates a dataframe with line elements for datashader"""
        # Create a dataframe with line elements
        x1 = self.xuds.grid.edge_node_coordinates[:,0,0]
        x2 = self.xuds.grid.edge_node_coordinates[:,1,0]
        y1 = self.xuds.grid.edge_node_coordinates[:,0,1]
        y2 = self.xuds.grid.edge_node_coordinates[:,1,1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.crs.is_geographic:
            if np.max(x1) > 180.0 or np.max(x2) > 180.0:
                cross_dateline = True
        transformer = Transformer.from_crs(self.crs,
                                            3857,
                                            always_xy=True)
        x1, y1 = transformer.transform(x1, y1)
        x2, y2 = transformer.transform(x2, y2)
        if cross_dateline:
            x1[x1 < 0] += 40075016.68557849
            x2[x2 < 0] += 40075016.68557849
        self.datashader_dataframe = pd.DataFrame(dict(x1=x1, y1=y1, x2=x2, y2=y2))

    def clear_datashader_dataframe(self):
        """Clears the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame() 

    def get_indices_at_points(self, x, y):

        # x and y are 2D arrays of coordinates (x, y) in the same projection as the model
        # if x is a float, convert to 2D array
        if np.ndim(x) == 0:
            x = np.array([[x]])
        if np.ndim(y) == 0:
            y = np.array([[y]])

        x0 = self.xuds.attrs["x0"]
        y0 = self.xuds.attrs["y0"]
        dx = self.xuds.attrs["dx"]
        dy = self.xuds.attrs["dy"]
        nmax = self.xuds.attrs["nmax"]
        mmax = self.xuds.attrs["mmax"]
        rotation = self.xuds.attrs["rotation"]
        nr_refinement_levels = self.xuds.attrs["nr_levels"]

        nr_cells = len(self.xuds["level"])

        cosrot = np.cos(-rotation * np.pi / 180)
        sinrot = np.sin(-rotation * np.pi / 180)

        # Now rotate around origin of SFINCS model
        x00 = x - x0
        y00 = y - y0
        xg = x00 * cosrot - y00 * sinrot
        yg = x00 * sinrot + y00 * cosrot

        # Find index of first cell in each level
        if not hasattr(self, "ifirst"):
            ifirst = np.zeros(nr_refinement_levels, dtype=int)
            for ilev in range(0, nr_refinement_levels):
                # Find index of first cell with this level
                ifirst[ilev] = np.where(self.xuds["level"].to_numpy()[:] == ilev + 1)[0][0]
            self.ifirst = ifirst

        ifirst = self.ifirst    

        i0_lev = []
        i1_lev = []
        nmax_lev = []
        mmax_lev = []
        nm_lev = []

        for level in range(nr_refinement_levels):

            i0 = ifirst[level]
            if level < nr_refinement_levels - 1:
                i1 = ifirst[level + 1]
            else:
                i1 = nr_cells
            i0_lev.append(i0)
            i1_lev.append(i1)
            nmax_lev.append(np.amax(self.xuds["n"].to_numpy()[i0:i1]) + 1)
            mmax_lev.append(np.amax(self.xuds["m"].to_numpy()[i0:i1]) + 1)
            nn = self.xuds["n"].to_numpy()[i0:i1] - 1
            mm = self.xuds["m"].to_numpy()[i0:i1] - 1
            nm_lev.append(mm * nmax_lev[level] + nn)

        # Initialize index array
        indx = np.full(np.shape(x), -999, dtype=int)

        for ilev in range(nr_refinement_levels):
            nmax = nmax_lev[ilev]
            mmax = mmax_lev[ilev]
            i0 = i0_lev[ilev]
            i1 = i1_lev[ilev]
            dxr = dx / 2**ilev
            dyr = dy / 2**ilev
            iind = np.floor(xg / dxr).astype(int)
            jind = np.floor(yg / dyr).astype(int)
            # Now check whether this cell exists on this level
            ind = iind * nmax + jind
            ind[iind < 0] = -999
            ind[jind < 0] = -999
            ind[iind >= mmax] = -999
            ind[jind >= nmax] = -999

            ingrid = np.isin(
                ind, nm_lev[ilev], assume_unique=False
            )  # return boolean for each pixel that falls inside a grid cell
            incell = np.where(
                ingrid
            )  # tuple of arrays of pixel indices that fall in a cell

            if incell[0].size > 0:
                # Now find the cell indices
                try:
                    cell_indices = (
                        binary_search(nm_lev[ilev], ind[incell[0], incell[1]])
                        + i0_lev[ilev]
                    )
                    indx[incell[0], incell[1]] = cell_indices
                except Exception:
                    print("Error in binary search")
                    pass

        return indx

    def make_topobathy_cog(self,
                           filename,
                           bathymetry_sets,
                           bathymetry_database=None,
                           dx=10.0):
        
        """Make a COG file with topobathy. This always make the topobathy COG in the same projection as the model."""

        # Get the bounds of the grid
        bounds = self.bounds()

        x0 = bounds[0]
        y0 = bounds[1]
        x1 = bounds[2]
        y1 = bounds[3]

        # Round up and down to nearest dx
        x0 = x0 - (x0 % dx)
        x1 = x1 + (dx - x1 % dx)
        y0 = y0 - (y0 % dx)
        y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx
        zz = np.empty((len(yy), len(xx),), dtype=np.float32)

        xx, yy = np.meshgrid(xx, yy)
        zz = bathymetry_database.get_bathymetry_on_points(xx,
                                                          yy,
                                                          dx,
                                                          self.crs,
                                                          bathymetry_sets)

        # And now to cog (use -999 as the nodata value)
        with rasterio.open(
            filename,
            "w",
            driver="COG",
            height=zz.shape[0],
            width=zz.shape[1],
            count=1,
            dtype=zz.dtype,
            crs=self.crs,
            transform=from_origin(x0, y1, dx, dx),
            nodata=-999.0,
        ) as dst:
            dst.write(zz, 1)

    def make_index_cog(self, filename, filename_topobathy):
    # def make_index_cog(self, filename, dx=10.0):
        """Make a COG file with indices of the quadtree grid cells."""

        # Read coordinates from topobathy file
        with rasterio.open(filename_topobathy) as src:
            # Get the bounds of the grid
            bounds = src.bounds
            dx = src.res[0]
            # Get the CRS of the grid
            self.crs = src.crs
            # Get the nodata value
            nodata = src.nodata
            # Get the transform of the grid
            transform = src.transform
            # Get the width and height of the grid
            width = src.width
            height = src.height

        # Now create numpy arrays with the coordinates of geotiff
        # Get the coordinates of the grid
        x0 = bounds.left
        y0 = bounds.bottom
        x1 = bounds.right
        y1 = bounds.top

        # # Round up and down to nearest dx
        # x0 = x0 - (x0 % dx)
        # x1 = x1 + (dx - x1 % dx)
        # y0 = y0 - (y0 % dx)
        # y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx

        nodata = 2147483647

        # # # Get the bounds of the grid
        # # bounds = self.bounds()

        # x0 = bounds[0]
        # y0 = bounds[1]
        # x1 = bounds[2]
        # y1 = bounds[3]

        # # Round up and down to nearest dx
        # x0 = x0 - (x0 % dx)
        # x1 = x1 + (dx - x1 % dx)
        # y0 = y0 - (y0 % dx)
        # y1 = y1 + (dx - y1 % dx)

        xx = np.arange(x0, x1, dx) + 0.5 * dx
        yy = np.arange(y1, y0, -dx) - 0.5 * dx
        ii = np.empty((len(yy), len(xx),), dtype=np.uint32)

        # # Create empty ds
        # ds = xr.Dataset(
        #     {
        #         "index": (["y", "x"], ii),
        #     },
        #     coords={
        #         "x": xx,
        #         "y": yy,
        #     },
        # )
        # # Set no data value in ds
        # ds["index"].attrs["_FillValue"] = nodata

        # Go through refinement levels in grid
        xx, yy = np.meshgrid(xx, yy)
        indices = self.get_indices_at_points(xx, yy)
        indices[np.where(indices == -999)] = nodata

        # Fill the array with indices
        ii[:, :] = indices        

        # # Write first to netcdf
        # ds.to_netcdf("index.nc")

        # And now to cog (use -999 as the nodata value)
        with rasterio.open(
            filename,
            "w",
            driver="COG",
            height=height,
            width=width,
            count=1,
            dtype=ii.dtype,
            crs=self.crs,
            transform=transform,
            nodata=nodata,
            overview_resampling=Resampling.nearest,
        ) as dst:
            dst.write(ii, 1)


class QuadtreeMask:
    def __init__(self, xuds, mask_name="mask"):
        """
        Initialize the QuadtreeMask with an xugrid dataset and a mask name.
        
        Parameters
        ----------
        xugrid_dataset : xu.UgridDataset
            The xugrid dataset containing the grid and mask.
        """
        self.xuds = xuds
        self.mask_name = mask_name
        self.datashader_dataframe = pd.DataFrame()

        if xuds is None:
            # Grid is still empty
            return

        crd_dict = self.xuds["crs"].attrs

        crs = CRS.from_epsg(4326)  # Default to WGS84
        if "projected_crs_name" in crd_dict:
            crs = CRS(crd_dict["projected_crs_name"])
        elif "geographic_crs_name" in crd_dict:
            crs = CRS(crd_dict["geographic_crs_name"])
        else:
            print("Could not find CRS in quadtree netcdf file. Assuming WGS84.")

        self.crs = crs

        # Initialize mask
        nr_cells = self.xuds.sizes["mesh2d_nFaces"]

        # Set all to zero
        self.xuds[mask_name] = xr.DataArray(data=np.zeros(nr_cells, dtype=np.int8), dims=["mesh2d_nFaces"])

    def set_to_zero(self):
        """Set all mask values to zero"""
        nr_cells = self.xuds.sizes["mesh2d_nFaces"]
        self.xuds[self.mask_name] = xr.DataArray(data=np.zeros(nr_cells, dtype=np.int8), dims=["mesh2d_nFaces"])

    def set_global(self, zmin, zmax, mask_value):

        mask = self.xuds[self.mask_name].values[:]
        z = self.xuds["z"].values[:]

        if z is not None:                
            # Set initial mask based on zmin and zmax
            iok = np.where((z>=zmin) & (z<=zmax))
            mask[iok] = mask_value
        else:
            # No data so set all to 0
            mask[:] = 0

        self.xuds[self.mask_name] = xr.DataArray(data=mask, dims=["mesh2d_nFaces"])

    def set_internal_polygons(self, polygon, zmin, zmax, mask_value):

        if polygon is None:
            return

        xy = self.xuds.grid.face_coordinates
        x    = xy[:,0]
        y    = xy[:,1]
        z    = self.xuds["z"].values[:]
        mask = self.xuds[self.mask_name].values[:]

        for ip, polygon in polygon.iterrows():
            inpol = inpolygon(x, y, polygon["geometry"])
            iok   = np.where((inpol) & (z>=zmin) & (z<=zmax))
            mask[iok] = mask_value

        self.xuds[self.mask_name] = xr.DataArray(data=mask, dims=["mesh2d_nFaces"])

    def set_boundary_polygons(self, polygon, zmin, zmax, mask_value):
        """Set the mask value for a given polygon"""

        if polygon is None:
            return

        xy = self.xuds.grid.face_coordinates
        x    = xy[:,0]
        y    = xy[:,1]
        z    = self.xuds["z"].values[:]
        mask = self.xuds[self.mask_name].values[:]

        # Indices are 1-based in SFINCS so subtract 1 for python 0-based indexing
        mu    = self.xuds["mu"].values[:]
        mu1   = self.xuds["mu1"].values[:] - 1
        mu2   = self.xuds["mu2"].values[:] - 1
        nu    = self.xuds["nu"].values[:]
        nu1   = self.xuds["nu1"].values[:] - 1
        nu2   = self.xuds["nu2"].values[:] - 1
        md    = self.xuds["md"].values[:]
        md1   = self.xuds["md1"].values[:] - 1
        md2   = self.xuds["md2"].values[:] - 1
        nd    = self.xuds["nd"].values[:]
        nd1   = self.xuds["nd1"].values[:] - 1
        nd2   = self.xuds["nd2"].values[:] - 1

        for ip, polygon in polygon.iterrows():
            inpol = inpolygon(x, y, polygon["geometry"])
            # Only consider points that are:
            # 1) Inside the polygon
            # 2) Have a mask > 0
            # 3) z>=zmin
            # 4) z<=zmax
            iok   = np.where((inpol) & (mask>0) & (z>=zmin) & (z<=zmax))
            for ic in iok[0]:
                okay = False
                # Check neighbors, cell must have at least one inactive neighbor
                # Left
                if md[ic]<=0:
                    # Coarser or equal to the left
                    if md1[ic]>=0:
                        # Cell has neighbor to the left
                        if mask[md1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer to the left
                    if md1[ic]>=0:
                        # Cell has neighbor to the left
                        if mask[md1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if md2[ic]>=0:
                        # Cell has neighbor to the left
                        if mask[md2[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    
                # Below
                if nd[ic]<=0:
                    # Coarser or equal below
                    if nd1[ic]>=0:
                        # Cell has neighbor below
                        if mask[nd1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer below
                    if nd1[ic]>=0:
                        # Cell has neighbor below
                        if mask[nd1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if nd2[ic]>=0:
                        # Cell has neighbor below
                        if mask[nd2[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True

                # Right
                if mu[ic]<=0:
                    # Coarser or equal to the right
                    if mu1[ic]>=0:
                        # Cell has neighbor to the right
                        if mask[mu1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer to the left
                    if mu1[ic]>=0:
                        # Cell has neighbor to the right
                        if mask[mu1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if mu2[ic]>=0:
                        # Cell has neighbor to the right
                        if mask[mu2[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True

                # Above
                if nu[ic]<=0:
                    # Coarser or equal above
                    if nu1[ic]>=0:
                        # Cell has neighbor above
                        if mask[nu1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                else:
                    # Finer below
                    if nu1[ic]>=0:
                        # Cell has neighbor above
                        if mask[nu1[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    if nu2[ic]>=0:
                        # Cell has neighbor above
                        if mask[nu2[ic]]==0:
                            # And it's inactive
                            okay = True
                    else:
                        # No neighbor, so set mask = 2
                        okay = True
                    
                if okay:
                    mask[ic] = mask_value

        self.xuds[self.mask_name] = xr.DataArray(data=mask, dims=["mesh2d_nFaces"])

    def to_gdf(self, option="all"):

        nr_cells = self.xuds.sizes["mesh2d_nFaces"]

        if nr_cells == 0:
            # Return empty geodataframe
            return gpd.GeoDataFrame()


        xz = self.xuds.grid.face_coordinates[:,0]
        yz = self.xuds.grid.face_coordinates[:,1]
        mask = self.xuds[self.mask_name].values[:] 

        gdf_list = []
        okay = np.zeros(mask.shape, dtype=int)
        if option == "all":
            iok = np.where((mask > 0))
        elif option == "include":
            iok = np.where((mask == 1))
        elif option == "open":
            iok = np.where((mask == 2))
        elif option == "outflow":
            iok = np.where((mask == 3))
        else:
            iok = np.where((mask > -999))
        okay[iok] = 1
        for icel in range(nr_cells):
            if okay[icel] == 1:
                point = shapely.geometry.Point(xz[icel], yz[icel])
                d = {"geometry": point}
                gdf_list.append(d)

        if gdf_list:
            gdf = gpd.GeoDataFrame(gdf_list, crs=self.crs)
        else:
            # Cannot set crs of gdf with empty list
            gdf = gpd.GeoDataFrame(gdf_list)

        return gdf

    def get_datashader_dataframe(self):
        """Create a dataframe with points elements for use with datashader"""

        if self.xuds is None:
            # Grid is still empty
            return

        # Coordinates of cell centers
        x = self.xuds.grid.face_coordinates[:,0]
        y = self.xuds.grid.face_coordinates[:,1]
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.crs.is_geographic:
            if np.max(x) > 180.0:
                cross_dateline = True
        mask = self.xuds[self.mask_name].values[:] 
        # Get rid of cells with mask = 0
        iok = np.where(mask>0)
        x = x[iok]
        y = y[iok]
        mask = mask[iok]
        if np.size(x) == 0:
            # Set empty dataframe
            self.datashader_dataframe = pd.DataFrame()
            return
        # Transform all to 3857 (web mercator)
        transformer = Transformer.from_crs(self.crs,
                                            3857,
                                            always_xy=True)
        x, y = transformer.transform(x, y)
        if cross_dateline:
            x[x < 0] += 40075016.68557849

        self.datashader_dataframe = pd.DataFrame(dict(x=x, y=y, mask=mask))

    def clear_datashader_dataframe(self):
        """Clear the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame()

    def map_overlay(self,
                    file_name,
                    xlim=None,
                    ylim=None,
                    active_color="yellow",
                    boundary_color="red",
                    downstream_color="blue",
                    neumann_color="purple",
                    outflow_color="green",
                    px=2,
                    width=800):

        if self.xuds is None:
            # No mask points (yet)
            return False

        try:

            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            # If it is still empty (because there are no active cells), return False    
            if self.datashader_dataframe.empty:
                return False

            transformer = Transformer.from_crs(4326,
                                        3857,
                                        always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)

            cvs = ds.Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)

            # Should update colors at some point. Now SFINCS specific, but it should work for HurryWave and SnapWave as well. 

            # Instead, we can create separate images for each mask and stack them
            dfact = self.datashader_dataframe[self.datashader_dataframe["mask"]==1]
            dfbnd = self.datashader_dataframe[self.datashader_dataframe["mask"]==2]
            dfout = self.datashader_dataframe[self.datashader_dataframe["mask"]==3]
            dfneu = self.datashader_dataframe[self.datashader_dataframe["mask"]==5]
            dfdwn = self.datashader_dataframe[self.datashader_dataframe["mask"]==6]
            img_a = tf.shade(tf.spread(cvs.points(dfact, 'x', 'y', ds.any()), px=px), cmap=active_color)
            img_b = tf.shade(tf.spread(cvs.points(dfbnd, 'x', 'y', ds.any()), px=px), cmap=boundary_color)
            img_o = tf.shade(tf.spread(cvs.points(dfout, 'x', 'y', ds.any()), px=px), cmap=outflow_color)
            img_n = tf.shade(tf.spread(cvs.points(dfneu, 'x', 'y', ds.any()), px=px), cmap=neumann_color)
            img_d = tf.shade(tf.spread(cvs.points(dfdwn, 'x', 'y', ds.any()), px=px), cmap=downstream_color)
            img   = tf.stack(img_a, img_b, img_o, img_n, img_d)

            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True

        except Exception as e:
            print(e)
            return False


class QuadtreeBathymetry:
    def __init__(self, xuds):
        """
        Initialize the QuadtreeBathymetry with an xugrid dataset.
        
        Parameters
        ----------
        xugrid_dataset : xu.UgridDataset
            The xugrid dataset containing the grid and mask.
        """
        self.xuds = xuds
        self.datashader_dataframe = pd.DataFrame()

        if xuds is None:
            # Grid is still empty
            return

        crd_dict = self.xuds["crs"].attrs

        crs = CRS.from_epsg(4326)  # Default to WGS84
        if "projected_crs_name" in crd_dict:
            crs = CRS(crd_dict["projected_crs_name"])
        elif "geographic_crs_name" in crd_dict:
            crs = CRS(crd_dict["geographic_crs_name"])
        else:
            print("Could not find CRS in quadtree netcdf file. Assuming WGS84.")

        self.crs = crs

    def get_datashader_dataframe(self):
        """Create a dataframe with points elements for use with datashader"""
        # Coordinates of cell centers
        x = self.xuds.grid.face_coordinates[:,0]
        y = self.xuds.grid.face_coordinates[:,1]
        z = self.xuds["z"].values[:] 
        # Check if grid crosses the dateline
        cross_dateline = False
        if self.crs.is_geographic:
            if np.max(x) > 180.0:
                cross_dateline = True
        # Get rid of cells with mask = 0
        # iok = np.where(mask>0)
        # x = x[iok]
        # y = y[iok]
        # z = z[iok]
        if np.size(x) == 0:
            # Set empty dataframe
            self.datashader_dataframe = pd.DataFrame()
            return
        # Transform all to 3857 (web mercator)
        transformer = Transformer.from_crs(self.crs,
                                            3857,
                                            always_xy=True)
        x, y = transformer.transform(x, y)
        if cross_dateline:
            x[x < 0] += 40075016.68557849

        self.datashader_dataframe = pd.DataFrame(dict(x=x, y=y, z=z))

    def clear_datashader_dataframe(self):
        """Clear the datashader dataframe"""
        self.datashader_dataframe = pd.DataFrame()

    def map_overlay(self,
                    file_name,
                    xlim=None,
                    ylim=None,
                    active_color="yellow",
                    boundary_color="red",
                    downstream_color="blue",
                    neumann_color="purple",
                    outflow_color="green",
                    px=2,
                    width=800):

        if self.xuds is None:
            # No mask points (yet)
            return False

        try:

            # Check if datashader dataframe is empty (maybe it was not made yet, or it was cleared)
            if self.datashader_dataframe.empty:
                self.get_datashader_dataframe()

            # If it is still empty (because there are no active cells), return False    
            if self.datashader_dataframe.empty:
                return False

            transformer = Transformer.from_crs(4326,
                                        3857,
                                        always_xy=True)
            xl0, yl0 = transformer.transform(xlim[0], ylim[0])
            xl1, yl1 = transformer.transform(xlim[1], ylim[1])
            if xl0 > xl1:
                xl1 += 40075016.68557849
            xlim = [xl0, xl1]
            ylim = [yl0, yl1]
            ratio = (ylim[1] - ylim[0]) / (xlim[1] - xlim[0])
            height = int(width * ratio)

            cvs = ds.Canvas(x_range=xlim, y_range=ylim, plot_height=height, plot_width=width)

            # Should update colors at some point. Now SFINCS specific, but it should work for HurryWave and SnapWave as well. 

            # Instead, we can create separate images for each mask and stack them
            dfact = self.datashader_dataframe[self.datashader_dataframe["mask"]==1]
            dfbnd = self.datashader_dataframe[self.datashader_dataframe["mask"]==2]
            dfout = self.datashader_dataframe[self.datashader_dataframe["mask"]==3]
            dfneu = self.datashader_dataframe[self.datashader_dataframe["mask"]==5]
            dfdwn = self.datashader_dataframe[self.datashader_dataframe["mask"]==6]
            img_a = tf.shade(tf.spread(cvs.points(dfact, 'x', 'y', ds.any()), px=px), cmap=active_color)
            img_b = tf.shade(tf.spread(cvs.points(dfbnd, 'x', 'y', ds.any()), px=px), cmap=boundary_color)
            img_o = tf.shade(tf.spread(cvs.points(dfout, 'x', 'y', ds.any()), px=px), cmap=outflow_color)
            img_n = tf.shade(tf.spread(cvs.points(dfneu, 'x', 'y', ds.any()), px=px), cmap=neumann_color)
            img_d = tf.shade(tf.spread(cvs.points(dfdwn, 'x', 'y', ds.any()), px=px), cmap=downstream_color)
            img   = tf.stack(img_a, img_b, img_o, img_n, img_d)

            path = os.path.dirname(file_name)
            if not path:
                path = os.getcwd()
            name = os.path.basename(file_name)
            name = os.path.splitext(name)[0]
            export_image(img, name, export_path=path)
            return True

        except Exception as e:
            print(e)
            return False

def odd(num):
    return np.mod(num, 2) == 1

def even(num):
    return np.mod(num, 2) == 0

def inpolygon(xq, yq, poly):
    coords = np.column_stack((xq.ravel(), yq.ravel()))
    pts = shapely.points(coords)
    inside = shapely.contains(poly, pts)   # vectorized
    return inside.reshape(xq.shape)

# def inpolygon(xq, yq, p): # p is a Polygon object, xq and yq are arrays of x and y coordinates  
#     shape = xq.shape
#     xq = xq.reshape(-1)
#     yq = yq.reshape(-1)
#     # Create list of points in tuples
#     q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
#     # Create list with inout logicals (starting with False)
#     inp = [False for i in range(xq.shape[0])]
#     # Now start with exterior
#     # Check if point is in exterior
#     pth = path.Path([(crds[0], crds[1]) for i, crds in enumerate(p.exterior.coords)])
#     # Check if point is in exterior
#     inext = pth.contains_points(q).astype(bool)
#     # Set inp to True where inext is True
#     inp = np.logical_or(inp, inext)
#     # Check if point is in interior
#     for interior in p.interiors:
#         pth = path.Path([(crds[0], crds[1]) for i, crds in enumerate(interior.coords)])
#         inint = pth.contains_points(q).astype(bool)
#         inp = np.logical_xor(inp, inint)
#     # inp = inexterior(q, p, inp)
#     return inp.reshape(shape)

def binary_search(val_array, vals):    
    indx = np.searchsorted(val_array, vals) # ind is size of vals 
    not_ok = np.where(indx==len(val_array))[0] # size of vals, points that are out of bounds
    indx[np.where(indx==len(val_array))[0]] = 0 # Set to zero to avoid out of bounds error
    is_ok = np.where(val_array[indx] == vals)[0] # size of vals
    indices = np.zeros(len(vals), dtype=int) - 1
    indices[is_ok] = indx[is_ok]
    indices[not_ok] = -1
    return indices

def interp2(x0, y0, z0, x1, y1, method="linear"):
    
    f = RegularGridInterpolator((y0, x0), z0,
                                bounds_error=False, fill_value=np.nan, method=method)
    # reshape x1 and y1
    if x1.ndim>1:
        sz = x1.shape
        x1 = x1.reshape(sz[0]*sz[1])
        y1 = y1.reshape(sz[0]*sz[1])    
        # interpolate
        z1 = f((y1,x1)).reshape(sz)        
    else:    
        z1 = f((y1,x1))
    
    return z1

def grid_in_polygon(x, y, p):

    # Dimensions of the cells
    rows, cols = x.shape[0] - 1, x.shape[1] - 1

    # Create polygons for each cell
    x1 = x[:-1, :-1].flatten()
    y1 = y[:-1, :-1].flatten()
    x2 = x[1:, 1:].flatten()
    y2 = y[1:, 1:].flatten()
    x3 = x[:-1, 1:].flatten()
    y3 = y[:-1, 1:].flatten()
    x4 = x[1:, :-1].flatten()
    y4 = y[1:, :-1].flatten()

    # Prepare the list of cell polygons
    cell_polygons = np.array([
        Polygon([(x1[i], y1[i]), (x3[i], y3[i]), (x2[i], y2[i]), (x4[i], y4[i])])
        for i in range(len(x1))
    ])

    # Prepare the polygon for faster intersections
    prepared_p = prep(p)

    # Vectorized intersection checks
    inp = np.array([prepared_p.intersects(cell) for cell in cell_polygons])

    # Reshape the result back to the grid shape
    inp = inp.reshape(rows, cols)

    return inp

# def binary_search(val_array, vals):
#     indx = np.searchsorted(val_array, vals)  # ind is size of vals
#     not_ok = np.where(indx == len(val_array))[
#         0
#     ]  # size of vals, points that are out of bounds
#     indx[np.where(indx == len(val_array))[0]] = (
#         0  # Set to zero to avoid out of bounds error
#     )
#     is_ok = np.where(val_array[indx] == vals)[0]  # size of vals
#     indices = np.zeros(len(vals), dtype=int) - 1
#     indices[is_ok] = indx[is_ok]
#     indices[not_ok] = -1
#     return indices
