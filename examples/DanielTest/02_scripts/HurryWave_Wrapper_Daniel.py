import matplotlib.pyplot as plt
import os
import numpy as np
import xugrid as xu
import geopandas as gpd
from shapely.geometry import Polygon

from cht_hurrywave.hurrywave import HurryWave
from cht_hurrywave.grid import HurryWaveGrid
# note, we don't use the Mask class from the mask.py nor the grid from grid.py! Instead, both from hurrywave_domain
from cht_hurrywave.observation_points import HurryWaveObservationPointsRegular, HurryWaveObservationPointsSpectra
from cht_model_builder.model_builder import MaskPolygon
from cht_utils.pli_file import read_pli_file
from cht_meteo.meteo import MeteoGrid
from pyproj import Transformer


# PATHS
main_path  = os.path.join(r'C:\Users\User\OneDrive\Documents\Python\PYTHON_MSC_CE\Year_2\Python_Thesis\cht_hurrywave\examples\DanielTest') 
if not os.path.exists(main_path):
    os.mkdir(main_path)

name = '21_25_April_2025_dx_0point15'
data_name = 'NorthSea_April2025'

model_setup = os.path.join(main_path, '04_modelruns', name)

data_path = os.path.join(main_path, '01_data')

path_to_exe = r'C:\Users\User\OneDrive\Documents\Python\PYTHON_MSC_CE\Year_2\Python_Thesis\cht_hurrywave\examples\DanielTest\06_executables\hurrywave\hurrywave.exe'
botfile = r'C:\Users\User\OneDrive\Documents\Python\PYTHON_MSC_CE\Year_2\Python_Thesis\cht_hurrywave\examples\DanielTest\01_data\SWAN_Noordzee.bot'

if not os.path.exists(model_setup):
    os.mkdir(model_setup)

print('Model setup path:', model_setup)
hw = HurryWave(path=model_setup)

# ERA 5 data path wind
era_5_file_wind = os.path.join(data_path,'ERA_5_data', data_name, f'{data_name}_era5_data.nc')
era_5_data_wind = data = xu.open_dataset(era_5_file_wind)

# Configure inputs
surge = 0 # Add a surge to the water level [m]

specs = {
    'mmax': 421,                    # Number of grid points in x-direction
    'nmax': 481,                    # Number of grid points in y-direction
    'dx': 0.15,                     # Grid spacing in x-direction [degrees]
    'dy': 0.15,              # Grid spacing in y-direction [degrees]
    'x0': -12,                      # X-coordinate of the first grid cell corner (1,1) in projected UTM zone [m]
    'y0': 48,                       # Y-coordinate of the first grid cell corner (1,1) in projected UTM zone [m]
    'rotation': 0.0,                # Grid rotation from x-axis in anti-clockwise direction [degrees]
    'latitude': 0.0,

    'dt': 90,                      # Time-step size [s]
    'dtwnd': 1800.0,                # Time-interval for wind update [s]

    'tstart': "20250421 000000",    # Start date [YYYYMMDD HHMMSS]
    'tstop': "20250425 235959",     # Stop date [YYYYMMDD HHMMSS]
    'tspinup': 1800.0,              # Spin-up duration [s]

    'dtmapout': 1800,               # Time-step for map output [s] - NOT IN DOCS
    'dthisout': 3600,                # Time-step for observation points output [s]

    # Physical parameters
    'rhoa': 1.25,                   # Air density [kg/m³]
    'rhow': 1024.0,                 # Water density [kg/m³]
    'fbed': 0.019,                  # Bed friction coefficient - NOT IN SPECS
    'cdcap': 0.0025,                # Cap on drag coefficient
    'winddrag': "zijlema",          # Wind drag formulation ('zijlema' or 'wu')
    'vmax_zijlema': 50.0,           # Max velocity for Zijlema wind drag - NOT IN SPECS

    # Wave parameters
    'dmx1': 0.2,                    # Initial x-direction dispersion
    'dmx2': 1e-05,                  # Final x-direction dispersion
    'freqmin': 0.04,                # Minimum frequency [Hz]
    'freqmax': 0.5,                 # Maximum frequency [Hz]
    'nsigma': 12,                   # Number of frequency bins
    'ntheta': 36,                   # Number of directional bins

    # Model master parameters
    'inputformat': "bin",           # Input format
    'outputformat': "net",          # Output format
    'quadruplets': 1,            # Flag for quadruplets in wave modeling [Boolean]
    'spinup_meteo': 1,           # Meteorological spin-up flag [Boolean] 
    'redopt': 1,                    # Redundancy option


    # Coordinate reference system (CRS) parameters
    'crs_name': "ETRS 89",           # CRS name
    # CRS type
    'crs_epsg': 4258,               # EPSG code for ETRS89
    'crsgeo': 0,                    # Geographic CRS flag - NOT IN DOCS
    'crs_utmzone': 'nil',           # UTM zone (if applicable) - NOT IN SPECS

    # Configuration files
    'depfile': "hurrywave.dep",       # Elevation (bathymetry and topography) at grid cell centres above reference level [m]
    'mskfile': "hurrywave.msk",       # Mask for inactive (0), active (1), boundary (2), or outflow (3) grid cells
    'obsfile': "hurrywave.obs",       # Observation points file for output time-series at specific locations
    'amufile': "hurrywave.amu",            # Delft3D-meteo ASCII wind x-component [m/s]
    'amvfile': "hurrywave.amv",            # Delft3D-meteo ASCII wind y-component [m/s]

}

const_length_x = 0.05 * 421 # # Length of the domain in x-direction [degrees]
const_length_y = 0.033333333 * 481 # Length of the domain in y-direction [degrees]

# print('Length of the domain in x-direction:', const_length_x, 'degrees')
# print('Length of the domain in y-direction:', const_length_y, 'degrees')

def change_space_resolution_x(new_dx, specs, const_length_x=const_length_x):
    """
    Change the space resolution in the x-direction of the model.
    :param hw: HurryWave object
    :param new_dx: New space resolution in the x-direction
    :return: New mmax (number of grid points in x) value and updates input
    """
    new_mmax = int(const_length_x / new_dx)

    specs['dx'] = new_dx
    specs['mmax'] = new_mmax

def change_space_resolution_y(new_dy, specs, const_length_y=const_length_y):
    """
    Change the space resolution in the x-direction of the model.
    :param hw: HurryWave object
    :param new_dx: New space resolution in the x-direction
    :return: New mmax (number of grid points in x) value and updates input
    """
    new_nmax = int(const_length_y / new_dy)

    specs['dy'] = new_dy
    specs['nmax'] = new_nmax

def change_start_time(new_start_time):
    """
    Change the start time of the model.
    :param hw: HurryWave object
    :param new_start_time: New start time in the format 'YYYYMMDD HHMMSS'
    :return: None
    """

    specs['tstart'] = new_start_time
    specs['tref'] = new_start_time
    

def update_input(hw,specs, const_length_x=const_length_x, const_length_y=const_length_y):
    """
    Update the input file of the model with the new parameters.
    :param hw: HurryWave object
    :return: None
    """
    change_space_resolution_x(specs['dx'], specs, const_length_x)
    change_space_resolution_y(specs['dy'], specs, const_length_y)
    change_start_time(specs['tstart'])
    # Update the input file with the new parameters
    hw.input.update(specs)
    hw.input.write()

update_input(hw, specs,const_length_x=const_length_x, const_length_y=const_length_y)

# Observation points

# Define the observation points for the model run

Use_custom_obs = True
observation_points_regular = HurryWaveObservationPointsRegular(hw)
observation_points_spectra = HurryWaveObservationPointsSpectra(hw)

def get_lat_lon_values_from_data(data):
    """
    Get the latitude and longitude values from the dataset.
    """
    lat = data.latitude.values if 'latitude' in data.dims else None
    lon = data.longitude.values if 'longitude' in data.dims else None
    return lat, lon

latitudes,longitudes = get_lat_lon_values_from_data(data)
counter = 0 
if Use_custom_obs:
    custom_x = [3.27503678, 2.93575, 4.15028575, 1.166099, 3.218932, 4.01222222, 4.05698307]
    custom_y = [51.99779895, 54.32566667, 52.92535269, 61.338188, 53.21701, 54.11666667, 52.54921399]

    # Define transformer from WGS84 (EPSG:4326) to ETRS89 (EPSG:4258)
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4258", always_xy=True)
    custom_x_etrs89, custom_y_etrs89 = transformer.transform(custom_x, custom_y)
    custom_names = ['Euro platform','Platform D15-A','Platform Hoorn Q1-A','North Cormorant','K13 Alpha','Platform F16-A','IJmuiden munitiestortplaats']
    
    # Write to file
    obs_file = os.path.join(model_setup, 'hurrywave.obs')
    with open(obs_file, 'w') as f:
        for lon, lat, name in zip(custom_x_etrs89, custom_y_etrs89, custom_names):
            f.write(f"{lon:.6f} {lat:.6f} # {name}\n")

      
else:
    obs_file = os.path.join(model_setup, 'hurrywave.obs')

    # Add the wave buoy measurements
    observation_points_regular.write()
    observation_points_spectra.write()

# Bottom grid
x0_b = -15
y0_b = 43
m_bot = 1120
n_bot = 1260
dx_bot = 0.025
dy_bot = 0.0166

x_bot = np.linspace(x0_b, x0_b + m_bot * dx_bot , m_bot + 1)
y_bot = np.linspace(y0_b, y0_b + n_bot * dy_bot , n_bot + 1)


xgr_bot, ygr_bot = np.meshgrid(x_bot, y_bot)

def read_swn_bot(filePath, idla=3, missing_value = 999):
    
    if idla==3:
        fid = open(filePath)
        dep = []
        while True:
            line = fid.readline()

            if (''==line):
                # print('eof')
                break
            
            dep.append([float(x) for x in line[:-1].split()])

        dep = np.array(dep)
        dep[dep==missing_value] = np.nan
        return dep
    else:
        print('other idla\'s not yet scripted')


dep = read_swn_bot(botfile, idla=3)
dep = dep * -1 

#%% create grid
grid = HurryWaveGrid(hw)

hw.grid = grid

hw.grid.set_bathymetry_from_other_source(xgr_bot, ygr_bot, dep, rectilinearSourceData=False)

hw.grid.write_dep_file()

#% Mask 
# Initialize polygons as None
include_polygon = None
exclude_polygon = None
boundary_polygon = None

# Write open boundary polygon to file
open_bound_file = os.path.join(model_setup, "open_bound.pli")
with open(open_bound_file, "w") as f:
    f.write("boundaries_1\n")
    f.write("8  2\n")
    f.write("-5.00  47.0\n")
    f.write("-5.00  48.5\n")
    f.write("-11.0  48.5\n")
    f.write("-11.0  63.5\n")
    f.write("10.0  63.5\n")
    f.write("10.0  64.5\n")
    f.write("-14.0  64.5\n")
    f.write("-14.0  47.0\n")


open_boundary_polygon = os.path.join(model_setup, "open_bound.pli")


polygons = read_pli_file(open_boundary_polygon)
polygon_list = []

for polygon in polygons:
    polygon_list.append(MaskPolygon(x=polygon.x,
                            y=polygon.y,
                            zmin=-9999,
                            zmax=1) ) 

# Convert polygon_list to a GeoDataFrame
polygon_geometries = [Polygon(zip(polygon.x, polygon.y)) for polygon in polygons]
polygon_gdf = gpd.GeoDataFrame({'geometry': polygon_geometries}, crs="WGS 84")
hw.grid.build_mask(
            zmin= -9999,
            zmax=0, 
            boundary_polygon= polygon_gdf
            )


hw.grid.write_msk_file()

old_bathy = hw.grid.ds["bed_level"].values

def change_bathy(hw,surge, old_bathy, plot = True):
    """
    Change the bathymetry of the model based on surge or tide
    :param hw: HurryWave object
    :param surge: Surge value to be added to the bathymetry
    :param old_bathy: Old bathymetry to be modified
    :param plot: Boolean to plot the new bathymetry
    :return: new bathymetry
    """
    surge = 0
    surge = surge * -1 # Convert to bottom depth change

    mask = hw.grid.ds.mask.values

    # Add the surge to the bathymetry on the active grid points
    new_bathy = old_bathy.copy()
    new_bathy[mask == 1] = old_bathy[mask == 1] + surge

    # Set the new bathymetry in the model
    hw.grid.ds["bed_level"].values = new_bathy
    hw.grid.write_dep_file()


    if plot:
        fig, ax = plt.subplots(1,1, figsize = (10,7))
        ax.set_aspect("equal")
        im = hw.grid.ds["bed_level"].plot(ax=ax, vmin=-5000, vmax=9000)

    return new_bathy

new_bathy = change_bathy(hw, surge, old_bathy)

meteo_grid = MeteoGrid(name = "era5",
                       source="ECMWF",
                       parameters=["wind"],
                       path=era_5_file_wind,
                       x_range=[-11,11],
                       y_range=[48,65],
                       xystride=1,
                       tstride=1)

meteo_grid.collect_based_netcdf(path = era_5_file_wind)


meteo_grid.write_to_delft3d(os.path.join(hw.path, "hurrywave"))

hw.input.variables.amufile = "hurrywave.amu"
hw.input.variables.amvfile = "hurrywave.amv"

#%% write the total model to file
hw.write()

# Add the batch file to the folder
def create_batch_file(path_to_exe):
    """
    Create a batch file to run the model.
    :param path_to_exe: Path to the hurrywave executable
    :return: None
    """
    batch_file = os.path.join(model_setup, "run.bat")
    with open(batch_file, 'w') as f:
        f.write(f"@echo off\n")
        f.write(f"cd {model_setup}\n")
        f.write(path_to_exe + "\n")
        f.write(f"exit\n")

create_batch_file(path_to_exe)
print("Finished creating input files")
