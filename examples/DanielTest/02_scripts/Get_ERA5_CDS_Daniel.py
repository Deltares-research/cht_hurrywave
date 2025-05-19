#%% Import packages
import numpy as np
import os
import cdsapi
import sys
import xarray as xr
# https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-complete?tab=form alternative to the old ecmwf
#%% Specify file name and beginning/end year
area = [65, -12, 48, 10] # DCSM area in the North Sea (degrees): North, West, South, East
area_name = 'NorthSea' # name of the area (for naming)
yi = 2013 # year start of the data
ye = 2013 # year end of the data
mi = 12 # month start of the data
me = 12 # month end of the data
di = 1 # day start of the data
de = 8 # day end of the data
hi = 0 # hour start of the data
he = 23 # hour end of the data

#%% Specify variables to download (find them in the ERA5 documentation or the CDS API web interface)
variables = ["10m_u_component_of_wind", "10m_v_component_of_wind","mean_wave_direction","mean_wave_period","peak_wave_period","significant_height_of_combined_wind_waves_and_swell"] # list of variables to download


# name is here the name you want to give
# main_path_download is the path where you want to download the data (make sure it is up the tree to avoid too long filepaths)
# final_data_path is the path where you want to save the final data

name = 'SInterklaas_NorthSea' # name of the data (for naming)
main_path_download = r'C:\Users\User\OneDrive\Documents\Python\PYTHON_MSC_CE\Year_2\Python_Thesis\ERA5_data_downloaded' # path to the download (make sure it is up the tree to avoid long filepaths)
final_data_path = r'C:\Users\User\OneDrive\Documents\Python\PYTHON_MSC_CE\Year_2\Python_Thesis\cht_hurrywave\examples\DanielTest\01_data\ERA_5_data' # path to the final data

# Create outpath if it doesnt exist
outpath = os.path.join(main_path_download, name)
os.makedirs(outpath, exist_ok=True)

# Create final_path if it doesn't exist
final_path = os.path.join(final_data_path, name)
os.makedirs(final_path, exist_ok=True)

res = 0.25 # resolution in degrees
years = ['%4d'%(ii) for ii in np.arange(yi,ye+1)]
months = ['%02d'%(ii) for ii in np.arange(mi,me+1)]
days = ['%02d'%(ii) for ii in np.arange(di,de+1)]
hours = ['%02d:00'%(ii) for ii in np.arange(hi,he+1)]

#################################### Download the files #############################################################

c = cdsapi.Client() # Make sure you installed the cdsapi package (pip install cdsapi) and obtained the key placed in C:\Users\User\.cdsapirc
catalogue = 'reanalysis-era5-single-levels' # or 'reanalysis-era5-pressure-levels'
product = 'reanalysis'
fmat = 'netcdf' # or 'grib' (grib is the default format, but netcdf is easier to work with in Python)

#%% Download data
for vv in variables:
    # create directory for each variable
    varpath = os.path.join(outpath,vv)
    os.makedirs(varpath, exist_ok=True)
    
    # Create one NetCDF file for each variable and year
    for yy in years:
        print('%s-%s, %s-%4d'%(area_name,vv,yy,ye))
        filename = '%s_%s_%s.nc'%(area_name,vv,yy)
        print(filename)
        data = c.retrieve(
            catalogue,{
                'product_type'  : product,
                'variable'      : vv,
                'year'          : yy,
                'month'         : months,
                'day'           : days,
                'area'          : area, # North, West, South, East. Default: global
                #'grid'          : [res, res], # Latitude/longitude grid: east-west (longitude) and north-south resolution (latitude). Default: 0.25 x 0.25
                'time'          : hours,
                'format'        : 'netcdf', # Supported format: grib and netcdf. Default: grib
                'data_format'   : 'netcdf',
                'download_format': 'unarchived'
            }, os.path.join(varpath, filename))
        
##################### Move and combine from the download path to the final path ###########################

# Top-level directory containing subdirectories with .nc files
top_dir = outpath
final_file = os.path.join(final_path, f'{name}_era5_data.nc')

# Collect all .nc files from all subdirectories
nc_files = []
for root, dirs, files in os.walk(top_dir):
    for file in files:
        if file.endswith(".nc"):
            nc_files.append(os.path.join(root, file))

# Load and store datasets
datasets = []
for file_path in nc_files:
    try:
        ds = xr.open_dataset(file_path)
        datasets.append(ds)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")

# Merge all datasets by variable name
combined = xr.merge(datasets)

# Save to NetCDF
final_path = os.path.join(top_dir, final_file)
combined.to_netcdf(final_path)

print(f"Combined file saved to: {final_path}")