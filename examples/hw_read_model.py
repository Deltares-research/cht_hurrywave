#%%
import numpy as np
import rioxarray
import matplotlib.pyplot as plt
import geopandas as gpd
import os
import xarray as xr
from cht_meteo.dataset import MeteoDataset; 
import datetime

from cht_hurrywave.hurrywave import HurryWave;


#%%
model_path = r"c:\git\cht_hurrywave\examples\tmp"
hw = HurryWave(path=model_path)

hw.read()
# %%
hw.grid.exterior.plot()
# %%
gdf = hw.grid.exterior
gdf.plot()

# %%
hw.grid.mask

# %%

import xarray as xr
import numpy as np
import geopandas as gpd
from rasterio.features import shapes
from shapely.geometry import shape

# 1. Create binary mask
da = xr.where(hw.grid.ds.mask > 0, 1, 0).astype(np.int16)

# 2. Assign spatial reference and nodata
da.rio.set_nodata(0)

# 3. Extract shapes where value == 1
results = shapes(da.values, mask=(da.values == 1), transform=da.rio.transform())

# 4. Build GeoDataFrame
geoms = [shape(geom) for geom, val in results if val == 1]
region = gpd.GeoDataFrame(geometry=geoms, crs=da.rio.crs)

# 5. Dissolve to get the exterior
region = region.dissolve()
# %%
