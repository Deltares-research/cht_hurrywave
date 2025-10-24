"""

Script to set up HW model for DCSM case


Created on 23-02-2024

@author: mvanderlugt
@author: kvanasselt
"""
#%%
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from cht_hurrywave.hurrywave import HurryWave
# note, we don't use the Mask class from the mask.py nor the grid from grid.py! Instead, both from hurrywave_domain
from cht_utils.geometry import RegularGrid
from cht_hurrywave.bathymetry import HurryWaveBathymetry
from cht_hurrywave.mask import HurryWaveMask
from cht_bathymetry.database import BathymetryDatabase
from cht_model_builder.model_builder import MaskPolygon
from cht_utils.pli_file import read_pli_file
from cht_meteo.meteo import MeteoGrid
from pyproj import CRS 

#%%
main_path  = os.path.join(r"c:\projects\toJosé") 
if not os.path.exists(main_path):
    os.mkdir(main_path)

model_setup = os.path.join(main_path, '04_modelruns', 'extended')

if not os.path.exists(model_setup):
    os.mkdir(model_setup)

hw = HurryWave(path=model_setup)


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
                print('eof')
                break
            
            dep.append([float(x) for x in line[:-1].split()])

        dep = np.array(dep)
        dep[dep==missing_value] = np.nan
        return dep
    else:
        print('other idla\'s not yet scripted')


botfile = r'c:\projects\toJosé\01_data\swannoordzee\swan_run\f998dcs13001\model_io\f998dcs13001.bot'
dep = read_swn_bot(botfile, idla=3)
dep = dep * -1 
#%%
plt.scatter(xgr_bot[:], ygr_bot[:], c=dep[:])
plt.colorbar()

#%% create grid
specs = {'x0': -12, 'y0': 48, 'dx': 0.05, 'dy':  0.033333333, 
        'mmax': 421, 'nmax': 481, 'rotation': 0.0, 
        'crs_name': 'WGS 84', 'crs_type': 'geographic',
        'dt': 300}
hw.input.update(specs)

grid = RegularGrid(hw, specs['x0'], specs['y0'],
                    specs['dx'], specs['dy'],
                    specs['nmax'], specs['mmax'],  # mind you: the order here is n,m!
                    specs['rotation'])

#% create bathy file
hw.input.update({'mskfile': 'hurrywave.msk', 'depfile': 'hurrywave.dep', '': ''})

# Get bathy/topo
bathymetry = HurryWaveBathymetry(hw)
bathymetry.set_bathymetry_from_other_source(grid, xgr_bot, ygr_bot, dep, rectilinearSourceData=False)

# xz, yz = grid.grid_coordinates_centres()
# plt.pcolor(xz, yz, bathymetry.z)
# plt.axis('equal')

# Save bathymetry
bathymetry.save(os.path.join(hw.path, hw.input.variables.depfile))

#%%
#% Mask 
open_boundary_polygon = os.path.join(model_setup, "open_bound.pli")

polygons = read_pli_file(open_boundary_polygon)
polygon_list = []

for polygon in polygons:
    polygon_list.append(MaskPolygon(x=polygon.x,
                            y=polygon.y,
                            zmin=-9999,
                            zmax=1) ) 


mask = HurryWaveMask(grid, bathymetry.z,
            zmin= -9999,
            zmax=0, 
            open_boundary_polygons= polygon_list
            )

hw.mask = mask
# save to file
mask.save(os.path.join(model_setup, hw.input.variables.mskfile))

#%% forcings
tstart = pd.to_datetime('2013-12-01 00:00')
tstop = pd.to_datetime('2013-12-08 00:00')
hw.input.update({'tref': tstart ,'tstart': tstart, 'tstop': tstop, 'dt': 300})

#%%

#wnd_file = r'p:\11204750-hurrywave\01_data\00_SWIVT\sessions\session001\SWAN4131AB\f999am07z015_000\model_io\wnd\f999am07z015_2013120610.wnd'

def read_swn_wnd(filePath, x0inp, y0inp, rot, mxinp, myinp, dxinp, dyinp, idla=3, missing_value = 99):
    
    xx = dxinp*np.arange(0,mxinp+1)
    yy = dyinp*np.arange(0,myinp+1)

    xinp0, yinp0 = np.meshgrid(xx, yy)
    cosrot = np.cos(rot*np.pi/180)
    sinrot = np.sin(rot*np.pi/180)
    xinp = x0inp + xinp0*cosrot - yinp0*sinrot
    yinp = y0inp + xinp0*sinrot + yinp0*cosrot

    if idla==3:
        fid = open(filePath)
        wnd = []
        while True:
            line = fid.readline()

            if (''==line):
                print('eof', filePath)
                break
            
            wnd.append([float(x) for x in line[:-1].split()])

        #Create array with all values in single array
        
        wnds = []
        for wnd_idx in wnd:
            wnds += wnd_idx

        wnds = np.array(wnds)
        wnd[wnd==missing_value] = np.nan

        idxs_u = (myinp+1) * (mxinp+1)

        # wnd files have structure first all east-ward winds, than all northward winds
        wndu = wnds[:idxs_u]
        wndv = wnds[idxs_u:]

        #reshape

        wndu = wndu.reshape(myinp+1, mxinp+1)
        wndv = wndv.reshape(myinp+1, mxinp+1)
        
        return xinp, yinp, wndu, wndv
    else:
        print('other idla\'s not yet scripted')

import glob

wndu = []; wndv = [];
filez = glob.glob(r'p:\11204750-hurrywave\01_data\00_SWIVT\sessions\session001\SWAN4131AB\f998dcs13001_000\f998dcs13001\model_io\wind\*.wnd')
for ix, wnd_file in enumerate(filez):
    xinp, yinp, wndui, wndvi = read_swn_wnd(wnd_file, x0inp=-12, y0inp=48, rot=0, mxinp=210, myinp=240, dxinp=0.1, dyinp=0.06667)
    
    # Add first wnd files 48 times to account for 2 extra days physical spinup

    extra_days = 2

    if ix == 0:
        for i in range(extra_days*24):
            wndu.append(wndui)
            wndv.append(wndvi)
        print(f"Add first wnd file {extra_days*24} times to account for physical spin-up")
    
    wndu.append(wndui)
    wndv.append(wndvi)


wu = np.stack(wndu,0)
wv = np.stack(wndv,0)
tstartinp = tstart
tstopinp = tstop
timeinp = pd.date_range(tstartinp, tstopinp, freq='1H')

#%%
class Dataset():

    def __init__(self, time, x, y, u, v, crs=CRS(4326)):

        self.quantity = []
        self.unit     = None  
        self.crs      = crs
        self.time     = time
        self.x        = x
        self.y        = y
        # self.u        = []
        # self.v        = []
        self.add_wind( u, v)

    class uvwind():

        def __init__(self, u, v):
            self.u        = u
            self.v        = v
            self.name     = 'wind'

    def add_wind(self, u, v):
        self.quantity.append(self.uvwind(u, v))

ds = Dataset(timeinp, xinp[0,:], yinp[:,0], wu, wv)
MeteoGrid.write_to_delft3d(ds, 'wind', parameters=['wind'], path=os.path.join(model_setup))

# wind forcing to input file
hw.input.update({'amufile': 'wind.amu', 'amvfile': 'wind.amv', 'tspinup': 1800.0})

#%% Boundary forcing files have been created with external matlab script: create_HurryWave_boundary_matlab.m (02-scripts)

extended = True


# Retrieve, rewrite and restore files

folder_bnd = r'p:\11204750-hurrywave\02_modelling\SWIVT_cases\f998dcs13\01_data\matlab_generated'

df_bhs = pd.read_csv(os.path.join(folder_bnd, 'bnd_hm0.txt'), 
                     header = None, 
                     delim_whitespace=True)

df_btp =  pd.read_csv(os.path.join(folder_bnd, 'bnd_tps.txt'),
                       header = None, 
                       delim_whitespace=True)

df_bwd =  pd.read_csv(os.path.join(folder_bnd, 'bnd_dir.txt'),
                       header = None, 
                       delim_whitespace=True)

df_bds = pd.read_csv(os.path.join(folder_bnd, 'bnd_dspr.txt'),
                      header = None, 
                      delim_whitespace=True)

df_coor =  pd.read_csv(os.path.join(folder_bnd, 'hurrywave.bnd'),
                        header = None, 
                        delim_whitespace=True)

# For the extended (2 days), we are adjusting the boundary files by adding two more days of forcing

dfs = [df_bhs, df_btp, df_bwd, df_bds]
names = ["hurrywave.bhs", "hurrywave.btp", "hurrywave.bwd", "hurrywave.bds" ]

if extended:
    for df,name in zip(dfs, names):
        print(f"Extending with 2 days")
        df = pd.concat([pd.DataFrame(df.iloc[0]).T, df], ignore_index = True)
        df.iloc[1:, 0] = df.iloc[1:, 0] + 3600 * 24 * 2
        df.round(4).to_csv(os.path.join(model_setup, name ), sep=' ', header =  False, index = False)

    df_coor.to_csv(os.path.join(model_setup, "hurrywave.bnd" ), sep=' ', index = False, header =  False)
    
    # write to input file
    hw.input.update({'bhsfile': 'hurrywave.bhs', 'btpfile': 'hurrywave.btp', 
                 'bwdfile': 'hurrywave.bwd', 'bdsfile': 'hurrywave.bds', 
                 'bndfile': 'hurrywave.bnd',
                 })
    
else:
    print("Using original files")
    # save to csv
    df_bhs.round(4).to_csv(os.path.join(model_setup, "hurrywave.bhs" ), sep=' ', header =  False, index = False)
    df_btp.round(4).to_csv(os.path.join(model_setup, "hurrywave.btp" ), sep=' ', header =  False, index = False)
    df_bwd.round(4).to_csv(os.path.join(model_setup, "hurrywave.bwd" ), sep=' ', header =  False, index = False)
    df_bds.round(4).to_csv(os.path.join(model_setup, "hurrywave.bds" ), sep=' ', header =  False, index = False)
    df_coor.to_csv(os.path.join(model_setup, "hurrywave.bnd" ), sep=' ', index = False, header =  False)

    # write to input file
    hw.input.update({'bhsfile': 'hurrywave.bhs', 'btpfile': 'hurrywave.btp', 
                    'bwdfile': 'hurrywave.bwd', 'bdsfile': 'hurrywave.bds', 
                    'bndfile': 'hurrywave.bnd',
                    })


#%% Output

hw.input.update({'dtmapout': 1800, 'dthisout': 300, 'obsfile': 'hurrywave.obs'})
observation_points_swan_dir = r'p:\11204750-hurrywave\02_modelling\SWIVT_cases\f998dcs13\01_data\f998dcs13001.pnt'

df_obs = pd.read_csv(observation_points_swan_dir, 
                     header = None, 
                     delim_whitespace=True)


df_obs.to_csv(os.path.join(model_setup, "hurrywave.obs" ), sep=' ', index = False, header =  False)


#%% write the total model to file
hw.write()

# %%
