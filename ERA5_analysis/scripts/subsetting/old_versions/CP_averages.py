#!/usr/bin/env python
# coding: utf-8

# This script loops through the subset desert southwest data previosly extracted, and computes the mean or sum of different variables over the Colorado Plateau (CP).  This will be used as the "average" or "cumulative" conditions of the Colorado Plateau region to compare to each other lat/lon point.

# In[1]:


# import functions
# OS interaction and time
import os
import sys
import cftime
import datetime
import time
import glob
import dask
import dask.bag as db
import calendar

# math and data
import numpy as np
import netCDF4 as nc
import xarray as xr
import scipy as sp
import pandas as pd
import pickle as pickle
from sklearn import linear_model
import matplotlib.patches as mpatches
from shapely.geometry.polygon import LinearRing
import statsmodels.stats.multitest as multitest

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider
import mpl_toolkits.axes_grid1.axes_size as Size
from mpl_toolkits.axes_grid1 import make_axes_locatable

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

# random
from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)


# In[2]:


# specify directories
data_in_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subsetted data
data_out_path = '/glade/u/home/zcleveland/scratch/ERA5/cp/'  # path to the Colorado Plateau data
esa_out_path = '/glade/u/home/zcleveland/scratch/ESA_data/cp/'  # path to Colorado Plateau ESA data
esa_in_path = '/glade/u/home/zcleveland/scratch/ESA_data/dsw/'  # path to ESA dsw data
sub_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/'  # path to subsetting scripts
plot_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/plotting/' # path to plotting scripts
fig_out_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/'  # path to generated figures
temp_scratch_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/temp/'  # path to temp directory in scratch


# In[3]:


# define list of variables
var_list = [
    'lsp',  # large scale precipitation (m of water) - accumu
    'cp',  # convective precipitation (m of water) - accumu
    'tp',  # total precipitation (m of water) - accumu -- DERIVED
    'sd',  # snow depth  (m of water equivalent) - instan
    'msl',  # mean sea level pressure (Pa) - instan
    'tcc',  # total cloud cover (0-1) - instan
    'stl1',  # soil temp layer 1 (K) - instan
    'stl2',  # soil temp layer 2 (K) - instan
    'stl3',  # soil temp layer 3 (K) - instan
    'stl4',  # soil temp layer 4 (K) - instan
    'swvl1',  # soil volume water content layer 1 (m^3 m^-3) - instan
    'swvl2',  # soil volume water content layer 2 (m^3 m^-3) - instan
    'swvl3',  # soil volume water content layer 3 (m^3 m^-3) - instan
    'swvl4',  # soil volume water content layer 4 (m^3 m^-3) - instan
    '2t',  # 2 meter temp (K) - instan
    '2d',  # 2 meter dew point (K) - instan
    'ishf',  # instant surface heat flux (W m^-2) - instan
    'ie',  # instant moisture flux (kg m^-2 s^-1) - instan
    'sshf',  # surface sensible heat flux (J m^-2) - accumu
    'slhf',  # surface latent heat flux (J m^-2) - accumu
    'ssr',  # surface net solar radiation (J m^-2) - accumu
    'str',  # surface net thermal radiation (J m^-2) - accumu
    'sro',  # surface runoff (m) - accumu
    'sf',  # total snowfall (m of water equivalent) - accumu
    'cape',  # convective available potential energy (J kg^-1) - instan
    'tcw',  # total column water (kg m^-2) - sfc (sum total of solid, liquid, and vapor in a column)
    'ssrd',  # surface solar radiation downwards (J m^-2) - accumu
    'strd',  # surface thermal radiation downwards (J m^-2) - accumu
]


# In[27]:


# create function to compute and save CP averages of ERA5 Desert Southwest Variables
def cp_subsetting_monthly(var='tp', regrid_flag=False, esa_flag=False, overwrite_flag=False):
    if (regrid_flag & esa_flag):
        print('regrid_flag and esa_flag cannot both be set to True:')
        print(f'regrid_flag: {regrid_flag}')
        print(f'esa_flag: {esa_flag}')
        return
    # create filename, filepath, and check their existence
    if regrid_flag:
        out_fn = f'{var}_regrid_198001_2019_cp.nc'
        out_fp = os.path.join(data_out_path, out_fn)
    elif esa_flag:
        out_fn = f'{var}_esa_198001_201912_cp.nc'
        out_fp = os.path.join(esa_out_path, out_fn)
    else:
        out_fn = f'{var}_198001_201912_cp.nc'
        out_fp = os.path.join(data_out_path, out_fn)

    if os.path.exists(out_fp):
        if overwrite_flag:
            pass
        else:
            print(f'{out_fn} already exists. Skipping . . .\n')
            return

    # open datasets
    if regrid_flag:
        files = glob.glob(f'{data_in_path}regrid-to-esa/*/*{var}*.nc')
    elif esa_flag:
        files = glob.glob(f'{esa_in_path}*{var}*.nc')
    else:
        files = glob.glob(f'{data_in_path}dsw/*/*{var}*.nc')
    files.sort()
    var_ds = xr.open_mfdataset(files)

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in var_ds.data_vars.keys() if f'{var[0].upper()}' in v.upper()][0]
    var_da = var_ds[var_name]

    # define lat/lon bounds of four corners to define Colorado Plateau
    if (regrid_flag or esa_flag):
        cp_lats = slice(39,35)
        cp_lons = slice(-111,-107)
    else:
        cp_lats = slice(39,35)
        cp_lons = slice(249,253)
    cp_da = var_da.sel(latitude=cp_lats, longitude=cp_lons)

    # sort by month and take average or sum
    # averaged values (instantaneous in ERA5 or sm in ESA)
    if (('AVG' in var_name) or ('sm' in var_name)):
        cp_monthly_data = cp_da.resample(time='1M').mean(dim=['time', 'latitude', 'longitude'])
        # rename variable to include CP
        cp_monthly_data = cp_monthly_data.to_dataset().rename(
            {f'{var_name}': f'{var_name}_CP'}
        )
    # summed values (accumulated in ERA5)
    else:
        cp_monthly_data = cp_da.resample(time='1M').sum(dim=['time', 'latitude', 'longitude'])
        # rename variable to include CP
        cp_monthly_data = cp_monthly_data.to_dataset().rename(
            {f'{var_name}': f'{var_name}_CP'}
        )

    # save to netcdf file
    cp_monthly_data.to_netcdf(out_fp)


# In[ ]:


# dsw original values
if __name__ == '__main__':
    for var in var_list:
        cp_subsetting_monthly(var=var, regrid_flag=False, overwrite_flag=True)


# In[ ]:


# dsw regridded values
if __name__ == '__main__':
    for var in var_list:
        cp_subsetting_monthly(var=var, regrid_flag=True, overwrite_flag=True)


# In[28]:


# esa data
if __name__ == '__main__':
    cp_subsetting_monthly(var='sm', esa_flag=True, overwrite_flag=True)

