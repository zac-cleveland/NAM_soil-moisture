#!/usr/bin/env python
# coding: utf-8

# This script takes the post-subset ERA5 data (from the Desert Southwest) and regrids it to match the ESA soil moisture data.  

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
import importlib

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
scratch_path = '/glade/u/home/zcleveland/scratch/'  # scratch directory on NCAR
repo_path = '/glade/u/home/zcleveland/NAM_soil-moisture/'  # github repository on NCAR

era5_dsw_path = os.path.join(scratch_path, 'ERA5/dsw/')
esa_dsw_path = os.path.join(scratch_path, 'ESA_data/dsw/')
era5_regrid_path = os.path.join(scratch_path, 'ERA5/regrid-to-esa/')


# In[63]:


# specify variable list
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


# In[62]:


# define a function to regrid the subset ERA5 data to the ESA grid
# ERA5 data - latitude:81, longitude:81
# ESA data - latitude:80, longitude:81

def regrid_ERA5_to_ESA(var='lsp', year=1980):
    # create filename and filepath and check for their existence
    out_fn = f'{var}_{year}01_{year}_12_regrid.nc'
    out_fp = os.path.join(era5_regrid_path, str(year), out_fn)

    if os.path.exists(out_fp):
        print(f'{out_fn} already exists. Skipping . . .')
        return
    else:
        print(f'Processing variable: {out_fn}.\n')

    # open ESA dataset
    esa_fp = os.path.join(esa_dsw_path, f'ESA_COMBINED_sm_{year}01_{year}12_dsw.nc')
    esa_ds = xr.open_dataset(esa_fp)

    # open ERA5 dataset
    era_fp = os.path.join(era5_dsw_path, str(year), f'{var}_{year}01_{year}12_dsw.nc')
    era_ds = xr.open_dataset(era_fp)

    # define target grid (esa)
    target_lat = esa_ds['latitude'].values
    target_lon = esa_ds['longitude'].values

    # adjust era longitude values to match the -180 to 180 convention of esa
    print('Regridding . . .\n')
    era_ds_adjusted = era_ds  # temporary
    era_ds_adjusted['longitude'] = np.where(era_ds['longitude'] > 180, era_ds['longitude'] - 360, era_ds['longitude'])
    era_regrid = era_ds_adjusted.interp(latitude=target_lat, longitude=target_lon, method='nearest')

    # save regridded data to netcdf
    print('Saving to netcdf . . .\n')
    era_regrid.to_netcdf(out_fp)


# In[ ]:


# run the script to regrid the era5 data
if __name__ == '__main__':
    for var in var_list:
        print(f'\n\nStarting var: {var}\n\n')
        print('Year: ', end='')
        for year in range(1980,2020):
            print(f'...{year}', end='')
            regrid_ERA5_to_ESA(var=var, year=year)

