#!/usr/bin/env python
# coding: utf-8

# This script calculates geopotential thickness and saves it as a netcdf file

# In[5]:


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

from scipy.signal import detrend
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
import IPython.core.display as di  # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)


# In[6]:


# specify directories
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to subset CP data
corr_out_path = '/glade/u/home/zcleveland/scratch/ERA5/correlations/'  # path to correlation calculation folder
der_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/derived/'  # path to derived scripts


# In[ ]:


# define a function to calculate geopotential height in m
def calc_z_height(year, month, **kwargs):

    out_fp = f'{my_era5_path}dsw/{year}/pl/Z_Height_{year}{month}_dsw.nc'

    # open geopotential dataset
    z_files = glob.glob(f'{my_era5_path}dsw/{year}/pl/z_{year}{month}_dsw.nc')
    ds = xr.open_mfdataset(z_files)

    da = ds['Z']

    z_height = (da / 9.80665).rename('Z_Height')

    z_height.to_netcdf(out_fp)


# In[ ]:


# define the main function to calculate the geopotential thickness between two layers
def calc_z_height_diff(year, lower_level, upper_level, **kwargs):

    out_fp = f'{my_era5_path}dsw/{year}/pl/Z_Height_{lower_level}-{upper_level}_{year}01_{year}12_dsw.nc'

    # open geopotential dataset
    z_files = glob.glob(f'{my_era5_path}dsw/{year}/pl/Z_Height*.nc')
    ds = xr.open_mfdataset(z_files)

    da = ds['Z_Height']

    dz = (da.sel(level=upper_level) - da.sel(level=lower_level)).rename(f'Z_Height_{lower_level}-{upper_level}')

    dz.to_netcdf(out_fp)


# In[1]:


for year in range(1980,2020):
    str_year = str(year)
    for month in range(1,13):
        if month<10:
            str_month = f'0{month}'
        else:
            str_month = str(month)
        calc_z_height(str_year, str_month)


# In[ ]:


upper_level = 500
lower_level = 1000
for year in range(1980,2020):
    calc_z_height_diff(year, lower_level, upper_level)

