#!/usr/bin/env python
# coding: utf-8

# This script is used for random testing and analysis for which I don't want to create a unique script, but want to have a way to reference my process if I have to end a session or get disconnected.

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
import math
import numpy as np
import netCDF4 as nc
import xarray as xr
import scipy as sp
import scipy.linalg
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
import matplotlib.image as mpimg
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

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

# paths to various directories
rda_era5_path = '/glade/campaign/collections/rda/data/ds633.0/'  # base path to ERA5 data on derecho
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
plot_out_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/'  # path to generated plots
scripts_main_path = '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/'  # path to my dicts, lists, and functions

# import variable lists and dictionaries
if scripts_main_path not in sys.path:
    sys.path.insert(0, scripts_main_path)  # path to file containing these lists/dicts
if 'get_var_data' in sys.modules:
    importlib.reload(sys.modules['get_var_data'])
if 'my_functions' in sys.modules:
    importlib.reload(sys.modules['my_functions'])
if 'my_dictionaries' in sys.modules:
    importlib.reload(sys.modules['my_dictionaries'])

# import common functions that I've created
from get_var_data import get_var_data, get_var_files, open_var_data, subset_var_data, time_to_year_month_avg, time_to_year_month_sum, time_to_year_month
from my_functions import month_num_to_name, ensure_var_list

# import lists and dictionaries
from my_dictionaries import (
sfc_instan_list, sfc_accumu_list, pl_var_list, derived_var_list, invar_var_list,
NAM_var_list, region_avg_list, flux_var_list, vector_var_list, misc_var_list,
var_dict, var_units, region_avg_dict, region_avg_coords, region_colors_dict
)


# In[ ]:


# testing subsetting by opening all days in month at once and saving after
print(f"\n{'--'*20}\nProcessing: Whole Month\n{'--'*20}\n")
start_time = time.time()
files = glob.glob('/glade/u/home/zcleveland/rda_era5/e5.oper.an.pl/198002/*_z.*.nc')

ds = xr.open_mfdataset(files)
ds_sub = ds.sel(latitude=slice(50,10), longitude=slice(230,270), level=[1000, 850, 700, 500, 300], drop=True)
da_mean = ds_sub['Z'].resample(time='1D').mean('time', skipna=True)
da_mean.to_netcdf(f'/glade/u/home/zcleveland/scratch/data_temp/z_198002_python_monthly.nc')

print(f'total time: {(time.time() - start_time):.4f}')
print(f"\n{'--'*20}\nCompleted: Whole Month\n{'--'*20}\n")


# In[ ]:


# testing subsetting data by going day by day, then concatenating monthly data
print(f"\n{'--'*20}\nProcessing: Day by Day\n{'--'*20}\n")
start_time = time.time()
files = glob.glob('/glade/u/home/zcleveland/rda_era5/e5.oper.an.pl/198001/*_z.*.nc')
for day, file in enumerate(files, start=1):
    loop_time = time.time()
    print(f'loop: {day}', end=' --- ')
    ds = xr.open_dataset(file)
    ds_sub = ds.sel(latitude=slice(50,10), longitude=slice(230,270), level=[1000, 850, 700, 500, 300], drop=True)
    da_mean = ds_sub['Z'].resample(time='1D').mean('time', skipna=True)
    da_mean.to_netcdf(f'/glade/u/home/zcleveland/scratch/data_temp/z_198001{day}_daily_python.nc')
    print(f'loop time: {(time.time() - loop_time):.4f}')
print(f'\ntotal loop time: {(time.time() - start_time):.4f}')
files = glob.glob('/glade/u/home/zcleveland/scratch/data_temp/z_198001*_daily_python.nc')
ds = xr.open_mfdataset(files)
ds.to_netcdf('/glade/u/home/zcleveland/scratch/data_temp/z_198001_python_individual.nc')
print(f'total time: {(time.time() - start_time):.4f}')
print(f"\n{'--'*20}\nCompleted: Day by Day\n{'--'*20}\n")


# In[ ]:


# testing by writing subset data first, then reopening all together and computing the mean
print(f"\n{'--'*20}\nProcessing: Subset then Mean\n{'--'*20}\n")
start_time = time.time()
files = glob.glob('/glade/u/home/zcleveland/rda_era5/e5.oper.an.pl/198003/*_z.*.nc')
for day, file in enumerate(files, start=1):
    loop_time = time.time()
    print(f'loop: {day}', end=' --- ')
    ds = xr.open_dataset(file)
    ds_sub = ds.sel(latitude=slice(50,10), longitude=slice(230,270), level=[1000, 850, 700, 500, 300], drop=True)
    ds_sub.to_netcdf(f'/glade/u/home/zcleveland/scratch/data_temp/z_198003{day}_subset_python.nc')
    print(f'loop time: {(time.time() - loop_time):.4f}')
print(f'\ntotal loop time: {(time.time() - start_time):.4f}')
files = glob.glob('/glade/u/home/zcleveland/scratch/data_temp/z_198003*_subset_python.nc')
ds = xr.open_mfdataset(files)
da_mean = ds['Z'].resample(time='1D').mean('time', skipna=True)
da_mean.to_netcdf('/glade/u/home/zcleveland/scratch/data_temp/z_198003_python_subset_first.nc')
print(f'total time: {(time.time() - start_time):.4f}')
print(f"\n{'--'*20}\nCompleted: Subset then Mean\n{'--'*20}\n")


# In[ ]:


# testing my function to subset the data otherwise
sys.path.insert(0, '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/')
from subset_era5_data import *
start_time = time.time()
print(f"\n{'--'*20}\nProcessing: My Function\n{'--'*20}\n")
main('z', region='WestUS_Mexico', year=1980, month=4, **{'overwrite_flag': False, 'save_nc': True, 'pl_levels': [1000, 850, 700, 500, 300]})
print(f'total time: {(time.time() - start_time):.4f}')
print(f"\n{'--'*20}\nCompleted: My Function\n{'--'*20}\n")

