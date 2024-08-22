#!/usr/bin/env python
# coding: utf-8

# This script is used to create monthly means of the already subset ERA5 data.  

# In[56]:


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
from dask.diagnostics import ProgressBar
from dask.distributed import Client
from dask_jobqueue import PBSCluster
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
import statsmodels.stats.multitest as multitest

# random
from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

# paths to various directories
rda_era5_path = '/glade/campaign/collections/rda/data/ds633.0/'  # base path to ERA5 data on derecho
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
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


# In[32]:


# variables to subset
# SFC instantaneous variables
sfc_instan_vars = [
    'sd',  # snow depth  (m of water equivalent)
    'msl',  # mean sea level pressure (Pa)
    'stl1',  # soil temp layer 1 (K)
    'swvl1',  # soil volume water content layer 1 (m^3 m^-3)
    '2t',  # 2 meter temp (K)
    '2d',  # 2 meter dew point (K)
    'cape',  # convective available potential energy (J kg^-1)
    'tcw',  # total column water (kg m^-2) -- sum total of solid, liquid, and vapor in a column
    'sstk',  # sea surface temperature (K)
    'viwve',  # vertical integral of eastward water vapour flux (kg m^-1 s^-1) - positive south -> north
    'viwvn',  # vertical integral of northward water vapour flux (kg m^-1 s^-1) - positive west -> east
    'viwvd',  # vertical integral of divergence of moisture flux (kg m^-2 s^-1) - positive divergencve
    'z_thick_1000-500',  # geopotential thickness from 1000 to 500 mb (m) -- DERIVED
]

# surface accumulation variables
sfc_accumu_vars = [
    'lsp',  # large scale precipitation (m of water)
    'cp',  # convective precipitation (m of water)
    'tp',  # total precipitation (m of water) -- DERIVED
    'sshf',  # surface sensible heat flux (J m^-2)
    'slhf',  # surface latent heat flux (J m^-2)
    'ssr',  # surface net solar radiation (J m^-2)
    'str',  # surface net thermal radiation (J m^-2)
    'sf',  # total snowfall (m of water equivalent)
    'ssrd',  # surface solar radiation downwards (J m^-2)
    'strd',  # surface thermal radiation downwards (J m^-2)
    'ttr',  # top net thermal radiation (OLR, J m^-2) -- divide by time (s) for W m^-2
]

# pressure level variables
pl_vars = [
    'z',  # geopotential (m^2 s^2)
    'z_height',  # geopotential height (m) -- DERIVED
    # 't',  # temperature (K)
    # 'u',  # u component of wind(m s^-1)
    # 'v',  # v component of wind (m s^-1)
    # 'q',  # specific humidity (kg kg^-1)
]


# In[4]:


def get_my_var_files(var):
    pattern = os.path.join(my_era5_path, 'WestUS_Mexico/*', f'{var}_*_WestUS_Mexico.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        raise ValueError(f'No files found for: {pattern}')
    return files    


# In[63]:


def main_monthly(var, **kwargs):
    out_fp = os.path.join(my_era5_path, 'WestUS_Mexico/', f'{var}_1980_2019_WestUS_Mexico.nc')
    if os.path.exists(out_fp):
        print(f'File already exists for: {var}')
        return

    var_files = get_my_var_files(var)
    ds = xr.open_mfdataset(var_files, parallel=True)
    var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v.upper()][0]
    da = ds[var_name]
    da_monthly = time_to_year_month_avg(da)
    da_monthly.to_netcdf(out_fp)
    # return da_monthly


# In[ ]:


if __name__ == '__main__':
    var_list = sfc_instan_vars + sfc_accumu_vars + pl_vars
    for var in var_list:
        print(f'{var} - ', end='')
        start_time = time.time()
        main_monthly(var)
        print(f'{(time.time()-start_time):.2f}')

