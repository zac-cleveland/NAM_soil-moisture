#!/usr/bin/env python
# coding: utf-8

# This script calculates correlations between various parameters and saves them to their own netcdf file

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
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
cp_in_path = '/glade/u/home/zcleveland/scratch/ERA5/cp/'  # path to subset CP data
corr_out_path = '/glade/u/home/zcleveland/scratch/ERA5/correlations/'  # path to correlation calculation folder
der_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/derived/'  # path to derived scripts


# In[3]:


# define list of variables

# surface instantaneous variables
sfc_instan_list = [
    'sd',  # snow depth  (m of water equivalent)
    'msl',  # mean sea level pressure (Pa)
    'tcc',  # total cloud cover (0-1)
    'stl1',  # soil temp layer 1 (K)
    'stl2',  # soil temp layer 2 (K)
    'stl3',  # soil temp layer 3 (K)
    'stl4',  # soil temp layer 4 (K)
    'swvl1',  # soil volume water content layer 1 (m^3 m^-3)
    'swvl2',  # soil volume water content layer 2 (m^3 m^-3)
    'swvl3',  # soil volume water content layer 3 (m^3 m^-3)
    'swvl4',  # soil volume water content layer 4 (m^3 m^-3)
    '2t',  # 2 meter temp (K)
    '2d',  # 2 meter dew point (K)
    'ishf',  # instant surface heat flux (W m^-2)
    'ie',  # instant moisture flux (kg m^-2 s^-1)
    'cape',  # convective available potential energy (J kg^-1)
    'tcw',  # total column water (kg m^-2) -- sum total of solid, liquid, and vapor in a column
    'sstk',  # sea surface temperature (K)
]

# surface accumulation variables
sfc_accumu_list = [
    'lsp',  # large scale precipitation (m of water)
    'cp',  # convective precipitation (m of water)
    'tp',  # total precipitation (m of water) -- DERIVED
    'sshf',  # surface sensible heat flux (J m^-2)
    'slhf',  # surface latent heat flux (J m^-2)
    'ssr',  # surface net solar radiation (J m^-2)
    'str',  # surface net thermal radiation (J m^-2)
    'sro',  # surface runoff (m)
    'sf',  # total snowfall (m of water equivalent)
    'ssrd',  # surface solar radiation downwards (J m^-2)
    'strd',  # surface thermal radiation downwards (J m^-2)
    'ttr',  # top net thermal radiation (OLR, J m^-2) -- divide by time (s) for W m^-2
]

# pressure level variables
pl_var_list = [
    # 'pv',  # potential vorticity (K m^2 kg^-1 s^-1)
    # 'crwc',  # specific rain water content (kg kg^-1)
    # 'cswc',  # specific snow water content (kg kg^-1)
    'z',  # geopotential (m^2 s^2)
    't',  # temperature (K)
    'u',  # u component of wind(m s^-1)
    'v',  # v component of wind (m s^-1)
    'q',  # specific humidity (kg kg^-1)
    'w',  # vertical velo|city (Pa s^-1)
    # 'vo',  # vorticity - relative (s^-1)
    # 'd',  # divergence (s^-1)
    'r',  # relative humidity (%)
    # 'clwc',  # specific cloud liquid water content
    # 'ciwc',  # specific cloud ice water content
    # 'cc',  # fraction of cloud cover (0-1)
]

# NAM variables
NAM_var_list = [
    'onset',
    'retreat',
    'length'
]


# In[42]:


# define a function to calculate the correlation between
# any two variables in certain months
def calc_correlation(var1='swvl1', var1_level=700, var1_month_list=[3, 4, 5], var1_region='cp',
                     var2='tp', var2_level=700, var2_month_list=[6, 7, 8], var2_region='dsw'):

    # months list
    var1_months = month_num_to_name(var=var1, months=var1_month_list)
    var2_months = month_num_to_name(var=var2, months=var2_month_list)

    fn_list = [str(var1), str(var1_months), str(var1_region), str(var2), str(var2_months), str(var2_region)]
    fn_core = '_'.join([i for i in fn_list if i != ''])

    # filename and path
    out_fn = f'corr_{fn_core}.nc'
    out_fp = os.path.join(corr_out_path, 'domain', out_fn)

    # check existence of file already
    if os.path.exists(out_fp):
        print(f'File already exists for: {out_fn}')
        print('\nSkipping . . .')
        return

    # open datasets

    # var 1
    if ((var1 in sfc_instan_list) or (var1 in sfc_accumu_list)):
        var1_data = open_sfc_data(var=var1, region=var1_region, months=var1_month_list)
    elif var1 in pl_var_list:
        var1_data = open_pl_data(var=var1, p_level=var1_level, months=var1_month_list)
    elif var1 in NAM_var_list:
        var1_data = open_NAM_data(var=var1)
    else:
        print('Something went wront . . .')
        return

    # var 2
    if ((var2 in sfc_instan_list) or (var2 in sfc_accumu_list)):
        var2_data = open_sfc_data(var=var2, region=var2_region, months=var2_month_list)
    elif var2 in pl_var_list:
        var2_data = open_pl_data(var=var2, p_level=var2_level, months=var2_month_list)
    elif var2 in NAM_var_list:
        var2_data = open_NAM_data(var=var2)
    else:
        print('Something went wront . . .')
        return

    if ((var1_data is None) or (var2_data is None)):
        print(f'No files were found a var: \n{var1_data}\n{var2_data}')
        return

    # calculate correlation
    var_corr = xr.corr(var1_data, var2_data, dim='year')

    # save to netCDF file
    var_corr.to_netcdf(out_fp)


# In[5]:


# define a function to turn a list of integers into months
def month_num_to_name(var, months):

    # make string for month letters from var_range (e.g. [6,7,8] -> 'JJA')
    if var in NAM_var_list:
        var_months = ''
    elif len(months) == 1:
        var_months = calendar.month_name[months[0]]  # use full month name if only 1 month
    elif ((len(months) > 1) & (len(months) <= 12)):
        var_months = ''.join([calendar.month_name[m][0] for m in months])
    return var_months


# In[6]:


# define function to open pressure level datasets
def open_pl_data(var='z', p_level=700, months=None):

    # grab files for pl var
    files = glob.glob(f'{my_era5_path}dsw/*/pl/{var.lower()}_*_dsw.nc')
    files.sort()

    if not files:
        return None

    # open dataset
    ds = xr.open_mfdataset(files, data_vars='minimal', coords='minimal', parallel=True, chunks={'level': 1})

    # subset the data bas
    da = ds[var.upper()].sel(level=p_level, drop=True)
    var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').mean(dim='time')
    return var_data


# In[7]:


# define a function to open surface data
def open_sfc_data(var='swvl1', region='dsw', months=None):

    # grab files for sfc var
    if region.lower() == 'dsw':
        files = glob.glob(f'{my_era5_path}dsw/*/{var.lower()}_*_dsw.nc')
    elif region.lower() == 'cp':
        files = glob.glob(f'{my_era5_path}cp/{var.lower()}_198001_201912_cp.nc')
    files.sort()

    if not files:
        return None

    # open dataset
    ds = xr.open_mfdataset(files)

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v][0]
    da = ds[var_name]
    # get data from var
    if var.lower() in sfc_instan_list:
        var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').mean(dim='time')
    elif var.lower() in sfc_accumu_list:
        var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').sum(dim='time')
    return var_data


# In[17]:


# define a function to open onset, retreat, and length of NAM data
def open_NAM_data(var='onset'):

    # grab files for NAM data
    files = glob.glob(f'{my_era5_path}dsw/NAM_{var.lower()}.nc')
    files.sort()

    if not files:
        return None

    # open dataset
    ds = xr.open_mfdataset(files)
    ds['year'] = ds['year'].dt.year  # convert to only year.  e.g. 2012-01-01 -> 2012

    # pull out actual variable name in the dataset since they can be different
    if ((var.lower() == 'onset') or (var.lower() == 'retreat')):
        da = ds['date'].dt.dayofyear
    elif var.lower() == 'length':
        da = ds['dayofyear']
    return da


# In[ ]:


# calculate correlations for onset, length, and summer precipitation
var_list1 = sfc_accumu_list + sfc_instan_list + pl_var_list
var_list2 = NAM_var_list + ['tp']
region_list = ['dsw', 'cp']
for var1 in var_list1:
    for var2 in var_list2:
        for region in region_list:
            with open(f'{der_script_path}corr.txt', 'a') as file:
                file.write(f'{var1} : {var2} : {region}\n')
            calc_correlation(var1=var1, var1_level=700, var1_month_list=[3, 4, 5], var1_region=region,
                             var2=var2, var2_level=700, var2_month_list=[6, 7, 8], var2_region='dsw')

