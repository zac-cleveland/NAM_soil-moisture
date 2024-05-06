#!/usr/bin/env python
# coding: utf-8

# This script calculates correlations between various parameters and saves them to their own netcdf file

# In[17]:


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


# In[18]:


# specify directories
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
cp_in_path = '/glade/u/home/zcleveland/scratch/ERA5/cp/'  # path to subset CP data
corr_out_path = '/glade/u/home/zcleveland/scratch/ERA5/correlations/'  # path to correlation calculation folder
sub_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/'  # path to subsetting scripts
der_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/derived/'  # path to derived scripts
plot_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/plotting/'  # path to plotting scripts
fig_out_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/'  # path to generated figures
temp_scratch_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/temp/'  # path to temp directory in scratch


# In[19]:


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
    'ttr',  # top net thermal radiation (OLR, J m^-2) - accumu -- divide by time (s) for W m^-2
    'sstk',  # sea surface temperature (K) - instan
]

NAM_var_list = [
    'onset',
    'retreat',
    'length'
]


# In[ ]:


# define a function to calculate the correlation between
# any given parameter and the NAM onset date
def calc_correlation(NAM_var = 'onset', var='swvl1', months=[3, 4, 5], cp_flag=False):

    # create string to make directory path for figure save
    if cp_flag:
        var_region = 'cp'
    else:
        var_region = 'dsw'

    # create list of months over which to average
    var_months_list = months  # [int(m) for m in str(months)]  # turn var integer into list (e.g. 678 -> [6,7,8])
    # make string for month letters from var_range (e.g. [6,7,8] -> 'JJA')
    var_months = ''.join([calendar.month_name[m][0] for m in var_months_list])

    # path to save figures
    out_fn = f'corr_{var}_{NAM_var}_{var_months}_{var_region}.nc'
    out_fp = os.path.join(corr_out_path, var_region, out_fn)

    # check existence of file already
    if os.path.exists(out_fp):
        print(f'File already exists for: {out_fn}')
        print('\nSkipping . . .')
        return

    # open onset dataset
    NAM_ds = xr.open_dataset(os.path.join(my_era5_path, f'dsw/NAM_{NAM_var}.nc'))
    NAM_ds['year'] = NAM_ds['year'].dt.year  # convert to only year.  e.g. 2012-01-01 -> 2012

    # extract data array of the NAM variable
    if NAM_var == 'length':
        NAM_da = NAM_ds['dayofyear']
        NAM_data = NAM_da.astype('float32')
    else:
        NAM_da = NAM_ds['date']
        NAM_data = NAM_da.dt.dayofyear.astype('float32')

    # open var dataset
    if cp_flag:
        var_files = glob.glob(f'{my_era5_path}{var_region}/*{var}_198001_201912_cp.nc')
        if len(var_files) != 1:
            print(f'Too many files for var_region: cp -- {len(var_files)}\n')
            print('Skipping . . .')
            return
    else:
        var_files = glob.glob(f'{my_era5_path}{var_region}/*/*{var}_*_{var_region}.nc')
        if len(var_files) != 40:
            print(f'Too many files for var_region: cp -- {len(var_files)}\n')
            print('Skipping . . .')
            return

    var_ds = xr.open_mfdataset(var_files)

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in var_ds.data_vars.keys() if f'{var.upper()}' in v][0]
    var_da = var_ds[var_name]

    # get data from var
    if 'AVG' in var_name:
        mon_mean = var_da.resample(time='1M').mean()
        var_mon_mean = mon_mean.sel(time=mon_mean['time.month'].isin(var_months_list))
        var_data = var_mon_mean.groupby('time.year').mean(dim='time')
    else:
        mon_sum = var_da.resample(time='1M').sum()
        var_mon_sum = mon_sum.sel(time=mon_sum['time.month'].isin(var_months_list))
        var_data = var_mon_sum.groupby('time.year').sum(dim='time')

    # calculate correlation
    var_corr = xr.corr(NAM_data, var_data, dim='year')

    # save correlation as netcdf file
    var_corr.to_netcdf(out_fp)


# In[ ]:


# calculate correlations for dsw
# for var in var_list:
#     print(var, '\n')
#     for NAM_var in NAM_var_list:
#         calc_correlation(NAM_var=NAM_var, var=var, months=[3, 4, 5], cp_flag=False)


# In[ ]:


# calculate correlations for cp
# for var in var_list:
#     print(var, '\n')
#     for NAM_var in NAM_var_list:
#         calc_correlation(NAM_var=NAM_var, var=var, months=[3, 4, 5], cp_flag=True)


# In[ ]:


# calculate correlation between onset and summer precip
# for NAM_var in NAM_var_list:
#     calc_correlation(NAM_var=NAM_var, var='tp', months=[6, 7, 8], cp_flag=False)


# In[ ]:


# calculate correlation between onset and summer AND fall precip
# for NAM_var in NAM_var_list:
#     calc_correlation(NAM_var=NAM_var, var='tp', months=[6, 7, 8, 9, 10, 11], cp_flag=False)


# In[ ]:


# calculate correlation between onset and YEARLY precip
# for NAM_var in NAM_var_list:
#     calc_correlation(NAM_var=NAM_var, var='tp', months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], cp_flag=False)


# In[ ]:


# define a function to calculate the correlation between
# any 2 parameters over specified months
def calc_var_correlation(var1='swvl1',var1_months=[3, 4, 5],
                         var2='tp', var2_months=[6, 7, 8], var_region='dsw'):

    # create list of months over which to average
    var1_months_list = var1_months  # [int(m) for m in str(var1_months)]  # turn var integer into list (e.g. 678 -> [6,7,8])
    var2_months_list = var2_months  # [int(m) for m in str(var2_months)]  # turn var integer into list (e.g. 678 -> [6,7,8])
    # make string for month letters from var_range (e.g. [6,7,8] -> 'JJA')
    var1_months = ''.join([calendar.month_name[m][0] for m in var1_months_list])
    var2_months = ''.join([calendar.month_name[m][0] for m in var2_months_list])

    # path to save figures
    out_fn = f'corr_{var1}_{var1_months}_{var2}_{var2_months}_{var_region}.nc'
    out_fp = os.path.join(corr_out_path, var_region, out_fn)

    # check existence of file already
    if os.path.exists(out_fp):
        print(f'File already exists for: {out_fn}')
        print('\nSkipping . . .')
        return

    # get var1 files depending on var_region
    if var_region == 'cp':
        var1_files = glob.glob(f'{my_era5_path}{var_region}/*{var1}_198001_201912_cp.nc')
    elif var_region == 'dsw':
        var1_files = glob.glob(f'{my_era5_path}{var_region}/*/*{var1}_*_dsw.nc')
    else:
        print(f'var_region not found: {var_region}')
        return

    # get var2 files for dsw only
    var2_files = glob.glob(f'{my_era5_path}{var_region}/*/*{var2}_*_dsw.nc')

    # open datasets
    var1_ds = xr.open_mfdataset(var1_files)
    var2_ds = xr.open_mfdataset(var2_files)

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var1_name = [v for v in var1_ds.data_vars.keys() if f'{var1.upper()}' in v][0]
    var1_da = var1_ds[var1_name]
    var2_name = [v for v in var2_ds.data_vars.keys() if f'{var2.upper()}' in v][0]
    var2_da = var2_ds[var2_name]

    # get data from var1
    if 'AVG' in var1_name:
        mon_mean = var1_da.resample(time='1M').mean()
        var1_mon_mean = mon_mean.sel(time=mon_mean['time.month'].isin(var1_months_list))
        var1_data = var1_mon_mean.groupby('time.year').mean(dim='time')
    else:
        mon_sum = var1_da.resample(time='1M').sum()
        var1_mon_sum = mon_sum.sel(time=mon_sum['time.month'].isin(var1_months_list))
        var1_data = var1_mon_sum.groupby('time.year').sum(dim='time')

    # get data from var2
    if 'AVG' in var2_name:
        mon_mean = var2_da.resample(time='1M').mean()
        var2_mon_mean = mon_mean.sel(time=mon_mean['time.month'].isin(var2_months_list))
        var2_data = var2_mon_mean.groupby('time.year').mean(dim='time')
    else:
        mon_sum = var2_da.resample(time='1M').sum()
        var2_mon_sum = mon_sum.sel(time=mon_sum['time.month'].isin(var2_months_list))
        var2_data = var2_mon_sum.groupby('time.year').sum(dim='time')

    # calculate correlation
    var_corr = xr.corr(var1_data, var2_data, dim='year')

    # save correlation as netcdf file
    var_corr.to_netcdf(out_fp)


# In[98]:


# define a function to calculate the correlation between the start of the monsoon averages over a certain region and other variables globally
def calc_correlation_global(var='ttr', months=[3, 4, 5], NAM_var='onset'):

    # make string for month letters from var_range (e.g. [6,7,8] -> 'JJA')
    if len(months) == 1:
        var_months = calendar.month_name[months[0]]  # use full month name if only 1 month
    elif ((len(months) > 1) & (len(months) <= 12)):
        var_months = ''.join([calendar.month_name[m][0] for m in months])
    else:
        print(f'invalid input for "months" : {months}')
        # with open(f'{der_script_path}gobal.txt', 'a') as file:
        #     file.write(f'\ninvalid input for "months" : {months}\n')
        return

    # path to save figures
    out_fn = f'corr_{var}_{NAM_var}_{var_months}_global.nc'
    out_fp = os.path.join(corr_out_path, 'global', out_fn)

    # check existence of file already
    if os.path.exists(out_fp):
        print(f'File already exists for: {out_fn}')
        print('\nSkipping . . .')
        # with open(f'{der_script_path}gobal.txt', 'a') as file:
        #     file.write(f'\nFile already exists for: {out_fn}\n')
        return

    # lat/lon range for averaging
    lats = slice(38,28)
    lons = slice(246, 256)
    # open onset dataset
    NAM_ds = xr.open_dataset(os.path.join(my_era5_path, f'dsw/NAM_{NAM_var}.nc'))
    NAM_ds['year'] = NAM_ds['year'].dt.year  # convert to only year.  e.g. 2012-01-01 -> 2012

    # extract data array of the NAM variable
    if NAM_var == 'length':
        NAM_da = NAM_ds['dayofyear']
        NAM_data = NAM_da.astype('float32')
    else:
        NAM_da = NAM_ds['date']
        NAM_data = NAM_da.dt.dayofyear.astype('float32')

    # select region and calculate mean
    NAM_avg = NAM_data.sel(latitude=lats, longitude=lons).mean(dim=['latitude', 'longitude'])

    # open var dataset
    var_files = glob.glob(f'{my_era5_path}global/*/*{var}_*_dsw.nc')

    var_ds = xr.open_mfdataset(var_files)

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in var_ds.data_vars.keys() if f'{var.upper()}' in v][0]
    var_da = var_ds[var_name]

    # get data from var
    if 'AVG' in var_name:
        mon_mean = var_da.resample(time='1M').mean()
        var_mon_mean = mon_mean.sel(time=mon_mean['time.month'].isin(months))
        var_data = var_mon_mean.groupby('time.year').mean(dim='time')
    else:
        mon_sum = var_da.resample(time='1M').sum()
        var_mon_sum = mon_sum.sel(time=mon_sum['time.month'].isin(months))
        var_data = var_mon_sum.groupby('time.year').sum(dim='time')

    # calculate correlation
    var_corr = xr.corr(NAM_avg, var_data, dim='year')

    # save correlation as netcdf file
    var_corr.to_netcdf(out_fp)


# In[ ]:


# calculate correlations for ttr
vars = ['ttr', 'sstk']
months_list = [
    [3],
    [4],
    [5],
    [6],
    [7],
    [8],
    [3, 4, 5],
    [6, 7, 8],
]
NAM_var_list = ['onset', 'retreat', 'length']
for var in vars:
    for months in months_list:
        for NAM_var in NAM_var_list:
            with open(f'{der_script_path}gobal.txt', 'a') as file:
                file.write(f'{var} - {months} - {NAM_var}\n')
            calc_correlation_global(var=var, months=months, NAM_var=NAM_var)

