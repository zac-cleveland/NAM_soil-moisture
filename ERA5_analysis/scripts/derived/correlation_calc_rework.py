#!/usr/bin/env python
# coding: utf-8

# This script calculates correlations between various parameters and saves them to their own netcdf file

# In[ ]:


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


# In[ ]:


# specify directories
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
cp_in_path = '/glade/u/home/zcleveland/scratch/ERA5/cp/'  # path to subset CP data
corr_out_path = '/glade/u/home/zcleveland/scratch/ERA5/correlations/'  # path to correlation calculation folder
der_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/derived/'  # path to derived scripts


# In[ ]:


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
    'vipile',  # vertical integral of potential, internal, and latent energy (J m^-2) - instan
    'viwve',  # vertical integral of eastward water vapour flux (kg m^-1 s^-1) - instan -- positive south -> north
    'viwvn',  # vertical integral of northward water vapour flux (kg m^-1 s^-1) - instan -- positive west -> east
    'viwvd',  # vertical integral of divergence of moisture flux (kg m^-2 s^-1) - instan -- positive divergencve
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

# all var in one list
var_list = sfc_instan_list + sfc_accumu_list + pl_var_list

# region average list
region_avg_list = [
    'cp',
    'mr',
    'son',
    'chi',
    'moj',
    'MeNmAz',
]


# In[ ]:


# dictionary of variables and their names
var_dict = {
    'sd': 'Snow Depth',
    'msl': 'Mean Sea Level Pressure',
    'tcc': 'Total Cloud Cover',
    'stl1': 'Soil Temp Layer 1',
    'stl2': 'Soil Temp Layer 2',
    'stl3': 'Soil Temp Layer 3',
    'stl4': 'Soil Temp Layer 4',
    'swvl1': 'Soil Volume Water Content Layer 1',
    'swvl2': 'Soil Volume Water Content Layer 2',
    'swvl3': 'Soil Volume Water Content Layer 3',
    'swvl4': 'Soil Volume Water Content Layer 4',
    '2t': '2 Meter Temp',
    '2d': '2 Meter Dew Point',
    'ishf': 'Instant Surface Heat Flux',
    'ie': 'Instant Moisture Flux',
    'cape': 'Convective Available Potential Energy',
    'tcw': 'Total Column Water',
    'sstk': 'Sea Surface Temperature',
    'vipile': 'vertical integral of potential, internal, and latent energy',
    'viwve': 'vertical integral of eastward water vapour flux',
    'viwvn': 'vertical integral of northward water vapour flux',
    'viwvd': 'vertical integral of divergence of moisture flux',
    'lsp': 'Large Scale Precipitation',
    'cp': 'Convective Precipitation',
    'tp': 'Total Precipitation',
    'sshf': 'Surface Sensible Heat Flux',
    'slhf': 'Surface Latent Heat Flux',
    'ssr': 'Surface Net Solar Radiation',
    'str': 'Surface Net Thermal Radiation',
    'sro': 'Surface Runoff',
    'sf': 'Total Snowfall',
    'ssrd': 'Surface Solar Radiation Downwards',
    'strd': 'Surface Thermal Radiation Downwards',
    'ttr': 'Top Net Thermal Radiation (OLR)',
    'z': 'Geopotential',
    't': 'Temperature',
    'u': 'U Component of Wind',
    'v': 'V Component of Wind',
    'q': 'Specific Humidity',
    'w': 'Vertical Velocity',
    'r': 'Relative Humidity',
    'onset': 'Onset',
    'retreat': 'Retreat',
    'length': 'Length'
}

# dictionary of regions and their names
region_avg_dict = {
    'cp': 'Colorado Plateau',
    'mr': 'Mogollon Rim',
    'son': 'Sonoran Desert',
    'chi': 'Chihuahuan Desert',
    'moj': 'Mojave Desert',
    'MeNmAz': 'Mexico, New Mexico, and Arizona Border',
}

# dictionary of regions and their coordinate boundaries
# [WEST, EAST, NORTH, SOUTH] -- WEST and EAST are on 0-360 latitude grid system
region_avg_coords = {
    'cp': [249, 253, 39, 35],
    'mr': [249, 251, 34, 33],
    'son': [246, 250, 32, 28],
    'chi': [252, 256, 33, 29],
    'moj': [243, 247, 37, 33],
    'MeNmAz': [246, 256, 38, 28],
}


# In[ ]:


# define a function to calculate the correlation between
# any two variables in certain months
def calc_correlation(var1='sd', var1_month_list=[3, 4, 5], var1_region='cp',
                     var2='tp', var2_month_list=[6, 7, 8], var2_region='dsw',
                     detrend_flag=True, overwrite_flag=False, **kwargs):

    # months list
    var1_months = month_num_to_name(var=var1, months=var1_month_list, **kwargs)
    var2_months = month_num_to_name(var=var2, months=var2_month_list, **kwargs)

    # filename and path

    if detrend_flag:
        detrend_str = 'detrend'
    else:
        detrend_str = ''

    fn_list = [str(var1), str(var1_months), str(var1_region),
               str(var2), str(var2_months), str(var2_region),
               str(detrend_str)]
    fn_core = '_'.join([i for i in fn_list if i != ''])
    out_fn = f'corr_{fn_core}.nc'

    if ((var1_region == 'global') or (var2_region == 'global')):
        out_fp = os.path.join(my_era5_path, 'correlations', 'global', out_fn)
    elif ((var1_region in region_avg_list) and (var2_region in region_avg_list)):
        out_fp = os.path.join(my_era5_path, 'correlations', f'regions/{var2_region}', out_fn)
    elif ((var1_region == 'dsw') or (var2_region == 'dsw')):
        out_fp = os.path.join(my_era5_path, 'correlations', 'dsw', out_fn)

    # check existence of file already
    if os.path.exists(out_fp):
        print(f'File already exists for: {out_fn}', end=' - ')
        if not overwrite_flag:
            print('overwrite_flag is set to False. Skipping . . .')
            return
        else:
            print('overwrite_flag is set to True. Overwriting . . .')

    # get files for variables
    var1_files = get_var_files(var1, var1_region, **kwargs)
    var2_files = get_var_files(var2, var2_region, **kwargs)

    if ((not var1_files) or (not var2_files)):
        print(f'missing files var1: {len(var1_files)} - var2: {len(var2_files)} . . .')
        return

    # open datasets
    var1_data = get_var_data(var1_files, var1, var1_month_list, var1_region, level=kwargs.get('var1_level', 700), **kwargs)
    var2_data = get_var_data(var2_files, var2, var2_month_list, var2_region, level=kwargs.get('var2_level', 700), **kwargs)

    if ((var1_data is None) or (var2_data is None)):
        print(f'missing var data {var1}: \n{var1_data}\n{var2}:\n{var2_data}')
        return

    # calculate correlation
    if detrend_flag:
        var1_data = apply_detrend(var1_data)
        var2_data = apply_detrend(var2_data)

    var_corr = apply_correlation(var1_data, var2_data)
    # return var_corr, out_fn

    # save to netCDF file
    var_corr.to_netcdf(out_fp)


# In[ ]:


# define a function to turn a list of integers into months
def month_num_to_name(var, months, **kwargs):

    # make string for month letters from var_range (e.g. [6,7,8] -> 'JJA')
    if var in NAM_var_list:
        var_months = ''
    elif len(months) == 1:
        var_months = calendar.month_name[months[0]]  # use full month name if only 1 month
    elif ((len(months) > 1) & (len(months) <= 12)):
        var_months = ''.join([calendar.month_name[m][0] for m in months])
    return var_months


# In[ ]:


# define a function to get the files for a given variable/region
def get_var_files(var, region, **kwargs):

    # grab files for sfc var
    if ((var in sfc_instan_list) or (var in sfc_accumu_list)):
        # dsw
        if region == 'dsw':
            files = glob.glob(f'{my_era5_path}dsw/*/{var.lower()}_*_dsw.nc')
        # regional averages
        elif region in region_avg_list:
            files = glob.glob(f'{my_era5_path}regions/{region}/{var.lower()}_198001_201912_{region}.nc')
        # global
        elif region == 'global':
            files = glob.glob(f'{my_era5_path}global/*/{var.lower()}_*_dsw.nc')

    # grab files for pl var
    elif var in pl_var_list:
        files = glob.glob(f'{my_era5_path}dsw/*/pl/{var.lower()}_*_dsw.nc')

    # grab files for NAM var
    elif var in NAM_var_list:
        files = glob.glob(f'{my_era5_path}dsw/NAM_{var.lower()}.nc')

    # if something went wrong
    else:
        files = []

    files.sort()
    return files


# In[ ]:


# define a function to open the datasets and return monthly averages
# returns shape (year:40, latitude:81, longitude:81) for dsw
# returns shape (year:40) for regional averages
# returns shape (year:40, latitude:721, longitude:1440) for dsw
def get_var_data(files, var, months, region, level, **kwargs):

    # open dataset
    ds = xr.open_mfdataset(files)

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v.upper()][0]
    da = ds[var_name]
    # select months and compute mean/sum for data

    # sfc var
    if var.lower() in sfc_instan_list:
        var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').mean(dim='time')
    elif var.lower() in sfc_accumu_list:
        var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').sum(dim='time')

    # pl var
    elif var.lower() in pl_var_list:
        var_data = da.sel(time=da['time.month'].isin(months), level=level).groupby('time.year').mean(dim='time')

    #NAM var
    elif ((var.lower() == 'onset') or (var.lower() == 'retreat')):
        var_data = da.dt.dayofyear
    elif var.lower() == 'length':
        var_data = da

    # something went wrong
    else:
        return None

    # check if NAM var requested for regional average
    if ((var.lower() in NAM_var_list) & (region in region_avg_list)):
        lats = slice(region_avg_coords[region][2], region_avg_coords[region][3])
        lons = slice(region_avg_coords[region][0], region_avg_coords[region][1])
        var_data = var_data.sel(latitude=lats, longitude=lons).mean(dim=['latitude', 'longitude'], skipna=True)

    return var_data


# In[ ]:


# define a function to detrend the data

# MANUALLY DETREND WITH LINEAR REGRESSION
def detrend_data(arr):

    # set up x array for the years
    arr_years = np.arange(0,40)

    # mask out nan values
    mask = np.isfinite(arr)
    arr_years_mask = arr_years[mask]
    arr_mask = arr[mask]

    if len(arr_mask) == 0:
        arr_detrend = np.empty(40)
        arr_detrend[:] = np.nan

    else:
        # compute linear regression
        result = sp.stats.linregress(arr_years_mask, arr_mask)
        m, b = result.slope, result.intercept

        # detrend the data
        arr_detrend = arr - (m*arr_years + b)

    return arr_detrend


# def detrend_data(arr):

#     # handle nans and infs
#     mask = np.isfinite(arr)
#     arr_masked = arr[mask]

#     # create new array to keep nan values back in
#     arr_detrend = np.empty(arr.shape)
#     arr_detrend[:] = np.nan

#     if ((len(arr[~mask]) / len(arr)) > 0.5):
#         print('too many nan values, cannot be detrended')
#         return arr_detrend
#     else:
#         # perform detrend
#         detrend_masked = sp.signal.detrend(arr_masked)

#         # input masked array into their spots
#         arr_detrend[mask] = detrend_masked

#     # return data
#     return arr_detrend


# define a function to mask data for detrending or correlating
def apply_detrend(da):

    # load data
    da.load()

    da_detrend = xr.apply_ufunc(
        detrend_data, da,
        input_core_dims=[['year']],
        output_core_dims=[['year']],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype]
    )

    return da_detrend


# In[ ]:


# define a function to calculate the Pearson correlation and p-value statistic
def compute_corr_pval(arr1, arr2):
    # mask out nan and inf values
    mask = np.isfinite(arr1) & np.isfinite(arr2)
    filtered_arr1 = arr1[mask]
    filtered_arr2 = arr2[mask]

    if len(filtered_arr1) < 2:  # check if there are enough data points
        return np.nan, np.nan

    corr, pval = sp.stats.pearsonr(filtered_arr1, filtered_arr2)
    return corr, pval


# define a function to apply the ufunc to the data
def apply_correlation(da1, da2):
    da1.load()
    da2.load()
    result = xr.apply_ufunc(
        compute_corr_pval, da1, da2,
        input_core_dims=[['year'], ['year']],
        output_core_dims=[[],[]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float]
    )
    corr_da = result[0]
    pval_da = result[1]

    corr_ds = xr.merge([corr_da.rename('pearson_r'), pval_da.rename('p_value')])
    return corr_ds


# In[ ]:


# # define function to open pressure level datasets
# def open_pl_data(var='z', p_level=700, months=None):

#     # grab files for pl var
#     files = glob.glob(f'{my_era5_path}dsw/*/pl/{var.lower()}_*_dsw.nc')
#     files.sort()

#     if not files:
#         return None

#     # open dataset
#     ds = xr.open_mfdataset(files, data_vars='minimal', coords='minimal', parallel=True, chunks={'level': 1})

#     # subset the data bas
#     da = ds[var.upper()].sel(level=p_level, drop=True)
#     var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').mean(dim='time')
#     return var_data


# In[ ]:


# # define a function to open surface data
# def open_sfc_data(var='swvl1', region='dsw', months=None):

#     # grab files for sfc var
#     if region.lower() == 'dsw':
#         files = glob.glob(f'{my_era5_path}dsw/*/{var.lower()}_*_dsw.nc')
#     elif region.lower() in region_avg_list:
#         files = glob.glob(f'{my_era5_path}regions/{region.lower()}/{var.lower()}_198001_201912_cp.nc')
#     elif region.lower() == 'global':
#         files = glob.glob(f'{my_era5_path}global/*/{var.lower()}_*_dsw.nc')
#     files.sort()

#     if not files:
#         return None

#     # open dataset
#     ds = xr.open_mfdataset(files)

#     # pull out actual variable name in the dataset since they can be different names/capitalized
#     var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v][0]
#     da = ds[var_name]
#     # get data from var
#     if var.lower() in sfc_instan_list:
#         var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').mean(dim='time')
#     elif var.lower() in sfc_accumu_list:
#         var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').sum(dim='time')
#     return var_data


# In[ ]:


# # define a function to open onset, retreat, and length of NAM data
# def open_NAM_data(var='onset'):

#     # grab files for NAM data
#     files = glob.glob(f'{my_era5_path}dsw/NAM_{var.lower()}.nc')
#     files.sort()

#     if not files:
#         return None

#     # open dataset
#     ds = xr.open_mfdataset(files)
#     ds['year'] = ds['year'].dt.year  # convert to only year.  e.g. 2012-01-01 -> 2012

#     # pull out actual variable name in the dataset since they can be different
#     if ((var.lower() == 'onset') or (var.lower() == 'retreat')):
#         da = ds['date'].dt.dayofyear
#     elif var.lower() == 'length':
#         da = ds['dayofyear']
#     return da


# In[ ]:


# # test cell
# var1 = 'vipile'
# var1_month_list = [3, 4, 5]
# var1_region = 'cp'
# var2 = 'onset'
# var2_month_list = [6, 7, 8]
# var2_region = 'dsw'
# detrend_list=[True,False]
# for detrend_flag in detrend_list:
#     calc_correlation(var1=var1, var1_level=700, var1_month_list=var1_month_list, var1_region=var1_region,
#                      var2=var2, var2_level=700, var2_month_list=var2_month_list, var2_region=var2_region,
#                      detrend_flag=detrend_flag)


# In[ ]:


# # calculate correlations for onset, retreat, length, and summer precipitation
# var_list1 = sfc_accumu_list + sfc_instan_list + pl_var_list
# var_list1.remove('sstk')
# var_list2 = NAM_var_list + ['tp', 'vipile', 'viwve', 'viwvn', 'viwvd']
# region_list = ['cp']
# detrend_list = [True, False]
# len_lists = len(var_list1)*len(var_list2)*len(region_list)*len(detrend_list)
# cnt = 0
# for var1 in var_list1:
#     for var2 in var_list2:
#         for region in region_list:
#             for detrend_flag in detrend_list:
#                 # with open(f'{der_script_path}corr.txt', 'a') as file:
#                 #     file.write(f'{var1}\t:\t{var2}\t:\t{region}\t:\tdetrend={detrend_flag}\t:\t{100*cnt/len_lists} %\n')
#                 print(f'{var1}\t:\t{var2}\t:\t{region}\t:\tdetrend={detrend_flag}\t:\t{100*cnt/len_lists} %')
#                 calc_correlation(var1=var1, var1_month_list=[3, 4, 5], var1_region=region, var1_level=700,
#                                  var2=var2, var2_month_list=[6, 7, 8], var2_region='dsw', var2_level=700,
#                                  detrend_flag=detrend_flag, overwrite_flag=False)
#                 cnt = cnt+1


# In[ ]:


# # calculate correlations for onset, length, and summer precipitation
# var_list1 = sfc_accumu_list + sfc_instan_list + pl_var_list
# var_list2 = ['vipile', 'viwve', 'viwvn', 'viwvd']
# region_list = ['dsw', 'cp']
# for var1 in var_list1:
#     for var2 in var_list2:
#         for region in region_list:
#             # with open(f'{der_script_path}corr.txt', 'a') as file:
#             #     file.write(f'{var1} : {var2} : {region}\n')
#             print(f'{var1} : {var2} : {region}\n')
#             calc_correlation(var1=var1, var1_level=700, var1_month_list=[3, 4, 5], var1_region=region,
#                              var2=var2, var2_level=700, var2_month_list=[6, 7, 8], var2_region='dsw')


# In[ ]:


# # define a function to calculate the correlation for global variables
# def calc_correlation_global(var1='ttr', var1_level=700, var1_month_list=[3, 4, 5], var1_region='global',
#                             var2='tp', var2_level=700, var2_month_list=[6, 7, 8], var2_region='dsw',
#                             detrend_flag=True, overwrite_flag=False, **kwargs):

#     # months list
#     var1_months = month_num_to_name(var=var1, months=var1_month_list)
#     var2_months = month_num_to_name(var=var2, months=var2_month_list)

#     fn_list = [str(var1), str(var1_months), 'global', str(var2), str(var2_months), 'MeNmAz']
#     fn_core = '_'.join([i for i in fn_list if i != ''])

#     # filename and path
#     out_fn = f'corr_{fn_core}.nc'
#     out_fp = os.path.join(corr_out_path, 'global', out_fn)

#     # check existence of file already
#     if os.path.exists(out_fp):
#         print(f'File already exists for: {out_fn}')
#         print('\nSkipping . . .')
#         return

#     # open datasets

#     # var 1
#     if ((var1 in sfc_instan_list) or (var1 in sfc_accumu_list)):
#         var1_data = open_sfc_data(var=var1, region=var1_region, months=var1_month_list)
#     elif var1 in pl_var_list:
#         var1_data = open_pl_data(var=var1, p_level=var1_level, months=var1_month_list)
#     elif var1 in NAM_var_list:
#         var1_data = open_NAM_data(var=var1)
#     else:
#         print('Something went wront . . .')
#         return

#     # var 2
#     if ((var2 in sfc_instan_list) or (var2 in sfc_accumu_list)):
#         var2_data = open_sfc_data(var=var2, region=var2_region, months=var2_month_list)
#     elif var2 in pl_var_list:
#         var2_data = open_pl_data(var=var2, p_level=var2_level, months=var2_month_list)
#     elif var2 in NAM_var_list:
#         var2_data = open_NAM_data(var=var2)
#     else:
#         print('Something went wront . . .')
#         return

#     if ((var1_data is None) or (var2_data is None)):
#         print(f'No files were found a var: \n{var1_data}\n{var2_data}')
#         return

#     lats = slice(38,28)
#     lons = slice(246, 256)

#     var2_data = var2_data.sel(latitude=lats, longitude=lons).mean(dim=['latitude', 'longitude'])

#     # calculate correlation
#     if detrend_flag:
#         var1_data = detrend_data(var1_data, 'year')
#         if var2 not in NAM_var_list:
#             var2_data = detrend_data(var2_data, 'year')

#     # calculate correlation
#     var_corr = apply_correlation(var1_data, var2_data)
#     # return var_corr, out_fn

#     # save to netCDF file
#     var_corr.to_netcdf(out_fp)


# In[ ]:


# # test cell
# var1='sstk'
# var1_month_list=[3, 4, 5]
# var1_region='global'
# var2='onset'
# var2_month_list=[6, 7, 8]
# var2_region='MeNmAz'
# detrend_list=[True]
# for detrend_flag in detrend_list:
#     a, b = calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=700,
#                             var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=700,
#                             detrend_flag=detrend_flag, overwrite_flag=False)


# In[ ]:


var_list1 = ['ttr', 'sstk']
var_list2 = NAM_var_list + ['tp', 'vipile', 'viwve', 'viwvn', 'viwvd']
var1_months_list = [
    [3, 4, 5],
    [6, 7, 8]
]
detrend_list = [True, False]
len_lists = len(var_list1)*len(var_list2)*len(var1_months_list)*len(detrend_list)
cnt = 0
start_time = time.time()
for var1 in var_list1:
    for var2 in var_list2:
        for var1_month_list in var1_months_list:
            for detrend_flag in detrend_list:
                # print(f'var1 : {var1} {var1_month_list} -- var2 : {var2}')
                with open(f'{der_script_path}corr.txt', 'a') as file:
                    file.write(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}\n')

                calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region='global', var1_level=700,
                                        var2=var2, var2_month_list=[6, 7, 8], var2_region='MeNmAz', var2_level=700,
                                        detrend_flag=detrend_flag, overwrite_flag=False)
                cnt=cnt+1


# In[ ]:




