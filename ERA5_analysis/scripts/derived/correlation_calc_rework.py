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


# In[2]:


# specify directories
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to subset CP data
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
    'length',
    'precipitation',
    'precipitation-rate'
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

# misc variables
misc_var_list = [
    'nino-3',
]


# In[4]:


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
    'onset': 'NAM Onset',
    'retreat': 'NAM Retreat',
    'length': 'NAM Length',
    'precipitation': 'Yearly NAM Season Precipitation',
    'precipitation-rate': 'NAM Precipitation Rate',
    'nino-3': r'Ni$\tilda{n}$o-3 Index',
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


# In[5]:


# define a function to calculate the correlation between
# any two variables in certain months
def calc_correlation(var1='sd', var1_month_list=[3, 4, 5], var1_region='cp',
                     var2='tp', var2_month_list=[6, 7, 8], var2_region='dsw',
                     detrend_flag=True, overwrite_flag=False, **kwargs):

    var1_month_list = ensure_var_list(var1_month_list)
    var2_month_list = ensure_var_list(var2_month_list)

    # months list
    var1_months = month_num_to_name(var=var1, months=var1_month_list, **kwargs)
    var2_months = month_num_to_name(var=var2, months=var2_month_list, **kwargs)

    # filename and path

    # set detrend string for naming convention
    if detrend_flag:
        detrend_str = ''
    else:
        detrend_str = 'NOT-DETRENDED'

    # create list of var names, months, regions, etc. for naming convention
    fn_list = [str(var1), str(var1_months), str(var1_region),
               str(var2), str(var2_months), str(var2_region),
               str(detrend_str)]
    # core of the naming convention
    fn_core = '_'.join([i for i in fn_list if i != ''])
    out_fn = f'corr_{fn_core}.nc'

    # set the file path to save output
    # global correlations
    if ((var1_region == 'global') or (var2_region == 'global')):
        out_fp = os.path.join(my_era5_path, 'correlations', 'global', out_fn)
    # region to region correlations
    elif ((var1_region in region_avg_list) and (var2_region in region_avg_list)):
        out_fp = os.path.join(my_era5_path, 'correlations', f'regions/{var2_region}', out_fn)
    # correlations within the DSW (region to DSW or DSW to DSW)
    elif ((var1_region == 'dsw') or (var2_region == 'dsw')):
        out_fp = os.path.join(my_era5_path, 'correlations', 'dsw', out_fn)

    # check existence of file already
    if os.path.exists(out_fp):
        print(f'File already exists for: {out_fn}', end=' - ')
        if not overwrite_flag:  # don't continue
            print('overwrite_flag is set to False. Skipping . . .')
            return
        else:  # continue
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

    # apply detrend if needed
    if detrend_flag:
        var1_data = apply_detrend(var1_data)
        var2_data = apply_detrend(var2_data)

    # calculate correlation
    var_corr = apply_correlation(var1_data, var2_data)
    # return var_corr, out_fn

    # save to netCDF file
    var_corr.to_netcdf(out_fp)


# In[6]:


# define a function to check if inputs are list or not
def ensure_var_list(x):

    if not isinstance(x, list):
        return [x]
    return x


# In[7]:


# define a function to turn a list of integers into months
def month_num_to_name(var, months, **kwargs):

    # make string for month letters from var_range (e.g. [6,7,8] -> 'JJA')
    if var in NAM_var_list:
        var_months = ''  # don't use months for onset, retreat, length
    elif len(months) == 1:
        var_months = calendar.month_name[months[0]]  # use full month name if only 1 month
    elif ((len(months) > 1) & (len(months) <= 12)):
        var_months = ''.join([calendar.month_name[m][0] for m in months])  # make string of months, i.e. 3, 4, 5 is MAM
    return var_months


# In[8]:


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
        files = glob.glob(f'{my_era5_path}dsw/NAM_{var}.nc')

    elif var in misc_var_list:
        files = glob.glob(f'{misc_data_path}{var}/{var}*.nc')

    # if something went wrong
    else:
        files = []

    files.sort()
    return files


# In[9]:


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
    elif var.lower() in NAM_var_list:
        if ((var.lower() == 'onset') or (var.lower() == 'retreat')):
            var_data = da.dt.dayofyear
        else:
            var_data = da

    elif var in misc_var_list:
        var_data = da.sel(time=da['time.month'].isin(months)).groupby('time.year').mean(dim='time')

    # something went wrong
    else:
        return None

    # check if NAM var requested for regional average
    if ((var.lower() in NAM_var_list) & (region in region_avg_list)):
        lats = slice(region_avg_coords[region][2], region_avg_coords[region][3])
        lons = slice(region_avg_coords[region][0], region_avg_coords[region][1])
        var_data = var_data.sel(latitude=lats, longitude=lons).mean(dim=['latitude', 'longitude'], skipna=True)

    return var_data


# In[10]:


# define a function to detrend the data

# MANUALLY DETREND WITH LINEAR REGRESSION
def detrend_data(arr):

    # set up x array for the years
    arr_years = np.arange(0,40)

    # mask out nan values
    mask = np.isfinite(arr)
    arr_years_mask = arr_years[mask]
    arr_mask = arr[mask]

    # make sure the array is not full of non-finite values
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


# In[11]:


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


# # test cell -- dsw
# var1 = 'sd'
# var1_month_list = [3, 4, 5]
# var1_region = 'cp'
# var2 = 'precipitation'
# var2_month_list = [6, 7, 8]
# var2_region = 'dsw'
# detrend_list=[True]
# for detrend_flag in detrend_list:
#     calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region,
#                      detrend_flag=detrend_flag, overwrite_flag=False)


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
#                 calc_correlation(var1=var1, var1_month_list=[3, 4, 5], var1_region=region,
#                                  var2=var2, var2_month_list=[6, 7, 8], var2_region='dsw',
#                                  detrend_flag=detrend_flag, overwrite_flag=False)
#                 cnt = cnt+1


# In[ ]:


# # cell to compute specific correlations
# var_list1 = sfc_accumu_list + sfc_instan_list + pl_var_list
# if 'sstk' in var_list1:
#     var_list1.remove('sstk')
# var1_month_lists = [[3, 4, 5]]
# var_list2 = ['precipitation-rate']
# var2_month_lists = [[6, 7, 8]]
# len_lists = len(var_list1)*len(var_list2)*len(var1_month_lists)*len(var2_month_lists)
# detrend_flag_list = [True, False]
# cnt = 0
# for var1 in var_list1:
#     for var1_month_list in var1_month_lists:
#         for var2 in var_list2:
#             for var2_month_list in var2_month_lists:
#                 for detrend_flag in detrend_flag_list:
#                     # with open(f'{der_script_path}corr.txt', 'a') as file:
#                     #     file.write(f'{var1}\t:\t{var2}\t:\t{region}\t:\tdetrend={detrend_flag}\t:\t{100*cnt/len_lists} %\n')
#                     print(f'{var1}\t:\t{var1_month_list}\t:\t{var2}\t:\t{var2_month_list}\t:\t{100*cnt/len_lists} %')
#                     calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region='cp',
#                                      var2=var2, var2_month_list=var2_month_list, var2_region='dsw',
#                                      detrend_flag=detrend_flag, overwrite_flag=False)
#                     cnt = cnt+1


# In[ ]:


# # test cell -- global
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


# var_list1 = ['ttr', 'sstk']
# var_list2 = NAM_var_list + ['precipitation', 'tp', 'vipile', 'viwve', 'viwvn', 'viwvd']
# var1_months_list = [
#     [3, 4, 5],
#     [6, 7, 8]
# ]
# detrend_list = [True, False]
# len_lists = len(var_list1)*len(var_list2)*len(var1_months_list)*len(detrend_list)
# cnt = 0
# start_time = time.time()
# for var1 in var_list1:
#     for var2 in var_list2:
#         for var1_month_list in var1_months_list:
#             for detrend_flag in detrend_list:
#                 # print(f'var1 : {var1} {var1_month_list} -- var2 : {var2}')
#                 with open(f'{der_script_path}corr.txt', 'a') as file:
#                     file.write(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}\n')

#                 calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region='global', var1_level=700,
#                                         var2=var2, var2_month_list=[6, 7, 8], var2_region='MeNmAz', var2_level=700,
#                                         detrend_flag=detrend_flag, overwrite_flag=False)
#                 cnt=cnt+1


# In[ ]:


# # cell to compute specific correlations -- global
# var_list1 = ['ttr', 'sstk']
# var_list2 = ['precipitation-rate']
# var1_months_list = [
#     [3, 4, 5],
#     [6, 7, 8]
# ]
# detrend_list = [True, False]
# len_lists = len(var_list1)*len(var_list2)*len(var1_months_list)*len(detrend_list)
# cnt = 0
# start_time = time.time()
# for var1 in var_list1:
#     for var2 in var_list2:
#         for var1_month_list in var1_months_list:
#             for detrend_flag in detrend_list:
#                 # print(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{der_script_path}corr.txt', 'a') as file:
#                     file.write(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}\n')

#                 calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region='global', var1_level=700,
#                                         var2=var2, var2_month_list=[6, 7, 8], var2_region='MeNmAz', var2_level=700,
#                                         detrend_flag=detrend_flag, overwrite_flag=False)
#                 cnt=cnt+1


# In[12]:


# # cell to calculate nino-3 correlations of NAM
# var1 = 'nino-3'
# var1_months_list = [[i, i+1, i+2] for i in range(1,11)]  # create list of list [[1,2,3], [2,3,4], ... [10,11,12]]
# var1_region = ''
# # var2_list = ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate']
# var2_list = ['tp']
# var2_month_list = [6, 7, 8]
# var2_region = 'dsw'
# detrend_flag=True
# overwrite_flag=False
# for var2 in var2_list:
#     print(f'var - {var2}: ')
#     for var1_month_list in var1_months_list:
#         print(f'\t{var1_month_list}\t', end='')
#         calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region,
#                          var2=var2, var2_month_list=var2_month_list, var2_region=var2_region,
#                          detrend_flag=detrend_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# cell to calculate moving window correlations
var1_list = ['sd', 'swvl1', 'stl1', '2t', 'tp', 'sf', 'sshf', 'slhf', 'ssr', 'str', 'ssrd', 'strd', 'sro', 'z', 'u', 'v']
var1_months_list = [[i, i+1, i+2] for i in range(1,11)]  # create list of list [[1,2,3], [2,3,4], ... [10,11,12]]
var1_region = 'cp'

var2_list = ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'z', 'cape', 'cp']
var2_months_list = [[3, 4, 5], [6, 7, 8]]
var2_region = 'dsw'

detrend_flag=True
overwrite_flag=False
len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
start_time = time.time()
cnt=0
for var1 in var1_list:
    for var2 in var2_list:
        for var2_month_list in var2_months_list:
            for var1_month_list in var1_months_list:
                # print(f'{var1}:\t{var2}:\t{var2_month_list}:\t{var1_month_list}')
                with open(f'{der_script_path}corr.txt', 'a') as file:
                    file.write(f'{var1}:\t{var2}:\t{var2_month_list}:\t{var1_month_list}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
                calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region,
                                 var2=var2, var2_month_list=var2_month_list, var2_region=var2_region,
                                 detrend_flag=detrend_flag, overwrite_flag=overwrite_flag)
                cnt=cnt+1


# In[ ]:


# cell to calculate more moving window correlations
var1_list = ['onset', 'retreat', 'length']
var1_months_list = [3, 4, 5]
var1_region = 'cp'

var2_list = ['tp', 'precipitation', 'precipitation-rate']
var2_month_list = [[3, 4, 5], [6, 7, 8]]
var2_region = 'dsw'

detrend_flag=True
overwrite_flag=False
len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
start_time = time.time()
cnt=0
for var1 in var1_list:
    for var2 in var2_list:
        for var2_month_list in var2_months_list:
            # print(f'{var1}:\t{var2}:\t{var2_month_list}')
            with open(f'{der_script_path}corr.txt', 'a') as file:
                file.write(f'{var1}:\t{var2}:\t{var2_month_list}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
            calc_correlation(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region,
                             var2=var2, var2_month_list=var2_month_list, var2_region=var2_region,
                             detrend_flag=detrend_flag, overwrite_flag=overwrite_flag)
            cnt=cnt+1


# In[ ]:




