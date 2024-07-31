#!/usr/bin/env python
# coding: utf-8

# This script is used to subset ERA5 data by lat/lon and time (e.g., daily averages). Pressure level variables will be subset and saved in 1 file.

# In[8]:


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
if 'my_dictionaries' in sys.modules:
    importlib.reload(sys.modules['my_dictionaries'])
# import my lists and dictionaries
from my_dictionaries import var_dict


# In[9]:


# variables to subset
# SFC instantaneous variables
sfc_instan_vars = [
    # 'sd',  # snow depth  (m of water equivalent)
    # 'msl',  # mean sea level pressure (Pa)
    # 'stl1',  # soil temp layer 1 (K)
    # 'swvl1',  # soil volume water content layer 1 (m^3 m^-3)
    # '2t',  # 2 meter temp (K)
    # '2d',  # 2 meter dew point (K)
    # 'cape',  # convective available potential energy (J kg^-1)
    # 'tcw',  # total column water (kg m^-2) -- sum total of solid, liquid, and vapor in a column
    # 'sstk',  # sea surface temperature (K)
    # 'viwve',  # vertical integral of eastward water vapour flux (kg m^-1 s^-1) - positive south -> north
    # 'viwvn',  # vertical integral of northward water vapour flux (kg m^-1 s^-1) - positive west -> east
    # 'viwvd',  # vertical integral of divergence of moisture flux (kg m^-2 s^-1) - positive divergencve
]

# surface accumulation variables
sfc_accumu_vars = [
    # 'lsp',  # large scale precipitation (m of water)
    # 'cp',  # convective precipitation (m of water)
    # 'sshf',  # surface sensible heat flux (J m^-2)
    # 'slhf',  # surface latent heat flux (J m^-2)
    # 'ssr',  # surface net solar radiation (J m^-2)
    # 'str',  # surface net thermal radiation (J m^-2)
    # 'sf',  # total snowfall (m of water equivalent)
    # 'ssrd',  # surface solar radiation downwards (J m^-2)
    # 'strd',  # surface thermal radiation downwards (J m^-2)
    # 'ttr',  # top net thermal radiation (OLR, J m^-2) -- divide by time (s) for W m^-2
]

# pressure level variables
pl_vars = [
    'z',  # geopotential (m^2 s^2)
    # 't',  # temperature (K)
    # 'u',  # u component of wind(m s^-1)
    # 'v',  # v component of wind (m s^-1)
    # 'q',  # specific humidity (kg kg^-1)
]

regions = {
    'dsw': {'latitude': slice(40, 20), 'longitude': slice(240, 260)},
    'WestUS_Mexico': {'latitude': slice(50, 10), 'longitude': slice(230, 270)},
    'global': None
}


# In[3]:


def find_var_directory(var, rda_era5_path):
    contents = os.listdir(rda_era5_path)  # all contents in rda_era5_path
    # only get directories from rda_era5_path  - disclude invariant directory
    directories = [item for item in contents if os.path.isdir(os.path.join(rda_era5_path, item)) and item != 'e5.oper.invariant']
    for dir in directories:
        files = os.listdir(os.path.join(rda_era5_path, dir, '197901'))
        for file in files:
            if f'_{var}.' in file:  # check for existence of the var key in the file names
                return dir
    raise FileNotFoundError(f"var: {var} not found in {rda_era5_path}")  # if no file found containing var


# In[4]:


def get_out_fn(var, region, year, month):
    return f'{var}_{year}{month:02d}_{region}.nc' if month else f'{var}_{year}01_{year}12_{region}.nc'


def get_out_fp(region, year, out_fn):
    return os.path.join(my_era5_path, region, f'{year}', out_fn)


# In[5]:


def find_var_files(var, year, month, var_base_dir):
    month_pattern = f'{year}{month:02d}' if month else f'{year}*'
    file_pattern = os.path.join(rda_era5_path, var_base_dir, month_pattern, f'*_{var}.*.nc')
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"files not found for {var} in {var_base_dir}")
    return files


# In[6]:


def subset_lat_lon(ds, region):
    region_slice = regions.get(region)
    if region_slice:
        ds =  ds.sel(latitude=region_slice['latitude'], longitude=region_slice['longitude'], drop=True)
    elif region == 'global':
        pass  # bad form, but I want the extra check to make sure I'm not crazy here......
    else:
        raise ValueError(f"Not a valid region: {region}")
    return ds


# In[7]:


def subset_pl(var_ds, **kwargs):
    var_ds = var_ds.sel(level=kwargs.get('pl_levels', [1000, 850, 700, 500, 300]))
    return var_ds


# In[8]:


def calc_daily_values(var, da, **kwargs):
    if var in sfc_instan_vars or var in pl_vars:
        da_daily = da.resample(time='1D').mean('time', skipna=True)
    elif var in sfc_accumu_vars:
        da_daily = da.sum('forecast_hour', skipna=True).resample(forecast_initial_time='1D').sum(skipna=True).rename({'forecast_initial_time': 'time'})
    else:
        raise ValueError(f"Not able to compute daily values for: {var}.")
    return da_daily


# In[9]:


def process_var_data(var, region, year, month, var_base_dir, **kwargs):
    # get var files for current year
    var_files = find_var_files(var, year, month, var_base_dir)

    # open datasets
    if var not in pl_vars:
        chunks = {'time': 24} if var not in sfc_accumu_vars else {'forecast_initial_time': 1, 'forecast_hour': 12}
    else:
        chunks = {'time': 24, 'level': 1}
    var_ds = xr.open_mfdataset(var_files, chunks=chunks, parallel=True)  # chunk by daily

    # subset by latitude/longitude
    var_ds = subset_lat_lon(var_ds, region)

    # subset by pressure level
    if var in pl_vars:
        var_ds = subset_pl(var_ds, **kwargs)

    # pull out variable name in actual dataset since they can be different
    var_name = [v for v in var_ds.data_vars.keys() if f'{var.upper()}' in v.upper()][0]
    var_da = var_ds[var_name]

    # compute daily values
    var_daily = calc_daily_values(var, var_da)
    return var_daily


# In[10]:


def main(var, region='dsw', year=1980, month=None, **kwargs):

    # find file base path in the rda dataset directory
    var_base_dir = find_var_directory(var, rda_era5_path)
    if not var_base_dir:
        raise ValueError(f"var_base_dir not found")

    # make output file name and path for saving
    out_fn = get_out_fn(var, region, year, month)
    out_fp = get_out_fp(region, year, out_fn)

    # check existence of file already
    if os.path.exists(out_fp):
        print(f"\nfile already exists for: {out_fn}.")
        if not kwargs.get('overwrite_flag', False):
            raise ValueError("overwrite_flag is False, set to True to overwrite. Skipping . . .")
        else:
            print("\noverwrite_flag is True. Overwriting . . .")

    # process data
    var_daily = process_var_data(var, region, year, month, var_base_dir, **kwargs)
    # save or return data
    if kwargs.get('save_nc', False):
        var_daily.to_netcdf(out_fp)
    else:
        return var_daily


# In[ ]:


# # test cell to check data processed correctly
# if __name__ == '__main__':
#     # args
#     region = 'WestUS_Mexico'
#     years = np.arange(1980,2020)
#     months = np.arange(1,13)
#     levels = [1000, 850, 700, 500, 300]

#     # kwargs
#     main_kwargs = {
#     'save_nc': True,
#     'overwrite_flag': False,
#     }

#     # set variable
#     var = 'z'
#     kwargs = main_kwargs.copy()
#     if var in pl_vars:  # add levels to kwargs if var is a pl var
#         kwargs.update({'pl_levels': levels})
#     print(f"\n\n{'- - '*20}\n\t\tProcessing: {var_dict[var]} {kwargs.get('pl_levels', '')}\n{'- - '*20}")

#     # set year
#     year = 1980
#     print(f"\n{year}: ", end='')
#     # set months
#     # month = None
#     month = 12
#     try:
#         if kwargs.get('save_nc', False):
#             main(var, region=region, year=year, month=month, **kwargs)
#         else:
#             var_da = main(var, region=region, year=year, month=month, **kwargs)
#     except Exception as e:
#         print(f"\nException raised: {e}")


# In[ ]:


# # call main function to subset non pressure level data (yearly)
# if __name__ == '__main__':

#     # args
#     region = 'WestUS_Mexico'
#     years = np.arange(1980,2020)
#     levels = [1000, 850, 700, 500, 300]

#     # kwargs
#     main_kwargs = {
#     'overwrite_flag': False,
#     }

#     # set up list of variables to process
#     var_list = sfc_instan_vars + sfc_accumu_vars
#     # loop through vars
#     for i, var in enumerate(var_list, start=1):
#         kwargs = main_kwargs.copy()
        
#         if var in pl_vars:  # add levels to kwargs if var is a pl var
#             kwargs.update({'pl_levels': levels})
#         print(f"\n\n{'- - '*20}\n\t\tProcessing var {i} of {len(var_list)}: {var_dict[var]}\n{'- - '*20}\n")

#         for year in years:
#             print(f"{year}...", end='')
#             try:
#                 main(var, region=region, year=year, **kwargs)
#             except Exception as e:
#                 print(f"\nException raised: {e}")


# In[ ]:


# call main function to subset pressure level data (monthly)
if __name__ == '__main__':

    # args
    region = 'WestUS_Mexico'
    years = np.arange(1980,2020)
    months = np.arange(1,13)
    levels = [1000, 850, 700, 500, 300]

    # kwargs
    main_kwargs = {
    'save_nc': True,
    'overwrite_flag': False,
    }

    # set up list of variables to process
    var_list = pl_vars
    # loop through vars
    for i, var in enumerate(var_list, start=1):
        kwargs = main_kwargs.copy()
        
        if var in pl_vars:  # add levels to kwargs if var is a pl var
            kwargs.update({'pl_levels': levels})
        print(f"\n\n{'- - '*20}\n\t\tProcessing var {i} of {len(var_list)}: {var_dict[var]}\n{'- - '*20}", flush=True)

        for year in years:
            print(f"\n{year}: ", end='', flush=True)
            for month in months:
                print(f" {month}...", end='', flush=True)
                try:
                    main(var, region=region, year=year, month=month, **kwargs)
                except Exception as e:
                    print(f"\nException raised: {e}", flush=True)


# In[3]:


# def find_small_files(base_dir, size_threshold=1 * 1024 * 1024):  # 1 MB threshold
#     small_files = []
#     for subdir, _, files in os.walk(base_dir):
#         for file in files:
#             if file.endswith('.nc'):
#                 file_path = os.path.join(subdir, file)
#                 file_size = os.path.getsize(file_path)
#                 if file_size < size_threshold:
#                     small_files.append((file_path, file_size))

#     return small_files


# In[4]:


# def check_for_files(vars, region):
#     missing_files = {}
#     if not isinstance(vars, list):
#         vars = [vars]
#     for var in vars:
#         missing_files[f'{var}_{region}'] = []
#         for year in range(1980,2020):
#             fn = get_out_fn(var, region, year)
#             fp = get_out_fp(region, year, fn)
#             if not os.path.exists(fp):
#                 missing_files[f'{var}_{region}'].append(f'{year}')
#     return missing_files


# In[ ]:


# # run find_small_files to check for incomplete files
# if __name__ == '__main__':
#     base_dir = '/glade/u/home/zcleveland/scratch/ERA5/WestUS_Mexico/'
#     small_files = find_small_files(base_dir)
    
#     for file_path, file_size in small_files:
#         print(f"{file_path}: {file_size} bytes")

#     missing_files = check_for_files(sfc_instan_vars+sfc_accumu_vars+pl_vars, 'WestUS_Mexico')
#     for var, year in missing_files.items():
#         if missing_files[f'{var}']:
#             print(f'{var}: {year}')

