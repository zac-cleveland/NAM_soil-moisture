#!/usr/bin/env python
# coding: utf-8

# This script loops through the ERA5 data that I have chosen to subset and checks for certain values within those .nc files.  It's a sanity check to make sure that when I subset the data, I didn't accidently create nan values or other unrealistic values.

# In[1]:


# import functions
# OS interaction and time
import os
import sys
import glob
import dask
import dask.bag as db
import importlib

# math and data
import math
import numpy as np
import xarray as xr
import pandas as pd

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
from my_dictionaries import sfc_instan_list, sfc_accumu_list, pl_var_list, derived_var_list, invar_var_list, NAM_var_list, region_avg_list, flux_var_list, misc_var_list, var_dict, var_units, region_avg_dict, region_avg_coords, region_colors_dict


# In[9]:


def main(var, **kwargs):
    print(f"\n{'--'*40}\nProcessing var: {var_dict[var]}\n")

    # get the data that I subset previously
    my_data = get_var_data(var, level=500)
    num_value_my_data = check_for_values(my_data)
    value = kwargs.get('value_to_check', 'not_finite')
    if value == 'not_finite' or value == 'nan' or value == 'inf':
        if num_value_my_data == 0:
            print('No NANs or INFs found in my data. Skipping . . . ')
            return

    if var not in derived_var_list:
        print(f'NANs or INFs found in my data. Processing raw data\nvalue: {value}')
    else:
        print(f'{var_dict[var]} is a derived var. RDA data will not be processed.')
        print(f'my_data: {num_value_my_data}...')
        return

    # find the base sub directory of the data for var in rda_era5_path
    var_base_dir = find_var_base_dir(var, **kwargs)
    if not var_base_dir:
        print(f'No var_base_dir for: {var}')
        return
    else:
        print(f'var_base_dir: {var_base_dir}')

    # loop through years and count values in each dataset to compare to my_data
    for year in np.arange(1980,2020):
        print(f'{year}: ', end='')
        # get list of file paths for each .nc file of var
        files = get_rda_files(var, year, var_base_dir)
        if not files:
            print(f'\nNo files for {var_dict[var]} in {var_base_dir} for {year}')
            return
        if not all(os.path.exists(file) for file in files):
            print('\nSome files do not exist. Returning . . .')
            return

        try:
            # open dataset
            ds = xr.open_mfdataset(files)
            var_name = [v for v in ds.data_vars.keys() if var.upper() in v.upper()][0]  # actual variable name in Dataset
            da = ds[var_name]

            # subset the data by latitude and longitude coordinates and into daily values
            da_sub = subset_data_coords(da, **kwargs)
            da_daily = subset_data_daily(var, da_sub, **kwargs)

            my_data = my_data.sel(time=my_data['time.year'].isin(year))
            num_value_my_data = check_for_values(my_data)

            # check for specified values in the data
            num_value_da_sub = check_for_values(da_sub, **kwargs)
            num_value_da_daily = check_for_values(da_daily, **kwargs)
            print(f"da_sub: {num_value_da_sub}...\tda_daily: {num_value_da_daily}...\tmy_data: {num_value_my_data}...")
        except Exception as e:
            print(f'Error processing var: {var_dict[var]}.\n\n{e}')


# In[3]:


def find_var_base_dir(var, **kwargs):
    base_dir = kwargs.get('base_dir', rda_era5_path)  # default to rda_era5_path, but user can specify others
    # loop through rda_era5_path sub directories to find the base directory of var
    contents = os.listdir(base_dir)  # all contents in rda_era5_path
    dirs = [item for item in contents if os.path.isdir(os.path.join(base_dir, item))]  # just directories, not files
    for dir in dirs:
        files = os.listdir(f'{base_dir}{dir}/{198001}')
        for file in files:
            if f'_{var}.' in file:  # check for existence of the var key in the file names
                return os.path.join(base_dir, dir)
    return None  # sub directory not found


# In[4]:


def get_rda_files(var, year, var_base_dir, **kwargs):
    # loop through base_path and create a list of file paths for the .nc files for var
    files = glob.glob(f'{var_base_dir}/{year}??/*_{var}.*.nc')
    files.sort()
    return files


# In[5]:


def subset_data_coords(da, **kwargs):
    coords = [240, 260, 40, 20]
    lons = slice(coords[0], coords[1])
    lats = slice(coords[2], coords[3])
    # subset the data
    return da.sel(latitude=lats, longitude=lons, drop=True)


# In[6]:


def subset_data_daily(var, da, **kwargs):
    if var in sfc_accumu_list:
        da = da.sum(dim='forecast_hour', skipna=True).resample(forecast_initial_time='1D').sum(skipna=True)
        return da.rename({'forecast_initial_time': 'time'}) # rename time dimension
    elif var in sfc_instan_list:
        return da.resample(time='1D').mean('time', skipna=True)
    elif var in pl_var_list:
        return da.sel(level=500, drop=True).resample(time='1D').mean('time', skipna=True)


# In[7]:


def check_for_values(da, **kwargs):
    # check data for specified values
    value = str(kwargs.get('value_to_check', 'not_finite'))  # default to nan, but user can specify others
    if value == 'not_finite':
        return (~np.isfinite(da)).sum().compute().values
    elif value == 'nan':
        return da.isnull().sum().compute().values
    elif value == 'inf':
        return np.isinf(da).sum().compute().values
    elif value == 'finite':
        return np.isfinite(da).sum().compute().values
    else:
        return (da == float(value)).sum().compute().values


# In[ ]:


# # run the code
# if __name__ == '__main__':
#     main('cp')


# In[ ]:


if __name__ == '__main__':
    var_list = sfc_accumu_list + sfc_instan_list + pl_var_list
    for var in var_list:
        main(var)


# In[ ]:




