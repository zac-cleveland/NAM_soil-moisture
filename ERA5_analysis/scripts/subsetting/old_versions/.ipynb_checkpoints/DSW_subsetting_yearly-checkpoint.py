#!/usr/bin/env python
# coding: utf-8

# In[30]:


import sys

import numpy as np
import netCDF4 as nc
import numpy.matlib
import datetime
import xarray as xr
from scipy import interpolate
from numpy import ma
from scipy import stats
import scipy.io as sio
import pickle as pickle
from sklearn import linear_model
import numpy.ma as ma
import matplotlib.patches as mpatches
from shapely.geometry.polygon import LinearRing

import scipy as sp
import pandas as pd

import time

from copy import copy 

# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider
import mpl_toolkits.axes_grid1.axes_size as Size
from mpl_toolkits.axes_grid1 import make_axes_locatable

# OS interaction
import os
import sys
import cftime

import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

import matplotlib.colors as mcolors

import glob
import dask
import dask.bag as db

from scipy import interpolate

import statsmodels.stats.multitest as multitest

from matplotlib.gridspec import GridSpec
from cartopy.crs import EqualEarth, PlateCarree


# In[31]:


era5_path = '/glade/campaign/collections/rda/data/ds633.0/'
out_path = '/glade/u/home/zcleveland/scratch/ERA5/dsw/'


# In[32]:


# Add other variables and their corresponding subdirectories here
var_list = [
    'lsp', # large scale precipitation
    'cp', # convective precipitation
    'sd', # snow depth
    'msl', # mean sea level pressure
    'tcc', # total cloud cover
    'stl1', # soil temp layer 1
    'stl2', # soil temp layer 2
    'stl3', # soil temp layer 3
    'stl4', # soil temp layer 4
    'swvl1', # soil volume water content layer 1
    'swvl2', # soil volume water content layer 2
    'swvl3', # soil volume water content layer 3
    'swvl4', # soil volume water content layer 4
    '2t', # 2 meter temp
    '2d', # 2 meter dew point
    'ishf', # instant surface heat flux
    'ie', # instant moisture flux
]
# var_list = ['lsp', ]
# Input parameters for desert southwest
lat_range = slice(40, 20)
lon_range = slice(240, 260)


# In[33]:


# function to extract, subset, and process data for single variable in the desert southwest
def dsw_subset_era5(variable='lsp', start_date=200101, end_date=200102):
    print(f'Processing variable: {variable}: {start_date}_{end_date}\n')

    # exit if not yearly data
    if (end_date-start_date)>11: 
        print(f'Time range greater than 1 year. Skipping... ')
        print(f'start_date: {start_date}\n end_date: {end_date}\n')
        return
    
    start_time = time.time() # keep track of time to process.
    start_year, start_month = f'{start_date}'[:4], f'{start_date}'[4:]
    end_year, end_month = f'{end_date}'[:4], f'{end_date}'[4:]
    # define output filename and path
    out_fn = f'{variable}_{start_date}_{end_date}_dsw'
    out_fp = os.path.join(out_path, out_fn)

    # check if file already exists
    if (os.path.exists(f'{out_fp}.nc') or 
        os.path.exists(f'{out_fp}_min.nc') or 
        os.path.exists(f'{out_fp}_max.nc') or
        os.path.exists(f'{out_fp}_avg.nc')):
        
        print(f'File {out_fn} already exists. Skipping...\n')
        return
    
    # find the directory that the variable exists in
    print(f'Searching for {var} in {era5_path}\n')
    contents = os.listdir(era5_path)
    directories = [item for item in contents if os.path.isdir(os.path.join(era5_path, item))]
    found = []
    for dir in directories:    
        files = os.listdir(f'{era5_path}{dir}/{197901}')
        for file in files:
            if f'_{variable}.' in file:
                print(f'{var} found at {dir}\n')
                found.append(1)
                var_dir = dir
                break
        if found:
            break
    if not found:
        print(f'No files found with {variable}. Skipping...\n')
        return
        
    # find files for the variable in the specified directory
    files = []
    for year in range(int(start_year), int(end_year)+1):
        for month in range(1,13):
            if month<10:
                year_month = f'{year}0{month}'
            else:
                year_month = f'{year}{month}'
            try:
                if ((f'{year_month}' < f'{start_date}') or (f'{year_month}' > f'{end_date}')):
                    pass
                else:
                    files += glob.glob(f'{era5_path}/{var_dir}/{year_month}/*_{variable}.*.nc', recursive=True)
            except Exception as e:
                print(f'Error in {era5_path}/{var_dir}/{year}0{month}/*_{variable}.*.nc: {e}\n')
    files.sort()

    # calculate total number of directories for sanity check
    total_directories = len(files)
    print(f'{total_directories} number of files\n')
    
    # create a list to hold data for each month
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        data_list = []
        print(f'opening datasets')
        data_list.append(xr.open_mfdataset(files))

        # check if there are actually any files
        if not data_list:
            print(f'No files found for variable: {variable}')
            print(f'\rTime elapsed: {time.time()-start_time: .2f} s\n')
            return

        # concatenate data for all months into single xarray and subset
        print(f'Combining Data')
        combined_data = xr.merge(data_list)
        ds_sub = combined_data.sel(latitude=lat_range, longitude=lon_range, drop=True)
        
        ### FOR ACCUMULCATIONS ###
        if sorted(ds_sub.dims) == ['forecast_hour', 'forecast_initial_time', 'latitude', 'longitude']:
            # calculate sum total
            daily_data = ds_sub.sum(dim='forecast_hour').resample(forecast_initial_time='1D').sum()
            daily_data = daily_data.rename({'forecast_initial_time': 'time'})
            # write data to NetCDF file
            print(f'Writing data to NetCDF\n')
            daily_data.to_netcdf(f'{out_fp}.nc')

        ### FOR DAILY AVERAGE, MIN, AND MAX
        elif sorted(ds_sub.dims) == ['latitude', 'longitude', 'time']:
            # calculate daily average, min, and max
            temp = ds_sub.resample(time='1D')
            daily_avg = temp.mean(dim='time')
            daily_min = temp.min(dim='time')
            daily_max = temp.max(dim='time')
            # write data to NetCDF file
            print(f'Writing data to NetCDF\n')
            daily_avg.to_netcdf(f'{out_fp}_avg.nc')
            daily_min.to_netcdf(f'{out_fp}_min.nc')
            daily_max.to_netcdf(f'{out_fp}_max.nc')
            
        else:
            print(F'Dimensional error finding daily values for {var}') 
            print(f'Dimensions are {sorted(ds_sub.dims)}.\n Skipping...\n')
            return


        print(f'\rTime elapsed: {time.time()-start_time: .2f} s\n')

    #return combined_data, ds_sub, ds_sub_daily


# In[38]:


# set time array to loop through
years = np.arange(2016,2020)
months = np.arange(1,13)

d~
# In[37]:


# Loop through variables in var_directories and process each one
for var in var_list:
    for year in years:
        start_date = int(f'{year}01')
        end_date = int(f'{year}12')
        dsw_subset_era5(var, start_date, end_date)


# In[ ]:




