#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:

out_text_file = []

era5_path = '/glade/campaign/collections/rda/data/ds633.0/' # base path to ERA5 data on derecho
out_path = '/glade/u/home/zcleveland/scratch/ERA5/dsw/' # base path to my subsetted data


# In[16]:


# Add other variables and their corresponding subdirectories here
var_list = [
    # 'lsp', # large scale precipitation (m of water) - accumu
    # 'cp', # convective precipitation (m of water) - accumu
    # 'sd', # snow depth  (m of water equivalent) - instan
    # 'msl', # mean sea level pressure (Pa) - instan
    # 'tcc', # total cloud cover (0-1) - instan
    # 'stl1', # soil temp layer 1 (K) - instan
    # 'stl2', # soil temp layer 2 (K) - instan
    # 'stl3', # soil temp layer 3 (K) - instan
    # 'stl4', # soil temp layer 4 (K) - instan
    # 'swvl1', # soil volume water content layer 1 (m^3 m^-3) - instan
    # 'swvl2', # soil volume water content layer 2 (m^3 m^-3) - instan
    # 'swvl3', # soil volume water content layer 3 (m^3 m^-3) - instan
    # 'swvl4', # soil volume water content layer 4 (m^3 m^-3) - instan
    # '2t', # 2 meter temp (K) - instan
    # '2d', # 2 meter dew point (K) - instan
    # 'ishf', # instant surface heat flux (W m^-2) - instan
    # 'ie', # instant moisture flux (kg m^-2 s^-1) - instan
    # 'sshf', # surface sensible heat flux (J m^-2) - accumu
    # 'slhf', # surface latent heat flux (J m^-2) - accumu
    # 'ssr', # surface net solar radiation (J m^-2) - accumu
    # 'str', # surface net thermal radiation (J m^-2) - accumu
    # 'sro', # surface runoff (m) - accumu
    # 'sf', # total snowfall (m of water equivalent) - accumu
    # 'cape', # convective available potential energy (J kg^-1) - instan
    'tcw', # total column water (kg m^-2) - sfc (sum total of solid, liquid, and vapor in a column)
]


# In[17]:


# function to extract, subset, and process data for single variable in the desert southwest
def dsw_subset_era5(variable='lsp', start_date=200101, end_date=200102):
    print(f'Processing variable: {variable}: {start_date}_{end_date}\n')

    # exit if not yearly data
    if (end_date-start_date)>11: 
        print(f'Time range greater than 1 year. Skipping... ')
        print(f'start_date: {start_date}\n end_date: {end_date}\n')
        return
    
    start_time = time.time() # keep track of time to process.
    # split start and end date to get year and month
    start_year, start_month = f'{start_date}'[:4], f'{start_date}'[4:]
    end_year, end_month = f'{end_date}'[:4], f'{end_date}'[4:]
    # define output filename and path
    out_fn = f'{variable}_{start_date}_{end_date}_dsw' # out file name
    out_fp = f'{out_path}{start_year}/{out_fn}' # out file path (including file name)

    # check if file already exists
    if (os.path.exists(f'{out_fp}.nc') or 
        os.path.exists(f'{out_fp}_min.nc') or 
        os.path.exists(f'{out_fp}_max.nc') or
        os.path.exists(f'{out_fp}_avg.nc')):
        
        print(f'File {out_fn} already exists. Skipping...\n')
        return
    
    # find the directory that the variable exists in
    print(f'Searching for {var} in {era5_path}\n')
    contents = os.listdir(era5_path) # all contents in era5_path
    # only get directories from era5_path
    directories = [item for item in contents if os.path.isdir(os.path.join(era5_path, item))]
    found = [] # initialize loop exit condition
    for dir in directories:    
        files = os.listdir(f'{era5_path}{dir}/{197901}') 
        for file in files:
            if f'_{variable}.' in file: # check for existence of the var key in the file names
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
    for year in range(int(start_year), int(end_year)+1): # input should be same year, so add 1 to end for "range" function
        for month in range(1,13): # months 1-12 (jan-dec)
            if month<10: # add a '0' to match date string format
                year_month = f'{year}0{month}'
            else:
                year_month = f'{year}{month}'
            try:
                if ((f'{year_month}' < f'{start_date}') or (f'{year_month}' > f'{end_date}')):
                    pass # in case the date is outside the range, just pass it
                else:
                    files += glob.glob(f'{era5_path}/{var_dir}/{year_month}/*_{variable}.*.nc', recursive=True)
            except Exception as e:
                print(f'Error in {era5_path}/{var_dir}/{year}0{month}/*_{variable}.*.nc: {e}\n')
    files.sort()

    # calculate total number of directories for sanity check
    total_directories = len(files)
    print(f'{total_directories} number of files\n')
    
    # create a list to hold data for each month
    with dask.config.set(**{'array.slicing.split_large_chunks': True}): # only for systems with dask
        data_list = []
        print(f'opening datasets')
        data_list.append(xr.open_mfdataset(files))

        # check if there are actually any files
        if not data_list:
            print(f'No files found for variable: {variable}')
            print(f'\rTime elapsed: {time.time()-start_time: .2f} s\n')
            return

        # concatenate data for all months into single xarray dataset
        print(f'Combining Data')
        combined_data = xr.merge(data_list)

        # subset the data for the Desert Southwest
        lat_range = slice(40, 20) # lat range 20 N to 40 N
        lon_range = slice(240, 260) # lon range 240 (120 W) to 260 (100 W)
        ds_sub = combined_data.sel(latitude=lat_range, longitude=lon_range, drop=True)

        # dimensions of accumulation data, instant data, etc.
        acc_dims = ['forecast_hour', 'forecast_initial_time', 'latitude', 'longitude']
        inst_dims = ['latitude', 'longitude', 'time']
        
        ### FOR ACCUMULCATIONS ###
        if set(ds_sub.dims) == set(acc_dims):
            # calculate sum total
            daily_data = ds_sub.sum(dim='forecast_hour').resample(forecast_initial_time='1D').sum()
            daily_data = daily_data.rename({'forecast_initial_time': 'time'}) # rename time dimension
            # write data to NetCDF file
            print(f'Writing data to NetCDF\n')
            daily_data.to_netcdf(f'{out_fp}.nc')

        ### FOR DAILY AVERAGE, MIN, AND MAX ###
        elif set(ds_sub.dims) == set(inst_dims):

            temp = ds_sub.resample(time='1D') # turn into daily data

            # find average and rename the variable to include AVG
            daily_avg = temp.mean(dim='time')
            var_xx = [varx for varx in daily_avg.data_vars.keys() if f'{variable.upper()}' in varx][0]
            daily_avg = daily_avg.rename_vars({f'{var_xx}': f'{var_xx}_AVG'})

            # find maximum and rename the variable to include MAX
            daily_max = temp.max(dim='time')
            var_xx = [varx for varx in daily_max.data_vars.keys() if f'{variable.upper()}' in varx][0]
            daily_max = daily_max.rename_vars({f'{var_xx}': f'{var_xx}_MAX'})

            # find minimum and rename the variable to include MIN
            daily_min = temp.min(dim='time')
            var_xx = [varx for varx in daily_min.data_vars.keys() if f'{variable.upper()}' in varx][0]
            daily_min = daily_min.rename_vars({f'{var_xx}': f'{var_xx}_MIN'})

            # combining data
            print(f'Combining data')
            daily_data = xr.merge([daily_avg,daily_max,daily_min], compat='override')
            
            # write data to NetCDF file
            print(f'Writing data to NetCDF\n')
            daily_data.to_netcdf(f'{out_fp}.nc')
            
        else:
            print(F'Dimensional error finding daily values for {var}') 
            print(f'Dimensions are {sorted(ds_sub.dims)}.\n Skipping...\n')
            return


        print(f'\rTime elapsed: {time.time()-start_time: .2f} s\n')

    #return combined_data, ds_sub, ds_sub_daily


# In[28]:


# set time array to loop through
years = np.arange(1980,2020)
months = np.arange(1,13)


# In[20]:


# Loop through variables in var_directories and process each one
for var in var_list:
    for year in years:
        start_date = int(f'{year}01')
        end_date = int(f'{year}12')
        dsw_subset_era5(var, start_date, end_date)


# In[ ]:




