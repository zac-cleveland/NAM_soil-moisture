#!/usr/bin/env python
# coding: utf-8

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


era5_path = '/glade/campaign/collections/rda/data/ds633.0/'  # base path to ERA5 data on derecho
out_path = '/glade/u/home/zcleveland/scratch/ERA5/dsw/'  # base path to my subsetted data
era5_invariant_path = '/glade/campaign/collections/rda/data/ds633.0/e5.oper.invariant/197901/'  # path to ERA5 directory for invariant data
my_invariant_path = '/glade/u/home/zcleveland/scratch/ERA5/invariants/'  # path to my invariant directory
era5_pl_path = '/glade/campaign/collections/rda/data/ds633.0/e5.oper.an.pl/'  # path to ERA5 pressure level data
sub_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/'  # path to my subsetting scripts


# In[5]:


# variable list to choose
var_list = [
    # 'lsp',  # large scale precipitation (m of water) - accumu
    # 'cp',  # convective precipitation (m of water) - accumu
    # 'tp',  # total precipitation (m of water) - accumu -- DERIVED
    # 'sd',  # snow depth  (m of water equivalent) - instan
    # 'msl',  # mean sea level pressure (Pa) - instan
    # 'tcc',  # total cloud cover (0-1) - instan
    # 'stl1',  # soil temp layer 1 (K) - instan
    # 'stl2',  # soil temp layer 2 (K) - instan
    # 'stl3',  # soil temp layer 3 (K) - instan
    # 'stl4',  # soil temp layer 4 (K) - instan
    # 'swvl1',  # soil volume water content layer 1 (m^3 m^-3) - instan
    # 'swvl2',  # soil volume water content layer 2 (m^3 m^-3) - instan
    # 'swvl3',  # soil volume water content layer 3 (m^3 m^-3) - instan
    # 'swvl4',  # soil volume water content layer 4 (m^3 m^-3) - instan
    # '2t',  # 2 meter temp (K) - instan
    # '2d',  # 2 meter dew point (K) - instan
    # 'ishf',  # instant surface heat flux (W m^-2) - instan
    # 'ie',  # instant moisture flux (kg m^-2 s^-1) - instan
    # 'sshf',  # surface sensible heat flux (J m^-2) - accumu
    # 'slhf',  # surface latent heat flux (J m^-2) - accumu
    # 'ssr',  # surface net solar radiation (J m^-2) - accumu
    # 'str',  # surface net thermal radiation (J m^-2) - accumu
    # 'sro',  # surface runoff (m) - accumu
    # 'sf',  # total snowfall (m of water equivalent) - accumu
    # 'cape',  # convective available potential energy (J kg^-1) - instan
    # 'tcw',  # total column water (kg m^-2) - sfc (sum total of solid, liquid, and vapor in a column)
    # 'ssrd',  # surface solar radiation downwards (J m^-2) - accumu
    # 'strd',  # surface thermal radiation downwards (J m^-2) - accumu
]


# In[7]:


invariant_list = [
    # 'cl',  # lake cover (0-1)
    # 'dl',  # lake depth (m)
    # 'cvl',  # low vegetation cover (0-1)
    # 'cvh',  # high vegetation cover (0-1)
    # 'tvl',  # type of low vegetation (~)
    # 'tvh',  # type of high vegetation (~)
    # 'slt',  # soil type1 (~)
    # 'sdfor',  # standard deviation of filtered subgrid orography (m)
    # 'z',  # geopotential (m^2 s^-2)
    # 'sdor',  # standard deviation of orography (~)
    # 'isor',  # anisotropy of sub-gridscale orography (~)
    # 'anor',  # angle of sub-gridscale orography (radians)
    # 'slor',  # slope of sub-gridscale orography (~)
    # 'lsm',  # land-sea mask (0-1)
]


# In[3]:


pl_var_list = [
    'pv',  # potential vorticity (K m^2 kg^-1 s^-1)
    'crwc',  # specific rain water content (kg kg^-1)
    'cswc',  # specific snow water content (kg kg^-1)
    'z',  # geopotential (m^2 s^2)
    't',  # temperature (K)
    'u',  # u component of wind(m s^-1)
    'v',  # v component of wind (m s^-1)
    'q',  # specific humidity (kg kg^-1)
    'w',  # vertical velocity (Pa s^-1)
    'vo',  # vorticity - relative (s^-1)
    'd',  # divergence (s^-1)
    'r',  # relative humidity (%)
    'clwc',  # specific cloud liquid water content
    'ciwc',  # specific cloud ice water content
    'cc',  # fraction of cloud cover (0-1)
]


# In[44]:


# function to extract, subset, and process data for single variable in the desert southwest
def dsw_subset_era5(variable='lsp', start_date=200101, end_date=200102):
    print(f'. . . Processing variable: {variable}: {start_date}_{end_date} . . .\n')

    # exit if not yearly data
    if (end_date-start_date)>11:
        print(f'Time range greater than 1 year. Skipping... ')
        print(f'start_date: {start_date}\n end_date: {end_date}\n')
        return

    start_time = time.time()  # keep track of time to process.
    # split start and end date to get year and month
    start_year, start_month = f'{start_date}'[:4], f'{start_date}'[4:]
    end_year, end_month = f'{end_date}'[:4], f'{end_date}'[4:]
    # define output filename and path
    out_fn = f'{variable}_{start_date}_{end_date}_dsw'  # out file name
    out_fp = f'{out_path}{start_year}/{out_fn}'  # out file path (including file name)

    # check if file already exists
    if (os.path.exists(f'{out_fp}.nc') or
        os.path.exists(f'{out_fp}_min.nc') or
        os.path.exists(f'{out_fp}_max.nc') or
        os.path.exists(f'{out_fp}_avg.nc')):

        print(f'File {out_fn} already exists. Skipping...\n')
        return

    # find the directory that the variable exists in
    print(f'Searching for {variable} in {era5_path}\n')
    contents = os.listdir(era5_path)  # all contents in era5_path
    # only get directories from era5_path
    directories = [item for item in contents if os.path.isdir(os.path.join(era5_path, item))]
    found = []  # initialize loop exit condition
    for dir in directories:
        files = os.listdir(f'{era5_path}{dir}/{197901}')
        for file in files:
            if f'_{variable}.' in file:  # check for existence of the var key in the file names
                print(f'{variable} found at {dir}\n')
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
    for year in range(int(start_year), int(end_year)+1):  # input should be same year, so add 1 to end for "range" function
        for month in range(1, 13):  # months 1-12 (jan-dec)
            if month < 10:  # add a '0' to match date string format
                year_month = f'{year}0{month}'
            else:
                year_month = f'{year}{month}'
            try:
                if ((f'{year_month}' < f'{start_date}') or (f'{year_month}' > f'{end_date}')):
                    pass  # in case the date is outside the range, just pass it
                else:
                    files += glob.glob(f'{era5_path}/{var_dir}/{year_month}/*_{variable}.*.nc', recursive=True)
            except Exception as e:
                print(f'Error in {era5_path}/{var_dir}/{year}0{month}/*_{variable}.*.nc: {e}\n')
    files.sort()

    # calculate total number of directories for sanity check
    total_directories = len(files)
    print(f'{total_directories} number of files\n')

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):

        # open datasets
        print(f'opening datasets')
        ds = xr.open_mfdataset(files)

        # subset the data for the Desert Southwest
        lat_range = slice(40, 20)  # lat range 20 N to 40 N
        lon_range = slice(240, 260)  # lon range 240 (120 W) to 260 (100 W)
        ds_sub = ds.sel(latitude=lat_range, longitude=lon_range, drop=True)

        # dimensions of accumulation data, instant data, etc.
        acc_dims = ['forecast_hour', 'forecast_initial_time', 'latitude', 'longitude']
        inst_dims = ['latitude', 'longitude', 'time']
        pl_dims = ['latitude', 'level', 'longitude', 'time']

        ### FOR ACCUMULCATIONS ###
        if set(ds_sub.dims) == set(acc_dims):
            # calculate sum total
            daily_data = ds_sub.sum(dim='forecast_hour').resample(forecast_initial_time='1D').sum()
            daily_data = daily_data.rename({'forecast_initial_time': 'time'}) # rename time dimension

        ### FOR DAILY AVERAGE, MIN, AND MAX ###
        elif set(ds_sub.dims) == set(inst_dims):

            temp = ds_sub.resample(time='1D')  # turn into daily data

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
            print(f'Combining data\n')
            daily_data = xr.merge([daily_avg,daily_max,daily_min], compat='override')

        ### FOR PRESSURE LEVELS ###
        elif set(ds_sub.dims) == set(pl_dims):

            # re-subset to only grab certain levels
            pl_levels = [1000, 900, 800, 700, 600, 500, 400, 300]
            ds_sub = ds_sub.sel(level=pl_levels, drop=True)

            temp = ds_sub.resample(time='1D')  # turn into daily data

            # find average and rename the variable to include AVG
            daily_avg = temp.mean(dim='time')
            var_xx = [varx for varx in daily_avg.data_vars.keys() if f'{variable.upper()}' in varx][0]
            daily_data = daily_avg.rename_vars({f'{var_xx}': f'{var_xx}_AVG'})

        else:
            print(F'Dimensional error finding daily values for {variable}')
            print(f'Dimensions are {sorted(ds_sub.dims)}.\n Skipping...\n')
            return

        # write data to NetCDF file
        print(f'Writing data to NetCDF\n')
        daily_data.to_netcdf(f'{out_fp}.nc')

    print(f'\rTime elapsed: {time.time()-start_time: .2f} s\n')

    # return ds, ds_sub, daily_data


# In[28]:


# set time array to loop through
years = np.arange(1980,2020)
months = np.arange(1,13)


# In[ ]:


# Loop through variables in var_list and process each one
# for var in var_list:
#     for year in years:
#         start_date = int(f'{year}01')
#         end_date = int(f'{year}12')
#         dsw_subset_era5(var, start_date, end_date)


# In[ ]:


# Loop through variables in pl_var_list and process each one
for var in pl_var_list:
    with open(f'{sub_script_path}pl_out.txt', 'a') as file:
        file.write(f'\n. . . . . . . . . . \n{var}\n. . . . . . . . . .\n')
    for year in years:
        with open(f'{sub_script_path}pl_out.txt', 'a') as file:
            file.write(f'{year}\n')
        start_date = int(f'{year}01')
        end_date = int(f'{year}12')
        dsw_subset_era5(var, start_date, end_date)


# In[34]:


# define a function to subset the invariant data from ERA5
def invariant_subset_era5(var='lsm'):

    # open dataset
    var_file = glob.glob(f'{era5_invariant_path}e5.oper.invariant.*_{var}*.nc')
    ds = xr.open_mfdataset(var_file)
    da = ds.isel(time=0)[f'{var.upper()}']

    # save invariant to netcdf file in local directory
    da.to_netcdf(f'{my_invariant_path}{var}_invariant.nc')

    # check if var = geopotential and if so, create elevation variable
    if var == 'z':
        elevation = da / 9.80665
        elevation.rename('ELEVATION')
        # save elevation to netcdf
        elevation.to_netcdf(f'{my_invariant_path}elevation_invariant.nc')


# In[35]:


# for var in invariant_list:
#     invariant_subset_era5(var=var)


# In[ ]:




