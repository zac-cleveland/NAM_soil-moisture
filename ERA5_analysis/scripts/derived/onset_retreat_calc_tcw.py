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


data_in_path = '/glade/u/home/zcleveland/scratch/ERA5/dsw/'  # path to subsetted data
sub_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/'  # path to subsetting scripts
plot_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/plotting/'  # path to plotting scripts
fig_out_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/' # path to generated figures
temp_scratch_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/temp/'  # path to temp directory in scratch
derived_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/derived/'  # path to derived scripts


# In[3]:


# variable list to choose
var_list = [
    # 'lsp', # large scale precipitation (m of water) - accumu
    # 'cp', # convective precipitation (m of water) - accumu
    # 'tp', # total precipitation (m of water) - accumu -- DERIVED
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
    'tcw',  # total column water (kg m^-2) - sfc (sum of solid/liquid/vapor)
]


# In[4]:


# open datasets for calculating onset timing

# total water content stats
tcw_max = xr.open_dataset(f'{data_in_path}tcw_max_stats.nc')
tcw_min = xr.open_dataset(f'{data_in_path}tcw_min_stats.nc')

# the average of the annual max/min daily total water content values
pw_max = tcw_max['MEAN']
pw_min = tcw_min['MEAN']

# daily tcw values
files = glob.glob(f'{data_in_path}*/tcw_*_dsw.nc')
files.sort()

# open files and pull out daily average
tcw = xr.open_mfdataset(files)
tcw = tcw['TCW_AVG']


# In[ ]:


# calculate onset date using methods of Zeng and Lu 2004

# normalized precipitable water index
npwi = (tcw-pw_min)/(pw_max-pw_min)

# set a threshold of 2*pi/10
threshold = 2*np.pi/10


# create mask for when npwi values exceed threshold
def onset_condition(da):
    # create a boolean mask where npwi exceeds the threshold
    mask = da > threshold

    # use rolling window of 3 along the time dimension and check if  True
    # return mask
    return mask.rolling(time=3).sum() >= 3


# apply onset condition across time dimension
onset_mask = npwi.groupby('time.year').apply(onset_condition)

# generate empty dataset to store onset times
onset_time = xr.Dataset(
    coords={
        'year': pd.date_range(start='1980-01-01', end='2019-01-01', freq='YS'),
        'latitude': npwi.latitude.values,
        'longitude': npwi.longitude.values
    }
)
# generate empty dataset to store retreat times
retreat_time = xr.Dataset(
    coords={
        'year': pd.date_range(start='1980-01-01', end='2019-01-01', freq='YS'),
        'latitude': npwi.latitude.values,
        'longitude': npwi.longitude.values
    }
)
# create date variable
dates1 = np.empty((40, 81, 81), dtype='datetime64[ns]')
dates2 = np.empty((40, 81, 81), dtype='datetime64[ns]')
dates1[:] = np.datetime64('NaT')  # store NaT values at temporary place holders
dates2[:] = np.datetime64('NaT')  # store NaT values at temporary place holders
onset_time['date'] = (('year', 'latitude', 'longitude'), dates1)
retreat_time['date'] = (('year', 'latitude', 'longitude'), dates2)
times = npwi.time

for year in range(1980, 2020):
    with open(f'{derived_script_path}onset_retreat.txt', 'a') as file:
        file.write(f'\n\nyear: {year}\nlat: ')
    # print(f'\n\nYear: {year}')
    temp_times = times.sel(time=str(year))
    for lat in npwi.latitude:
        # print('\n', end='')
        if (lat % 1 == 0):
            with open(f'{derived_script_path}onset_retreat.txt', 'a') as file:
                file.write(f'{int(lat.values)}... ')
            # print(f'\nLat: {int(lat.values)} \nLon: ')
        for lon in npwi.longitude:
            # if (lon % 5 == 0):
            #     print(f'{int(lon.values)} ... ', end='')

            temp_mask = onset_mask.sel(time=str(year), latitude=lat, longitude=lon)
            temp_idx = np.where(temp_mask)
            temp_onset_coord = {'year': str(year), 'latitude': lat, 'longitude': lon}
            temp_retreat_coord = {'year': str(year), 'latitude': lat, 'longitude': lon}

            if temp_idx[0].size>0:
                temp_onset_time = temp_times[temp_idx[0][0] - 2]  # subtract 2 to get 1st index
                temp_retreat_time = temp_times[temp_idx[0][-1]]  # last index
            else:
                temp_onset_time = np.nan
                temp_retreat_time = np.nan

            onset_time['date'].loc[temp_onset_coord] = temp_onset_time
            retreat_time['date'].loc[temp_retreat_coord] = temp_retreat_time

# save as netcdf
onset_time.to_netcdf(f'{data_in_path}NAM_onset.nc')
retreat_time.to_netcdf(f'{data_in_path}NAM_retreat.nc')

with open(f'{derived_script_path}onset_retreat.txt', 'a') as file:
    file.write('\n\nDone')
print('Done')


# In[4]:


# define a function to calculate onset and retreat trends
def calc_onset_trend(start_year, end_year):

    # open datasets
    onset_ds = xr.open_dataset(os.path.join(data_in_path, 'NAM_onset.nc'))
    retreat_ds = xr.open_dataset(os.path.join(data_in_path, 'NAM_retreat.nc'))

    # extract time frame
    onset_time = onset_ds['date'].sel(year=slice(str(start_year), str(end_year)), drop=True)
    retreat_time = retreat_ds['date'].sel(year=slice(str(start_year), str(end_year)), drop=True)

    # close datasets
    onset_ds.close()
    retreat_ds.close()

    # convert from datetime to ordinal day
    onset_day = onset_time.dt.dayofyear
    retreat_day = retreat_time.dt.dayofyear

    # compute mean
    onset_mean = np.floor(onset_day.mean(dim='year'))
    retreat_mean = np.floor(retreat_day.mean(dim='year'))

    # compute length of NAM
    NAM_length = retreat_mean - onset_mean

    # compute the gradient
    onset_grad = onset_day.diff('year')
    retreat_grad = retreat_day.diff('year')
    NAM_length_grad = NAM_length.diff('year')

    # compute mean gradients
    mean_onset_grad = onset_grad.mean(dim='year')
    mean_retreat_grad = retreat_grad.mean(dim='year')
    mean_NAM_length_grad = NAM_length_grad.mean(dim='year')

    # save all to netcdf
    # onset
    onset_mean.to_netcdf(f'{data_in_path}NAM_onset_mean.nc')
    onset_grad.to_netcdf(f'{data_in_path}onset_gradient.nc')
    mean_onset_grad.to_netcdf(f'{data_in_path}onset_mean_gradient.nc')

    # retreat
    retreat_mean.to_netcdf(f'{data_in_path}NAM_retreat_mean.nc')
    retreat_grad.to_netcdf(f'{data_in_path}retreat_gradient.nc')
    mean_retreat_grad.to_netcdf(f'{data_in_path}retreat_mean_gradient.nc')

    # NAM length
    NAM_length.to_netcdf(f'{data_in_path}NAM_length.nc')
    NAM_length_grad.to_netcdf(f'{data_in_path}NAM_length_gradient.nc')
    mean_NAM_length_grad.to_netcdf(f'{data_in_path}mean_NAM_length_gradient.nc')


# In[6]:


if __name__ == '__main__':
    calc_onset_trend(1980,2019)

