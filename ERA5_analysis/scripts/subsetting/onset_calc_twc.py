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
import cartopy.feature as cfeature
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

import calendar


# In[2]:


data_in_path = '/glade/u/home/zcleveland/scratch/ERA5/dsw/' # path to subsetted data
sub_script_path = '/glade/u/home/zcleveland/ERA5_analysis/scripts/subsetting/' # path to subsetting scripts
plot_script_path = '/glade/u/home/zcleveland/ERA5_analysis/scripts/plotting/' # path to plotting scripts
fig_out_path = '/glade/u/home/zcleveland/ERA5_analysis/figures/' # path to generated figures
temp_scratch_path = '/glade/u/home/zcleveland/ERA5_analysis/temp/' # path to temp directory in scratch


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
    'tcw', # total column water (kg m^-2) - sfc (sum total of solid, liquid, and vapor in a column)
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
files = glob.glob(f'{data_in_path}*/tcw*.nc')
files.sort()

# open files and pull out daily average
tcw = xr.open_mfdataset(files)
tcw = tcw['TCW_AVG']


# In[5]:


# calculate onset date using methods of Zeng and Lu 2004

# normalized precipitable water index
npwi = (tcw-pw_min)/(pw_max-pw_min)

# set a threshold of 2*pi/10
threshold = 2*np.pi/10

# create mask for when npwi values exceed threshold
def onset_condition(da):
    # create a boolean mask where npwi exceeds the threshold
    mask = da > threshold
    
    # use rolling window with length 3 along the time dimension and check if all values are True
    return mask.rolling(time=3).sum() >= 3

# apply onset condition across time dimension
onset_mask = npwi.groupby('time.year').apply(onset_condition)

# generate empty dataset to store 
onset_time = xr.Dataset(
    coords={
        'year': pd.date_range(start='1980-01-01', end='2019-01-01', freq='YS'),
        'latitude': npwi.latitude.values,
        'longitude': npwi.longitude.values
    }
)
# create date variable
dates = np.empty((40,81,81), dtype='datetime64[ns]')
dates[:] = np.datetime64('NaT')  # store NaT values at temporary place holders
onset_time['date'] = (('year', 'latitude', 'longitude'), dates)
times = npwi.time

for year in range(1980,2020):
    print(f'\n\nYear: {year}')
    temp_time = times.sel(time=str(year))
    for lat in npwi.latitude:
        print('\n', end='')
        if (lat%1==0):
            print(f'\nLat: {int(lat.values)} \nLon: ')
        for lon in npwi.longitude:
            if (lon%5==0):
                print(f'{int(lon.values)} ... ', end='')
                
            temp_mask = onset_mask.sel(time=str(year), latitude=lat, longitude=lon)
            temp_idx = np.where(temp_mask)
            temp_coord = {'year': str(year), 'latitude': lat, 'longitude': lon}

            if temp_idx[0].size>0:
                time = temp_time[temp_idx[0][0]]
            else:
                time = np.nan
                
            onset_time['date'].loc[temp_coord] = time

# save as netcdf
onset_time.to_netcdf(f'{data_in_path}NAM_onset.nc')

print('Done')

