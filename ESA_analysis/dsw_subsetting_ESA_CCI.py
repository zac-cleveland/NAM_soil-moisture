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


# In[2]:


esa_path = '/glade/u/home/zcleveland/scratch/ESA_data/COMBINED/' # base path to full ESA dataset I downloaded
out_path = '/glade/u/home/zcleveland/scratch/ESA_data/dsw/' # base path to save the subsetted data


# In[52]:


# create function to subset ESA data into the desert southwest combined into 1 year
def dsw_subsetting_ESA(year=1980):

    start_time = time.time() # keep track of time to process.
    
    # define output filename and path
    out_fn = f'ESA_COMBINED_sm_{year}01_{year}12_dsw.nc' # out file name
    out_fp = f'{out_path}{out_fn}' # out file path (including file name)

    # check if file already exists
    if (os.path.exists(f'{out_fp}')):        
        print(f'File {out_fn} already exists. Skipping...\n')
        return

    print(f'Processing ESA COMBINED sm data for {year} . . .\n')

    # find full datasets
    files = glob.glob(f'{esa_path}{year}/ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-{year}*-fv08.1.nc')
    files.sort()
    print(f'{len(files)} number of files \n')

    # open datasets using dask for memory control
    with dask.config.set(**{'array.slicing.split_large_chunks': True}): # only for systems with dask
        ds = xr.open_mfdataset(files)
        
        # define lat/lon boundaries for subsetting
        lat_range = slice(40,20)
        lon_range = slice(-120,-100)
        ds_sub = ds.sel(lat=lat_range, lon=lon_range, drop=True)

        # rename coordinates to match ERA5 data
        ds_sub = ds_sub.rename({'lat': 'latitude', 'lon': 'longitude'})        

        # double check dimension names for sanity
        if set(ds_sub.dims) == set(['time', 'latitude', 'longitude']):
            print(f'Dimensions reset from {set(ds.dims)} to {set(ds_sub.dims)} \n')
        else:
            print(f'Something wrong with dimension renaming . . .\n')
            return

        # write files to new .nc file
        print(f'Writing {out_fn} to netCDF . . .\n')
        ds_sub.to_netcdf(f'{out_fp}')
        print(f'{out_fn} completed \n{time.time()-start_time:.2f} s\n')


# In[53]:


for year in range(1980,2020):
    dsw_subsetting_ESA(year)
    print(f'\n{year} COMPLETE \n\n')

print('All years complete.')


# In[ ]:




