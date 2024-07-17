#!/usr/bin/env python
# coding: utf-8

# In[10]:


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
import matplotlib.patches as mpatches
from shapely.geometry.polygon import LinearRing
import statsmodels.stats.multitest as multitest

# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
import matplotlib.image as mpimg
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm

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

# paths to various directories
rda_era5_path = '/glade/campaign/collections/rda/data/ds633.0/'  # base path to ERA5 data on derecho
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
my_esa_path = '/glade/u/home/zcleveland/scratch/ESA_data/'  # path to both the original ESA data and the data that I subset
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


# In[46]:


# create function to subset ESA data into the desert southwest combined into 1 year
def main(region, year):
    # define output filename and path
    out_fn = f'ESA_sm_{year}01_{year}12_{region}.nc'
    out_fp = os.path.join(my_esa_path, f'{region}', out_fn)

    # check if file already exists
    if (os.path.exists(f'{out_fp}')):
        print(f'File {out_fn} already exists. Skipping...\n')
        return

    # find full datasets
    files = glob.glob(f'{my_esa_path}original/{year}/ESACCI-SOILMOISTURE-L3S-SSMV-COMBINED-{year}*-fv08.1.nc')
    files.sort()
    if not files:
        raise FileNotFoundError(f'Files not found for ESA_sm for year: {year}')

    # open datasets using dask for memory control
    with dask.config.set(**{'array.slicing.split_large_chunks': True}): # only for systems with dask
        ds = xr.open_mfdataset(files)
        # rename lat and lon to latitude and longitude
        ds = ds.rename({'lat': 'latitude', 'lon': 'longitude', 'sm': 'ESA_sm'})
        # change the longitude coordinates from a -180 to 180 grid to a 0-360 grid to match other data
        ds = ds.assign_coords(longitude=((ds.longitude + 360) % 360))
        ds = ds.sortby(ds.longitude)
        # subset if region is not global
        if region != 'global':
            ds = ds.sel(latitude=slice(40, 20), longitude=slice(240, 260), drop=True)

        # write files to new .nc file
        ds.to_netcdf(f'{out_fp}')


# In[ ]:


# run the code
if __name__ == '__main__':
    years = np.arange(1980,2020)
    regions = ['dsw', 'global']

    for region in regions:
        for year in years:
            print(f'region: {region}\t-\tyear: {year}')
            main(region, year)

