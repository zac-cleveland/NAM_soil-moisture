#!/usr/bin/env python
# coding: utf-8

# This script subsets data for a region over the Sonora Desert and calculates the average values of given variables in that region.

# In[16]:


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


# In[17]:


my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'
sub_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/'


# In[18]:


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


# In[19]:


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
    'length': 'NAM Length'
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
    'mr': [249, 251, 33, 34],
    'son': [246, 250, 28, 32],
    'chi': [252, 256, 29, 33],
    'moj': [243, 247, 33, 37],
    'MeNmAz': [246, 256, 38, 28],
}


# In[27]:


# define main function to execute subsetting
def main(var=None, region=None, start_date=198001, end_date=201912, overwrite_flag=False):

    # get coordinates of region and create output filename and path
    if region in region_avg_list:
        coords = region_avg_coords[region]
        out_fn = f'{var}_{start_date}_{end_date}_{region}.nc'
        out_fp = os.path.join(my_era5_path, region, out_fn)
    elif verify_coords(region):
        coords = region
        region = '-'.join(map(str, region))
        out_fn = f'{var}_{start_date}_{end_date}_{region}.nc'
        out_fp = os.path.join(my_era5_path, 'region_avg', out_fn)
    else:
        print('Something is wrong with the region specified')
        return

    # make sure output file doesn't already exist
    if os.path.exists(out_fp):
        if not overwrite_flag:
            print('File already exists.  Set overwrite_flag to True to overwrite it')
            return
        else:
            print('File already exists.  overwrite_flag is set to True.  File will be overwritten')

    # get input files for extracting
    files = get_input_files(var)
    if not files:
        print('files is empty')
        return

    # open dataset
    ds = xr.open_mfdataset(files)

    # compute average, sum, etc. for region
    region_ds = get_region_data(ds, var, region, coords)

    # save dataset
    region_ds.to_netcdf(out_fp)


# In[21]:


# define a finction to get coordinates of region to subset
def verify_coords(region=None):

    # check that region is not None
    if region is None:
        print('region cannot be None')
        return False

    # if list of coords is input, verify it is valid, and return the same list
    # verify region is a list
    if isinstance(region, list):
        if len(region) != 4:
            print('you must specify either a predefined region from the list, or input a list of 4 coordinates')
            print('[EAST, WEST, NORTH, SOUTH]')
            return False
        # verify all values are integers
        for value in region:
            if not isinstance(value, int):
                print('all coordinate values in list must be integers')
                return False
        # verify WEST and EAST are within 0-360 and that WEST < EAST
        if not ((0 <= region[0] <= 360) & (0 <= region[1] <= 360) & (region[0] <= region[1])):
            print('Something wrong with WEST and/or EAST coordinate')
            return False
        # verify NORTH and SOUTH are within -90 and 90 and that NORTH > SOUTH
        if not ((-90 <= region[2] <= 90) & (-90 <= region[3] <= 90) & (region[2] >= region[3])):
            print('Something wrong with NORTH and/or SOUTH coordinate')
            return False
        # if all conditions met, return region
        return True


# In[33]:


# define a function to get input files for extracting
# returns list of files
def get_input_files(var=None):

    if var in pl_var_list:
        files = glob.glob(os.path.join(my_era5_path, f'dsw/*/pl/{var}_*_dsw.nc'))
    else:
        files = glob.glob(os.path.join(my_era5_path, f'dsw/*/{var}_*_dsw.nc'))
    return files


# In[29]:


# define a function to get regional data and return the average, sum, etc.
def get_region_data(ds, var, region, coords):

    # get actual variable name from dataset
    var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v.upper()][0]

    # slice ds into region coords
    west, east, north, south = coords[0], coords[1], coords[2], coords[3]
    da = ds[var_name].sel(latitude=slice(west, east), longitude=slice(north, south))

    if var in sfc_accumu_list:
        region_da = da.resample(time='1M').sum(dim=['time', 'latitude', 'longitude'])
    elif var in sfc_instan_list:
        region_da = da.resample(time='1M').mean(dim=['time', 'latitude', 'longitude'])
    elif var in pl_var_list:
        region_da = da.resample(time='1M').sum(dim=['time', 'latitude', 'longitude'])
    else:
        print('Something wrong with input var name')
        return

    # rename da variable to include region
    region_ds = region_da.to_dataset().rename(
        {f'{var_name}': f'{var_name.upper()}_{region.upper()}'}
    )

    # return the dataset
    return region_ds


# In[34]:


# run the code!
if __name__ == '__main__':

    start_time = time.time()

    start_date = 198001
    end_date = 201912

    for region in region_avg_list:
        for var in var_list:
            print(f'Processing {var_dict[var]}')
            print(f'{var} -- {region} -- {start_date} -- {end_date}')
            with open(os.path.join(sub_script_path, 'region_avg_subsetting.txt'), 'a') as file:
                file.write(f'Processing {var_dict[var]}\n{var} -- {region} -- {start_date} -- {end_date}')
            main(var=var, region=region, start_date=start_date, end_date=end_date)
            elapsed_time = time.time()-start_time
            print(f'Elapsed time: {elapsed_time}')
            with open(os.path.join(sub_script_path, 'region_avg_subsetting.txt'), 'a') as file:
                file.write(f'Elapsed time: {elapsed_time}\n')


# In[ ]:




