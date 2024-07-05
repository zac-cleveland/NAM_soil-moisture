#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from matplotlib.colors import TwoSlopeNorm

from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.axes_divider import HBoxDivider
import mpl_toolkits.axes_grid1.axes_size as Size
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.image as mpimg

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

# random
from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)


# In[ ]:


my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
scratch_path = '/glade/u/home/zcleveland/scratch/'
plot_out_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/'
temp_scratch_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/temp/'


# In[ ]:


# Variable lists
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
    'vipile',  # vertical integral of potential, internal, and latent energy (J m^-2)
    'viwve',  # vertical integral of eastward water vapour flux (kg m^-1 s^-1) - positive south -> north
    'viwvn',  # vertical integral of northward water vapour flux (kg m^-1 s^-1) - positive west -> east
    'viwvd',  # vertical integral of divergence of moisture flux (kg m^-2 s^-1) - positive divergencve
    'z_thick_1000-500',  # geopotential height thickness (m) - difference between two height levels
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
    'z_height',  # geopotential height (m)
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

# NAM variables
NAM_var_list = [
    'onset',
    'retreat',
    'length',
    'precipitation',
    'precipitation-rate'
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
    'baja',
]

# variables that are fluxes and need to be multiplied by -1 for easier understanding
flux_var_list = [
    'sshf',  # surface sensible heat flux (J m^-2)
    'slhf',  # surface latent heat flux (J m^-2)
    'ttr',  # top net thermal radiation (OLR, J m^-2) -- divide by time (s) for W m^-2
    'ishf',  # instant surface heat flux (W m^-2)
    'ie',  # instant moisture flux (kg m^-2 s^-1)
    'str',  # surface thermal radiation (J m^-2)
]

# misc variables
misc_var_list = [
    'nino-3',
]


# Variable dictionaries

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
    'vipile': 'vertical integral of potential, internal, and latent energy',
    'viwve': 'vertical integral of eastward water vapour flux',
    'viwvn': 'vertical integral of northward water vapour flux',
    'viwvd': 'vertical integral of divergence of moisture flux',
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
    'z_height': 'Geopotential Height',
    'z_thick_1000-500': 'Geopotential Height Thickness from 1000 to 500 mb',
    't': 'Temperature',
    'u': 'U Component of Wind',
    'v': 'V Component of Wind',
    'q': 'Specific Humidity',
    'w': 'Vertical Velocity',
    'r': 'Relative Humidity',
    'onset': 'NAM Onset',
    'retreat': 'NAM Retreat',
    'length': 'NAM Length',
    'precipitation': 'Yearly NAM Season Precipitation',
    'precipitation-rate': 'NAM Precipitation Rate',
    'nino-3': r'Ni$\tilda{n}$o-3 Index',
}

# variable units in latex format for plotting
var_units = {
    'sd': r'(m)',
    'msl': r'(Pa)',
    'tcc': r'(0-1)',
    'stl1': r'(K)',
    'stl2': r'(K)',
    'stl3': r'(K)',
    'stl4': r'(K)',
    'swvl1': r'$(m^3 m^{-3})$',
    'swvl2': r'$(m^3 m^{-3})$',
    'swvl3': r'$(m^3 m^{-3})$',
    'swvl4': r'$(m^3 m^{-3})$',
    '2t': r'(K)',
    '2d': r'(K)',
    'ishf': r'$(W m^{-2})$',
    'ie': r'$(kg m^{-2} s^{-1})$',
    'cape': r'$(J kg^{-1})$',
    'tcw': r'$(kg m^{-2})$',
    'sstk': r'(K)',
    'vipile': r'$(J m^{-2})$',
    'viwve': r'$(kg m^{-1} s^{-1})$',
    'viwvn': r'$(kg m^{-1} s^{-1})$',
    'viwvd': r'$(kg m^{-2} s^{-1})$',
    'lsp': r'(m)',
    'cp': r'(m)',
    'tp': r'(m)',
    'sshf': r'$(J m^{-2})$',
    'slhf': r'$(J m^{-2})$',
    'ssr': r'$(J m^{-2})$',
    'str': r'$(J m^{-2})$',
    'sro': r'(m)',
    'sf': r'(m)',
    'ssrd': r'$(J m^{-2})$',
    'strd': r'$(J m^{-2})$',
    'ttr': r'$(J m^{-2})$',
    'z': r'$(m^2 s^{-2})$',
    'z_height': '$(m)$',
    'z_thick_1000-500': '$(m)$',
    't': r'(K)',
    'u': r'$(m s^{-1})$',
    'v': r'$(m s^{-1})$',
    'q': r'$(kg kg^{-1})$',
    'w': r'$(Pa s^{-1})$',
    'r': r'(%)',
    'onset': '',
    'retreat': '',
    'length': r'# of days',
    'precipitation': r'(m)',
    'precipitation-rate': r'(m day^{-1}, NAM Season Precip / NAM Length)',
    'nino-3': r'(Ni$\tilda{n}$o-3 Index Anomaly)',
}

# dictionary of regions and their names
region_avg_dict = {
    'cp': 'Colorado Plateau',
    'mr': 'Mogollon Rim',
    'son': 'Sonoran Desert',
    'chi': 'Chihuahuan Desert',
    'moj': 'Mojave Desert',
    'MeNmAz': 'MEX, NM, AZ Border',
    'baja': r'Coast of Baja, CA (5$\degree$ x 5$\degree$)',
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
    'baja': [242, 247, 27, 22],
}

# dictionary of colors for the plot of each region
region_colors_dict = {
    'cp': 'blue',
    'mr': 'darkorange',
    'son': 'green',
    'chi': 'red',
    'moj': 'purple',
    'MeNmAz': 'brown',
    'baja': 'yellow',
    'dsw': 'black'
}


# In[ ]:


# define a funciton to plot a time series of ERA5 vs ESA soil moisture over the Colorado Plateau
def plot_timeseries_sm_cp(overwrite_flag=False):
    # create out filename and path -- check existence
    out_fn = 'ERA5_ESA_sm_cp_ts.png'
    out_fp = os.path.join(plot_out_path, 'time_series', out_fn)

    if overwrite_flag:
        pass
    elif os.path.exists(out_fp):
        print(f'{out_fn} already exists.  Set overwrite_flag=True to overwrite.')
        print('Skipping . . .\n')
        return
    else:
        print(f'Starting plot for {out_fn}')

    # get file path for datasets
    era_file = f'{scratch_path}ERA5/cp/swvl1_198001_201912_cp.nc'
    regrid_file = f'{scratch_path}ERA5/cp/swvl1_regrid_198001_2019_cp.nc'
    esa_file = f'{scratch_path}ESA_data/cp/sm_esa_198001_201912_cp.nc'

    # open datasets
    era_ds = xr.open_dataset(era_file)
    regrid_ds = xr.open_dataset(regrid_file)
    esa_ds = xr.open_dataset(esa_file)

    # set up figure
    plt.figure(figsize=(12,6))
    era_ds['SWVL1_AVG_CP'].plot(label='Original ERA5 Data', color='r')
    # regrid_ds['SWVL1_AVG_CP'].plot(label='Regrid ERA5 data', color='b')
    esa_ds['sm_CP'].plot(label='ESA Data', color='g')
    plt.title('ERA5 vs. ESA\nSoil Moisture')
    plt.legend()
    plt.xlabel('Time (months)')
    plt.ylabel('Soil Moisture Content (m$^3$ m$^{-3}$)')
    plt.tight_layout()
    plt.savefig(out_fp, dpi=600, bbox_inches='tight')
    plt.show()
    plt.close()


# In[ ]:


# plot_timeseries_sm_cp(overwrite_flag=False)


# In[ ]:


# define a function to plot a time series of given variables
def main_time_series(var_list, region, **kwargs):

    # get kwargs
    start_date = kwargs.get('start_date', 198001)
    end_date = kwargs.get('end_date', 201912)
    var_month_list = kwargs.get('var_month_list', [i for i in range(1,13)])
    level = kwargs.get('level', 500)
    color_cycle = kwargs.get('color_cycle', plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # split start and end date to get year and month
    start_year, start_month = f'{start_date}'[:4], f'{start_date}'[4:]
    end_year, end_month = f'{end_date}'[:4], f'{end_date}'[4:]

    var_month_list = ensure_var_list(var_month_list)
    var_list = ensure_var_list(var_list)

    # months list
    var_months = month_num_to_name(var=var_list[0], months=var_month_list, **kwargs)

    var_data_list = []

    for var in var_list:
        # get var files
        var_files = get_var_files(var, region, **kwargs)
        var_da = open_var_data(var_files, var, type='da', **kwargs)

        # open dataset
        var_data = subset_var_data(var_files, var, var_month_list, region, da=var_da, mean_flag=True, **kwargs)
        var_data_list.append(var_data)

    # plot time series
    fig, ax1 = plt.subplots(figsize=(12,6), facecolor='white')

    color = color_cycle[0]
    ax1.plot(var_data_list[0], label=var_dict[var_list[0]], color=color)
    # customize plot
    ax1.set_xlabel('Year')
    years = np.arange(1980,2020)
    ax1.set_xticks((years-1980)[::4])
    ax1.set_xticklabels([str(i) for i in years][::4])
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.set_ylabel(var_dict[var_list[0]], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    if len(var_data_list) != 1:
        axes = [ax1]
        for i in range(1, len(var_list)):
            ax = ax1.twinx()
            ax.spines['right'].set_position(('outward', 60*(i-1)))
            color=color_cycle[i]
            ax.plot(var_data_list[i], label=var_dict[var_list[i]], color=color)
            ax.set_ylabel(var_dict[var_list[i]], color=color)
            ax.tick_params(axis='y', labelcolor=color)

            # ax.set_facecolor('white')

            axes.append(ax)
            ax.spines['right'].set_visible(True)

        for ax in axes:
            ax.legend(loc='lower right')

    fig.tight_layout()

    # show plot
    plt.show()


# In[ ]:


# define a function to plot a time series of a given variable
def plot_time_series(var_data, **kwargs):
    i = kwargs.get('i', '')
    ax = ax1.twinx()
    # ax.spines['right'].set_position(('outward', 60))
    ax.plot(var_data, label=var_dict[var_list[i]])
    ax.set_ylabel(var_dict[var_list[i]])

    axes.append(ax)


# In[ ]:


# define a function to turn a list of integers into months
def month_num_to_name(var, months, **kwargs):

    # make string for month letters from var_range (e.g. [6,7,8] -> 'JJA')
    if var in NAM_var_list:
        var_months = ''  # don't use months for onset, retreat, length
    elif len(months) == 1:
        var_months = calendar.month_name[months[0]]  # use full month name if only 1 month
    elif len(months) == 12:
        var_months = 'YEAR'
    elif ((len(months) > 1) & (len(months) < 12)):
        var_months = ''.join([calendar.month_name[m][0] for m in months])  # make string of months, i.e. 3, 4, 5 is MAM
    return var_months


# In[ ]:


# define a function to check if inputs are list or not
def ensure_var_list(x):

    if not isinstance(x, list):
        return [x]
    return x


# In[ ]:


# define a function to get var files, open dataset, and subset if needed
def get_var_data(var, region='dsw', months=[i for i in range(1,13)], **kwargs):
    r"""
    Retrieves the data for a given variable from my subet ERA5 dataset.  User can choose to return a dataset or data array
    and whether to subset that data based on a region or time.  Any subset data is returned as a data array.

    Parameters
    ----------
    var : str
            The variable desired
    region : str
            The region desired
    months : list, int
            A list of months desired [1, 2, ..., 12]

    Returns
    -------
    var_data : xarray Data Array
            A data array containing the desired data, either in full or subset based on user input

    Kwargs
    ------
    subset_flag : bool
            True or False.  Whether to subset the data or not
    level : int
            The pressure level desired.  Only applied for pressure level data
    var_type : str
            Specify whether to return a dataset or data array
    mean_flag : bool
            True or False.  Whether to compute the mean (or sum) over the specified months
    group_type : str
            How to group data prior to computing mean or sum across time.
            Options include 'year', 'month', 'dayofyear', etc.

    See Also
    --------
    get_var_files : returns all files for specified variable
    open_var_data : opens the variable dataset or data array
    subset_var_data : subsets data array based on user input

    Notes
    -----

    """

    files = get_var_files(var, region, **kwargs)
    var_data = open_var_data(files, var, **kwargs)
    if kwargs.get('subset_flag', False):
        return subset_var_data(var_data, var, months, region, **kwargs)
    return var_data


# In[ ]:


# define a function to get the files for a given variable/region
def get_var_files(var, region, **kwargs):

    # grab files for sfc var
    if ((var in sfc_instan_list) or (var in sfc_accumu_list)):
        # dsw
        if region != 'global':
            files = glob.glob(f'{my_era5_path}dsw/*/{var.lower()}_*_dsw.nc')

        elif region == 'global':
            files = glob.glob(f'{my_era5_path}global/*/{var.lower()}_*_dsw.nc')

    # grab files for pl var
    elif var in pl_var_list:
        files = glob.glob(f'{my_era5_path}dsw/*/pl/{var.lower()}_*_dsw.nc')

    # grab files for NAM var
    elif var in NAM_var_list:
        files = glob.glob(f'{my_era5_path}dsw/NAM_{var}.nc')

    elif var in misc_var_list:
        files = glob.glob(f'{misc_data_path}{var}/{var}*.nc')

    elif var in invar_var_list:
        files = glob.glob(f'{my_era5_path}invariants/{var}_invariant.nc')

    # if something went wrong
    else:
        print('something went wrong finding files')
        files = []

    files.sort()
    return files


# In[ ]:


# define a function to open variable datasets
def open_var_data(files, var, **kwargs):
    # get kwargs
    var_type = kwargs.get('var_type', 'da')  # default to returning a data array

    # open dataset
    ds = xr.open_mfdataset(files)
    if type == 'ds':
        return ds

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v.upper()][0]
    return ds[var_name]


# In[ ]:


# define a function to open subset an input data set (or array) by:
# latitude/longitude
# time
# averages
def subset_var_data(var_data, var, months, region, **kwargs):

    if isinstance(var_data, xr.Dataset):
        # pull out actual variable name in the dataset since they can be different names/capitalized
        var_name = [v for v in var_data.data_vars.keys() if f'{var.upper()}' in v.upper()][0]
        da = var_data[var_name]
    elif isinstance(var_data, xr.DataArray):
        da = var_data
    else:
        print('something wrong with var_data in subset_var_data')
        return None

    # subset to regional average if region is specified
    if region in region_avg_list:
        lats = slice(region_avg_coords[region][2], region_avg_coords[region][3])
        lons = slice(region_avg_coords[region][0], region_avg_coords[region][1])
        da = da.sel(latitude=lats, longitude=lons).mean(dim=['latitude', 'longitude'], skipna=True)

    # subset to level if var is a pl var
    if var.lower() in pl_var_list:
        level = kwargs.get('level', None)
        if level is not None:
            da = da.sel(level=level)

    # just return da if var is NAM var, convert to dayofyear for onset and retreat dates
    if var.lower() in NAM_var_list:
        if ((var.lower() == 'onset') or (var.lower() == 'retreat')):
            return da.dt.dayofyear
        else:
            return da

    # subset the data specified by months
    da_sub = da.sel(time=da['time.month'].isin(months))

    # subset further and compute mean/sum if specified by mean_flag
    mean_flag = kwargs.get('mean_flag', False)
    if mean_flag:
        groupby_type = f"time.{kwargs.get('group_type', 'year')}"

        if var.lower() in sfc_accumu_list:
            return da_sub.groupby(groupby_type).sum(dim='time')
        else:
            return da_sub.groupby(groupby_type).mean(dim='time')

    # if mean_flag is False, jsut return whole data array
    else:
        return da_sub


# In[ ]:


# define the function to calculate the regression between two variables
def main_regression(var1, var1_region, var2, var2_region, detrend_flag=True, show_flag=True, save_flag=False, overwrite_flag=False, **kwargs):

    var1_month_list = kwargs.get('var1_month_list', [i for i in range(1,13)])
    var1_level = kwargs.get('var1_level', None)
    var2_month_list = kwargs.get('var2_month_list', [i for i in range(1,13)])
    var2_level = kwargs.get('var2_level', None)

    var1_months = month_num_to_name(var1, var1_month_list)
    var2_months = month_num_to_name(var2, var2_month_list)

    if var1 in pl_var_list:
        if var1_level is None:
            print('var1_level cannot be None if var1 is a pl_var')
            return
        var1_level_str = f"{var1_level}_"
    else:
        var1_level_str = ''
    if var2 in pl_var_list:
        if var2_level is None:
            print('var2_level cannot be None if var2 is a pl_var')
            return
        var2_level_str = f"{var2_level}_"
    else:
        var2_level_str = ''

    out_fp = f'{plot_out_path}regressions/regress_{var1}_{var1_level_str}{var1_months}_{var1_region}_{var2}_{var2_level_str}{var2_months}_{var2_region}.png'
    if os.path.exists(out_fp):
        print('File already exists.')
        if save_flag and not overwrite_flag:
            print('overwrite_flag is False. Skipping . . .')
            return
        elif not overwrite_flag:
            if show_flag and not save_flag:
                img = mpimg.imread(out_fp)
                plt.imshow(img)
                plt.axis('off')
                plt.set_title(f'{var_dict[var2]} ({var2_months}{var2_months}_{var2_region.upper()})\nRegressed on\n{var_dict[var1]} ({var1_months}{var1_months}_{var1_region.upper()})')
                plt.tight_layout()
                plt.show()
                plt.close()
        elif save_flag and overwrite_flag:
                print('overwrite_flag is True.  Overwriting . . .')

    # # get var files
    # var1_files = get_var_files(var1, var1_region, **kwargs)
    # var2_files = get_var_files(var2, var2_region, **kwargs)

    # if ((not var1_files) or (not var2_files)):
    #     print(f'missing files var1: {len(var1_files)} - var2: {len(var2_files)} . . .')
    #     return

    # # open var datasets and get data array
    # var1_da = open_var_data(var1_files, var1, type='da', **kwargs)
    # var2_da = open_var_data(var2_files, var2, type='da', **kwargs)

    # # subset the variable data
    # mean_flag=kwargs.get('mean_flag', False)
    # var1_data = subset_var_data(var1_files, var1, var1_month_list, var1_region, da=var1_da,
    #                             mean_flag=mean_flag, level=kwargs.get('var1_level', 500))
    # var2_data = subset_var_data(var2_files, var2, var2_month_list, var2_region, da=var2_da,
    #                             mean_flag=mean_flag, level=kwargs.get('var2_level', 500))

    # get var data
    var1_data = get_var_data(var1, region=var1_region, months=var1_month_list,
                             level=var1_level, subset_flag=True, **kwargs)
    var2_data = get_var_data(var2, region=var2_region, months=var2_month_list,
                             level=var2_level, subset_flag=True, **kwargs)

    # apply detrend if needed
    if detrend_flag:
        var1_data = apply_detrend(var1_data, **kwargs)
        var2_data = apply_detrend(var2_data, **kwargs)

    # check if var is a flux and need to be flipped
    # only flip if one OR the other is a flux, but not both
    if var1 in flux_var_list:
        var1_data = var1_data * -1
        print(f'var1 in flux_var_list. flipping . . .')
    if var2 in flux_var_list:
        var2_data = var2_data * -1
        print(f'var2 in flux_var_list. flipping . . .')

    # regress var2 onto var1
    regression_ds = apply_regression(var1_data, var2_data, **kwargs)

    # return data if not plotted or saved
    if not show_flag and not save_flag:
        print('show_flag and save_flag are False. Returning var1_data, var2_data, regression_ds . . .')
        return var1_data, var2_data, regression_ds

    # plot regression data
    var1_min = var1_data.min(dim=kwargs.get('group_type', 'time'))
    var1_max = var1_data.max(dim=kwargs.get('group_type', 'time'))
    var2_min = regression_ds['slope'] * var1_min + regression_ds['intercept']
    var2_max = regression_ds['slope'] * var1_max + regression_ds['intercept']
    var2_spread = var2_max - var2_min

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection=projection))

    # call plotting function
    regress_cf, regress_cs = plot_regression_data(var2_spread, var1_data, var2_data, regression_ds, **kwargs)

    # add coastlines, state borders, and other features
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    plt.colorbar(regress_cf, ax=ax, label=f'{var_units[var2]}', pad=0.02)
    plt.clabel(regress_cs, inline=True, fontsize=8, fmt='%1.1f')
    fig.suptitle(f'{var_dict[var2]} ({var2_level_str}{var2_months}_{var2_region.upper()})\nRegressed on\n{var_dict[var1]} ({var1_level_str}{var1_months}_{var1_region.upper()})')
    plt.tight_layout()

    if save_flag:
        plt.savefig(out_fp, bbox_inches='tight', dpi=300)
        plt.close()
    if show_flag:
        plt.show()
        plt.close()


# In[ ]:


# define a function to plot the regression data
def plot_regression_data(var2_spread, var1_data, var2_data, regression_ds, **kwargs):

    # create contour levels and hatches for plotting
    vmin = np.nanmin(var2_spread)
    vmax = np.nanmax(var2_spread)
    cf_levels = np.linspace(vmin, vmax, 50)

    if vmin < 0 and vmax > 0:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap = 'RdBu_r'
    else:
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        cmap = 'Blues' if vmax <= 0 else 'Reds'

    # plot the data using contourf
    regress_cf = plt.contourf(var2_spread.longitude, var2_spread.latitude,
                              var2_spread, levels=cf_levels,
                              cmap=cmap, norm=norm, extend='both')

    regress_cs = plt.contour(regression_ds.longitude, regression_ds.latitude,
                             regression_ds['slope'], levels=10, linewidths=0.5, linestyles='--', colors='black')

    # # extract coordinates where p-value < 0.1 (dots) and p-value < 0.05 (triangles)
    # lat, lon = np.meshgrid(regression_ds['pvalue'].latitude, regression_ds['pvalue'].longitude, indexing='ij')
    # mask_dots = (regression_ds['pvalue'] <= 0.1) & (regression_ds['pvalue'] >= 0.05)
    # mask_triangles = regression_ds['pvalue'] <= 0.05

    # # Plot dots (p-value < 0.1 and >= 0.05)
    # p1 = plt.scatter(lon[mask_dots], lat[mask_dots], color='black', marker='.',
    #             s=5, transform=ccrs.PlateCarree(), label='0.05 <= p < 0.1')

    # # Plot triangles (p-value < 0.05)
    # p05 = plt.scatter(lon[mask_triangles], lat[mask_triangles], color='black', marker='^',
    #             s=8, transform=ccrs.PlateCarree(), label='p < 0.05')

    # return regress_cf, p1, p05
    return regress_cf, regress_cs


# In[ ]:


# define a function to regress data
def regress_data(arr1, arr2):

    # mask out nan values
    mask = np.isfinite(arr1) & np.isfinite(arr2)
    arr1_mask = arr1[mask]
    arr2_mask = arr2[mask]

    if len(arr1_mask) < 2:  # check if there are enough data points
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    if len(np.unique(arr1_mask)) < 2:  # check that not all x values are identical
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    else:
        # compute linear regression
        res = sp.stats.linregress(arr1_mask, arr2_mask)
        return res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr, res.intercept_stderr


# define a function to mask data for detrending or correlating
def apply_regression(da1, da2, **kwargs):

    input_dims = kwargs.get('group_type', 'time')
    # load data
    da1.load()
    da2.load()

    result = xr.apply_ufunc(
        regress_data, da1, da2,
        input_core_dims=[[input_dims], [input_dims]],
        output_core_dims=[[], [], [], [], [], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float, float, float, float]
    )
    regression_ds = xr.Dataset({
        'slope': result[0],
        'intercept': result[1],
        'rvalue': result[2],
        'pvalue': result[3],
        'stderr': result[4],
        'intercept_stderr': result[5]
    })
    # regress_da = xr.DataArray(result)
    return regression_ds


# In[ ]:


# define a function to detrend the data

# MANUALLY DETREND WITH LINEAR REGRESSION
def detrend_data(arr):

    # set up x array for the years
    arr_time = np.arange(0,len(arr))

    # mask out nan values
    mask = np.isfinite(arr)
    arr_time_mask = arr_time[mask]
    arr_mask = arr[mask]

    # make sure the array is not full of non-finite values
    if len(arr_mask) == 0:
        arr_detrend = np.empty(len(arr))
        arr_detrend[:] = np.nan

    else:
        # compute linear regression
        result = sp.stats.linregress(arr_time_mask, arr_mask)
        m, b = result.slope, result.intercept

        # detrend the data
        arr_detrend = arr - (m*arr_time + b)

    return arr_detrend


# define a function to mask data for detrending or correlating
def apply_detrend(da, **kwargs):

    input_dims = kwargs.get('group_type', 'time')
    # load data
    da.load()

    da_detrend = xr.apply_ufunc(
        detrend_data, da,
        input_core_dims=[[input_dims]],
        output_core_dims=[[input_dims]],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype]
    )

    return da_detrend


# In[ ]:


# # test cell to copmute and plot regression data
# var1 = 'sf'
# var1_region = 'cp'
# var1_month_list = [3, 4, 5]
# var2 = 'z_thick_1000-500'
# var2_region = 'dsw'
# var2_month_list = [6, 7, 8]
# var2_level = 500
# detrend_flag = True
# mean_flag = True
# group_type = 'year'
# show_flag = True
# save_flag = False
# overwrite_flag = False

# main_regression(var1, var1_region, var2, var2_region,
#                 var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
#                 detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
#                 show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)

# # # var1_data, var2_data, regression_ds = main_regression(var1, var1_region, var2, var2_region, var1_month_list=var1_month_list,
# # #                                                       var2_month_list=var2_month_list, var2_level=var2_level,
# # #                                                       detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag,
# # #                                                       mean_flag=mean_flag, group_type=group_type)


# In[ ]:


# cell to copmute and plot regression data
var1_list = ['2t', 'stl1', 'sstk', 'sshf', 'slhf', 'sd', 'swvl1']
# var1_region = 'dsw'
var1_region = 'cp'
var1_month_list = [i for i in range(1,13)]
var2_list = ['z_height', 'z_thick_1000-500', 'msl', 'tp']
# var2_region = 'cp'
var2_region = 'dsw'
var2_month_list = [i for i in range(1,13)]
var2_level = 500
detrend_flag = True
mean_flag = False
group_type = 'time'
show_flag = False
save_flag = True
overwrite_flag = False

for var1 in var1_list:
    for var2 in var2_list:
        print(f'var1: {var1}\t-\tvar2: {var2}')
        if var1 == 'sstk':
            main_regression(var1, 'baja', var2, var2_region,
                            var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                            detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                            show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
        else:
            main_regression(var1, var1_region, var2, var2_region,
                            var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                            detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                            show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# cell to copmute and plot regression data
var1_list = ['2t', 'stl1', 'sstk', 'sshf', 'slhf', 'sd', 'swvl1']
# var1_region = 'dsw'
var1_region = 'cp'
var1_month_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
var2_list = ['z_height', 'z_thick_1000-500', 'msl', 'tp', 'onset', 'retreat', 'length', 'precipitation', 'precipitation-rate']
# var2_region = 'cp'
var2_region = 'dsw'
var2_month_list = [6, 7, 8]
var2_level = 500
detrend_flag = True
mean_flag = True
group_type = 'year'
show_flag = False
save_flag = True
overwrite_flag = False

for var1 in var1_list:
    for var1_month_list in var1_month_lists:
        for var2 in var2_list:
            print(f'var1: {var1} - {var1_month_list}\t-\tvar2: {var2}')
            if var1 == 'sstk':
                main_regression(var1, 'baja', var2, var2_region,
                                var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                                detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                                show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
            else:
                main_regression(var1, var1_region, var2, var2_region,
                                var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                                detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                                show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# cell to copmute and plot regression data
var1_list = ['2t', 'stl1', 'sstk', 'sshf', 'slhf', 'sd', 'swvl1']
var1_region = 'dsw'
# var1_region = 'cp'
var1_month_list = [i for i in range(1,13)]
var2_list = ['z_height', 'z_thick_1000-500', 'msl', 'tp']
var2_region = 'cp'
# var2_region = 'dsw'
var2_month_list = [i for i in range(1,13)]
var2_level = 500
detrend_flag = True
mean_flag = False
group_type = 'time'
show_flag = False
save_flag = True
overwrite_flag = False

for var1 in var1_list:
    for var2 in var2_list:
        print(f'var1: {var1}\t-\tvar2: {var2}')
        main_regression(var1, var1_region, var2, var2_region,
                        var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                        detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                        show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# cell to copmute and plot regression data
var1_list = ['2t', 'stl1', 'sstk', 'sshf', 'slhf', 'sd', 'swvl1']
var1_region = 'dsw'
# var1_region = 'cp'
var1_month_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
var2_list = ['z_height', 'z_thick_1000-500', 'msl', 'tp', 'onset', 'retreat', 'length', 'precipitation', 'precipitation-rate']
var2_region = 'cp'
# var2_region = 'dsw'
var2_month_list = [6, 7, 8]
var2_level = 500
detrend_flag = True
mean_flag = True
group_type = 'year'
show_flag = False
save_flag = True
overwrite_flag = False

for var1 in var1_list:
    for var1_month_list in var1_month_lists:
        for var2 in var2_list:
            print(f'var1: {var1} - {var1_month_list}\t-\tvar2: {var2}')
            main_regression(var1, var1_region, var2, var2_region,
                            var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                            detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                            show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# cell to copmute and plot regression data
var1_list = ['2t', 'stl1', 'sstk', 'sshf', 'slhf', 'sd', 'swvl1']
var1_region = 'dsw'
# var1_region = 'cp'
var1_month_list = [i for i in range(1,13)]
var2_list = ['z_height', 'z_thick_1000-500', 'msl', 'tp']
# var2_region = 'cp'
var2_region = 'dsw'
var2_month_list = [i for i in range(1,13)]
var2_level = 500
detrend_flag = True
mean_flag = False
group_type = 'time'
show_flag = False
save_flag = True
overwrite_flag = False

for var1 in var1_list:
    for var2 in var2_list:
        print(f'var1: {var1}\t-\tvar2: {var2}')
        main_regression(var1, var1_region, var2, var2_region,
                        var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                        detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                        show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# cell to copmute and plot regression data
var1_list = ['2t', 'stl1', 'sstk', 'sshf', 'slhf', 'sd', 'swvl1']
var1_region = 'dsw'
# var1_region = 'cp'
var1_month_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
var2_list = ['z_height', 'z_thick_1000-500', 'msl', 'tp', 'onset', 'retreat', 'length', 'precipitation', 'precipitation-rate']
# var2_region = 'cp'
var2_region = 'dsw'
var2_month_list = [6, 7, 8]
var2_level = 500
detrend_flag = True
mean_flag = True
group_type = 'year'
show_flag = False
save_flag = True
overwrite_flag = False

for var1 in var1_list:
    for var1_month_list in var1_month_lists:
        for var2 in var2_list:
            print(f'var1: {var1} - {var1_month_list}\t-\tvar2: {var2}')
            main_regression(var1, var1_region, var2, var2_region,
                            var1_month_list=var1_month_list, var2_month_list=var2_month_list, var2_level=var2_level,
                            detrend_flag=detrend_flag, mean_flag=mean_flag, group_type=group_type,
                            show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)

