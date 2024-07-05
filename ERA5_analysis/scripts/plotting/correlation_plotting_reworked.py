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


# In[ ]:


my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
sub_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/'  # path to subsetting scripts
plot_script_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/plotting/'  # path to plotting scripts
plot_out_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/'  # path to generated plots
temp_scratch_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/temp/'  # path to temp directory in scratch


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


# In[ ]:


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
    'z_thick': 'Geopotential Height Thickness',
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
    'z_thick': '$(m)$',
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
    'MeNmAz': 'Mexico, New Mexico, and Arizona Border',
}

# dictionary of regions and their coordinate boundaries
# [WEST, EAST, NORTH, SOUTH] -- WEST and EAST are on 0-360 latitude grid system
region_avg_coords = {
    'cp': [249, 253, 39, 35],
    'mr': [249, 251, 34, 33],
    'son': [246, 250, 32, 28],
    'chi': [252, 256, 33, 29],
    'moj': [243, 247, 37, 33],
    'MeNmAz': [246, 256, 38, 28],
}

# dictionary of colors for the plot of each region
region_colors_dict = {
    'cp': 'blue',
    'mr': 'darkorange',
    'son': 'green',
    'chi': 'red',
    'moj': 'purple',
    'MeNmAz': 'brown',
    'dsw': 'black'
}


# In[ ]:


# define main funciton to execute plotting based on type
def main(var1='swvl1', var1_month_list=[3, 4, 5], var1_region='cp',
         var2='tp', var2_month_list=[6, 7, 8], var2_region='dsw',
         detrend_flag=True, show_flag=True, save_flag=False, overwrite_flag=False, **kwargs):

    # months list
    var1_months = month_num_to_name(var=var1, months=var1_month_list)
    var2_months = month_num_to_name(var=var2, months=var2_month_list)

    # in/out file name and paths
    in_fn, in_fp, out_fn, out_fp, fn_core = get_fn_fp(var1, var1_months, var1_region,
                                                      var2, var2_months, var2_region,
                                                      detrend_flag, **kwargs)

    # check existence of input file
    if not os.path.exists(in_fp):
        print(f'corr file not found : {in_fp}')
        return

    # check existence of file already
    if ((os.path.exists(out_fp)) and (save_flag)):
        print(f'File already exists for: {out_fn}', end=' - ')
        if not overwrite_flag:
            print('overwrite_flag is False. Skipping . . .')
            return
        else:
            print('overwrite_flag is True. Overwriting . . .')

    # open dataset
    ds = xr.open_dataset(in_fp)
    corr_da = ds['pearson_r']
    pval_da = ds['p_value']

    # check if var is a flux and need to be flipped
    # only flip if one OR the other is a flux, but not both
    if ((var1 in flux_var_list) != (var2 in flux_var_list)):
        print(f'plotting flux\t-\tvar1: {var1}\t-\tvar2:{var2}')
        print(f'old corr_da avg: {corr_da.mean().compute()}')
        corr_da = corr_da * -1
        print(f'new corr_da avg: {corr_da.mean().compute()}')

    # plot the results
    if ((var1_region == 'dsw') or (var2_region == 'dsw')):
        plot_2d_correlation_dsw(corr_da, pval_da)
    elif ((var1_region == 'global') or (var2_region == 'global')):
        plot_2d_correlation_global(corr_da, pval_da)

    # plot features
    # plt.title(f'Correlation Between \n{var_dict[var1]} ({var1_months}, {var1_region}) \n& {var_dict[var2]} ({var2_months}, {var2_region})')
    plt.tight_layout()
    if show_flag:
        plt.show()
        plt.close('all')
    if save_flag:
        plt.savefig(out_fp, dpi=300, bbox_inches='tight')
        plt.close('all')

    ds.close()

    # return ds


# In[ ]:


# define a function to plot 2d map of correlations for dsw
def plot_2d_correlation_dsw(corr_da, pval_da):

    projection = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection=projection))

    # create contour levels and hatches for plotting
    corr_levels = np.arange(-1, 1.05, 0.05)

    # plot the data using contourf
    corr_cf = plt.contourf(corr_da.longitude, corr_da.latitude,
                           corr_da, levels=corr_levels,
                           cmap='RdBu_r', extend='both')

    # extract coordinates where p-value < 0.1 (dots) and p-value < 0.05 (triangles)
    lat, lon = np.meshgrid(pval_da.latitude, pval_da.longitude, indexing='ij')
    mask_dots = (pval_da <= 0.1) & (pval_da >= 0.05)
    mask_triangles = pval_da <= 0.05

    # Plot dots (p-value < 0.1 and >= 0.05)
    plt.scatter(lon[mask_dots], lat[mask_dots], color='black', marker='.',
                s=5, transform=ccrs.PlateCarree(), label='0.05 <= p < 0.1')

    # Plot triangles (p-value < 0.05)
    plt.scatter(lon[mask_triangles], lat[mask_triangles], color='black', marker='^',
                s=8, transform=ccrs.PlateCarree(), label='p < 0.05')

    # add coastlines, state borders, and other features
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    # plt.colorbar(corr_cf, ax=ax, label='Pearson Correlation Coefficient', pad=0.02)
    # plt.legend(bbox_to_anchor=(1, 1, 0.25, 0.15))


# In[ ]:


# define a function to plot 2d map of correlations for global
def plot_2d_correlation_global(corr_da, pval_da):

    projection = ccrs.PlateCarree(central_longitude=180)
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw=dict(projection=projection))

    # create contour levels and hatches for plotting
    corr_levels = np.arange(-1, 1.05, 0.05)

    # plot the data using contourf
    corr_cf = plt.contourf(corr_da.longitude, corr_da.latitude,
                           corr_da, levels=corr_levels,
                           cmap='RdBu_r', extend='both', transform=ccrs.PlateCarree())

    # parse lats/lons to declutter plot
    lat, lon = np.meshgrid(pval_da.latitude[::10], pval_da.longitude[::20], indexing='ij')

    # count the number of neighbors below the p-value thresholds for each point
    num_neighbors_below_0p1 = count_neighbors_below_threshold(pval_da, 0.1)
    num_neighbors_below_0p05 = count_neighbors_below_threshold(pval_da, 0.05)

    # create mask for dots and triangles
    mask_dots = (num_neighbors_below_0p1[::10, ::20] >= 4)
    mask_triangles = (num_neighbors_below_0p05[::10, ::20] >= 4)

    # Plot dots (p-value < 0.1 and >= 0.05)
    plt.scatter(lon[mask_dots], lat[mask_dots], color='black', marker='.',
                s=2, transform=ccrs.PlateCarree(), label='0.05 <= p < 0.1')

    # Plot triangles (p-value < 0.05)
    plt.scatter(lon[mask_triangles], lat[mask_triangles], color='black', marker='^',
                s=4, transform=ccrs.PlateCarree(), label='p < 0.05')

    # add coastlines, state borders, and other features
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.STATES, linewidth=0.5)

    # plt.colorbar(corr_cf, ax=ax, label='Pearson Correlation Coefficient', pad=0.02)
    # plt.legend(bbox_to_anchor=(1, 1, 0.25, 0.15))


# define a function to reduce number of lat/lon points plotted
# for p values for clarity.
# function assigns the center point of a 3x3 grid a pvalue based
# on the 8 points around it if 4 or more fall into one of 3 categories
# 0 - 0.05; 0.05 - 0.1; > 0.1
def count_neighbors_below_threshold(da, threshold):
    # create a mask where values are below the threshold
    mask_below_threshold = da < threshold

    # create a mask for the borders of the grid
    border_mask = np.ones_like(da, dtype=bool)
    border_mask[1:-1, 1:-1] = False

    # count the number of neighbors below the threshold for each point
    num_neighbors = np.zeros_like(da, dtype=int)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:  # exclude central point from calculation
                continue
            num_neighbors += mask_below_threshold.shift(latitude=i, longitude=j, fill_value=False).values

    # exclude the borders from the count
    num_neighbors[border_mask] = 0

    return num_neighbors


# In[ ]:


# define a function to turn a list of integers into months
def month_num_to_name(var, months):

    if var in NAM_var_list:
        var_months = ''
    elif len(months) == 1:
        var_months = calendar.month_name[months[0]]  # use full month name if only 1 month
    elif ((len(months) > 1) & (len(months) <= 12)):
        var_months = ''.join([calendar.month_name[m][0] for m in months])

    return var_months


# In[ ]:


# define a function to create and return the filename and
# file path for the input file and output plot
def get_fn_fp(var1, var1_months, var1_region,
               var2, var2_months, var2_region,
               detrend_flag=True, **kwargs):

    # check detrend_flag
    if detrend_flag:
        detrend_str = ''
    else:
        detrend_str = 'NOT-DETRENDED'

    # set var_level string for naming
    if var1 in pl_var_list:
        var1_level_str = kwargs.get('var1_level', 500)
    else:
        var1_level_str = ''
    if var2 in pl_var_list:
        var2_level_str = kwargs.get('var2_level', 500)
    else:
        var2_level_str = ''

    # create core of file name and in/out file name.
    # the .nc input file will match the .png output file for 2d corr plots
    fn_list = [str(var1), str(var1_level_str), str(var1_months), str(var1_region),
               str(var2), str(var2_level_str), str(var2_months), str(var2_region),
               str(detrend_str)]
    fn_core = '_'.join([i for i in fn_list if i != ''])

    in_fn = f'corr_{fn_core}.nc'
    out_fn = f'corr_{fn_core}.png'

    # create in and out file paths based on regions
    if (var1_region == 'global') or (var2_region == 'global'):
        in_fp = os.path.join(my_era5_path, 'correlations/global', in_fn)
        out_fp = os.path.join(plot_out_path, 'correlations/global', out_fn)
    elif (var1_region == 'dsw') or (var2_region == 'dsw'):
        in_fp = os.path.join(my_era5_path, 'correlations/dsw', in_fn)
        out_fp = os.path.join(plot_out_path, 'correlations/dsw', out_fn)
    elif (var1_region in region_avg_list) and (var2_region in region_avg_list):
        in_fp = os.path.join(my_era5_path, 'correlations/regions', in_fn)
        out_fp = os.path.join(plot_out_path, 'correlations/regions', out_fn)

    return in_fn, in_fp, out_fn, out_fp, fn_core


# In[ ]:


# define a function to check if inputs are list or not
def ensure_var_list(x):

    if not isinstance(x, list):
        return [x]
    return x


# In[ ]:


# define a function to call main() multiple times based on list of input variables
def execute_main_lists(var1_list, var1_month_lists, var1_level_list, var1_region_list,
                       var2_list, var2_month_lists, var2_level_list, var2_region_list,
                       detrend_flag=True, show_flag=True, save_flag=False, overwrite_flag=False):

    global cnt, start_time, len_lists

    # Ensure each input is a list
    var1_list = ensure_var_list(var1_list)
    var1_month_lists = ensure_var_list(var1_month_lists)
    var1_month_lists = [ensure_var_list(i) for i in var1_month_lists]  # list of lists
    var1_level_list = ensure_var_list(var1_level_list)
    var1_region_list = ensure_var_list(var1_region_list)
    var2_list = ensure_var_list(var2_list)
    var2_month_lists = ensure_var_list(var2_month_lists)
    var2_month_lists = [ensure_var_list(i) for i in var2_month_lists]  # list of lists
    var2_level_list = ensure_var_list(var2_level_list)
    var2_region_list = ensure_var_list(var2_region_list)

    # len_lists = (len(var1_list) * len(var1_month_lists) * len(var1_level_list) * len(var1_region_list) *
    #              len(var2_list) * len(var2_month_lists) * len(var2_level_list) * len(var2_region_list))
    # start_time = time.time()
    # cnt = 1

    for var1 in var1_list:
        for var1_month_list in var1_month_lists:
            for var1_level in var1_level_list:
                for var1_region in var1_region_list:
                    for var2 in var2_list:
                        for var2_month_list in var2_month_lists:
                            for var2_level in var2_level_list:
                                for var2_region in var2_region_list:
                                    # Print the current progress
                                    elapsed_time = time.time() - start_time
                                    progress = 100 * cnt / len_lists
                                    print(f'{i}: {var1}:\t{var1_month_list}:\t{var1_level}:\t{var1_region}:\n'
                                          f'{i}: {var2}:\t{var2_month_list}:\t{var2_level}:\t{var2_region}')

                                    # Call the main function with the current set of variables
                                    main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
                                         var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
                                         detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)

                                    print(f'progress: {cnt} / {len_lists} ({progress:.3f}%)\t: time elapsed={elapsed_time:.1f}s')

                                    cnt += 1


# In[ ]:


# define a function to check existence of large number of files
def check_for_files(var1_list, var1_month_lists, var1_level_list, var1_region_list,
                    var2_list, var2_month_lists, var2_level_list, var2_region_list,
                    detrend_flag=True, show_flag=True, save_flag=False, overwrite_flag=False):

    # Ensure each input is a list
    var1_list = ensure_var_list(var1_list)
    var1_month_lists = ensure_var_list(var1_month_lists)
    var1_month_lists = [ensure_var_list(i) for i in var1_month_lists]  # list of lists
    var1_level_list = ensure_var_list(var1_level_list)
    var1_region_list = ensure_var_list(var1_region_list)
    var2_list = ensure_var_list(var2_list)
    var2_month_lists = ensure_var_list(var2_month_lists)
    var2_month_lists = [ensure_var_list(i) for i in var2_month_lists]  # list of lists
    var2_level_list = ensure_var_list(var2_level_list)
    var2_region_list = ensure_var_list(var2_region_list)

    files_exist = []
    files_not_exist = []
    redundant_files = []

    for var1 in var1_list:
        for var1_month_list in var1_month_lists:
            for var1_level in var1_level_list:
                for var1_region in var1_region_list:
                    for var2 in var2_list:
                        for var2_month_list in var2_month_lists:
                            for var2_level in var2_level_list:
                                for var2_region in var2_region_list:

                                    # months list
                                    var1_months = month_num_to_name(var=var1, months=var1_month_list)
                                    var2_months = month_num_to_name(var=var2, months=var2_month_list)

                                    # in/out file name and paths
                                    in_fn, in_fp, out_fn, out_fp, fn_core = get_fn_fp(var1, var1_months, var1_region,
                                                                                      var2, var2_months, var2_region,
                                                                                      detrend_flag)

                                    if os.path.exists(in_fp) and in_fp not in files_exist and in_fp not in files_not_exist:
                                        files_exist.append(in_fp)
                                    elif not os.path.exists(in_fp) and in_fp not in files_not_exist:
                                        files_not_exist.append(in_fp)
                                    else:
                                        redundant_files.append(in_fp)

    return files_exist, files_not_exist, redundant_files


# In[ ]:


# # test cell

# var1 = 'sshf'
# var1_level = 500
# var1_month_list = [3, 4, 5]
# var1_region = 'cp'

# var2 = 'z_thick_1000-500'
# var2_level = 700
# var2_month_list = [6, 7, 8]
# var2_region = 'dsw'

# detrend_flag = True
# show_flag = True
# save_flag = False
# overwrite_flag = False

# main(var1, var1_month_list, var1_region,
#      var2, var2_month_list, var2_region,
#      detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# # plot correlations for dsw
# # calculate correlations for onset, retreat, length, and summer precipitation
# var_list1 = sfc_accumu_list + sfc_instan_list + pl_var_list
# var_list1.remove('sstk')
# var_list2 = NAM_var_list + ['tp', 'vipile', 'viwve', 'viwvn', 'viwvd']
# region_list = ['cp']
# detrend_list = [True, False]
# save_flag=False
# len_lists = len(var_list1)*len(var_list2)*len(region_list)*len(detrend_list)
# cnt = 0
# for var1 in var_list1:
#     for var2 in var_list2:
#         for region in region_list:
#             for detrend_flag in detrend_list:
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}\t:\t{var2}\t:\t{region}\t:\tdetrend={detrend_flag}\t:\t{100*cnt/len_lists} %\n')
#                 # print(f'{var1}\t:\t{var2}\t:\t{region}\t:\tdetrend={detrend_flag}\t:\t{100*cnt/len_lists} %')
#                 main(var1=var1, var1_month_list=[3, 4, 5], var1_region=region, var1_level=700,
#                      var2=var2, var2_month_list=[6, 7, 8], var2_region='dsw', var2_level=700,
#                      detrend_flag=detrend_flag, save_flag=save_flag, overwrite_flag=False)
#                 cnt = cnt+1


# In[ ]:


# # cell to plot specific correlations -- dsw
# var_list1 = sfc_accumu_list + sfc_instan_list + pl_var_list
# var_list1.remove('sstk')
# var1_month_lists = [[3, 4, 5]]
# var_list2 = ['precipitation', 'precipitation-rate']
# var2_month_lists = [[6, 7, 8]]
# len_lists = len(var_list1)*len(var_list2)*len(var1_month_lists)*len(var2_month_lists)
# cnt = 0
# for var1 in var_list1:
#     for var1_month_list in var1_month_lists:
#         for var2 in var_list2:
#             for var2_month_list in var2_month_lists:
#                 # with open(f'{der_script_path}corr.txt', 'a') as file:
#                 #     file.write(f'{var1}\t:\t{var2}\t:\t{region}\t:\tdetrend={detrend_flag}\t:\t{100*cnt/len_lists} %\n')
#                 print(f'{var1}\t:\t{var1_month_list}\t:\t{var2}\t:\t{var2_month_list}\t:\t{100*cnt/len_lists} %')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region='cp',
#                      var2=var2, var2_month_list=var2_month_list, var2_region='dsw',
#                      detrend_flag=True, save_flag=False, overwrite_flag=False)
#                 cnt = cnt+1


# In[ ]:


# # test cell -- global

# var1='ttr'
# var1_month_list=[3, 4, 5]
# var1_region='global'
# var2='onset'
# var2_month_list=[6, 7, 8]
# var2_region='MeNmAz'
# detrend_flag=True
# save_flag=False
# overwrite_flag=False

# main(var1, var1_month_list, var1_region,
#      var2, var2_month_list, var2_region,
#      detrend_flag, save_flag=save_flag, overwrite_flag)


# In[ ]:


# # test cell -- global

# var1='sstk'
# var1_month_list=[3, 4, 5]
# var1_region='global'
# var2='onset'
# var2_month_list=[6, 7, 8]
# var2_region='MeNmAz'
# detrend_flag=True
# save_flag=False
# overwrite_flag=False
# main(var1, var1_month_list, var1_region,
#      var2, var2_month_list, var2_region,
#      detrend_flag, save_flag=save_flag, overwrite_flag)


# In[ ]:


# # cell to plot correlations -- global
# var_list1 = ['ttr', 'sstk']
# var_list2 = NAM_var_list + ['tp', 'vipile', 'viwve', 'viwvn', 'viwvd']
# var1_months_list = [
#     [3, 4, 5],
#     [6, 7, 8]
# ]
# detrend_list = [True, False]
# len_lists = len(var_list1)*len(var_list2)*len(var1_months_list)*len(detrend_list)
# cnt = 0
# start_time = time.time()
# for var1 in var_list1:
#     for var2 in var_list2:
#         for var1_month_list in var1_months_list:
#             for detrend_flag in detrend_list:
#                 # print(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}\n')

#                 main(var1=var1, var1_month_list=var1_month_list, var1_region='global',
#                      var2=var2, var2_month_list=[6, 7, 8], var2_region='MeNmAz',
#                      detrend_flag=detrend_flag, save_flag=False, overwrite_flag=False)
#                 cnt=cnt+1


# In[ ]:


# # cell to plot specific correlations -- global
# var_list1 = ['ttr', 'sstk']
# var_list2 = ['precipitation', 'precipitation-rate']
# var1_months_list = [
#     [3, 4, 5],
#     [6, 7, 8]
# ]
# detrend_list = [True, False]
# len_lists = len(var_list1)*len(var_list2)*len(var1_months_list)*len(detrend_list)
# cnt = 0
# start_time = time.time()
# for var1 in var_list1:
#     for var2 in var_list2:
#         for var1_month_list in var1_months_list:
#             for detrend_flag in detrend_list:
#                 print(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 # with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 #     file.write(f'{var1}\t: {var2}\t: {var1_month_list}\t: detrend={detrend_flag}\t: {100*cnt/len_lists} %\t: time={time.time()-start_time}\n')

#                 main(var1=var1, var1_month_list=var1_month_list, var1_region='global',
#                      var2=var2, var2_month_list=[6, 7, 8], var2_region='MeNmAz',
#                      detrend_flag=detrend_flag, save_flag=False, overwrite_flag=False)
#                 cnt=cnt+1


# In[ ]:


# # cell to plot nino-3 correlations
# var1 = 'nino-3'
# var1_months_list = [[i, i+1, i+2] for i in range(1,11)]  # create list of list [[1,2,3], [2,3,4], ... [10,11,12]]
# var1_region = ''

# var2_list = ['onset', 'retreat', 'length', 'precipitation', 'precipitation-rate']
# var2_month_list = [6, 7, 8]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=False

# for var2 in var2_list:
#     print(f'var - {var2}: ')
#     for var1_month_list in var1_months_list:
#         print(f'\t{var1_month_list}\t', end='')
#         main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region,
#              var2=var2, var2_month_list=var2_month_list, var2_region=var2_region,
#              detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# # cell to plot a bunch of moving window correlations previously defined below

# # common to all iterations

# # var1_list = ['sd', 'swvl1', 'stl1', '2t', 'tp', 'sf', 'sshf', 'slhf', 'ssr', 'str', 'ssrd', 'strd', 'sro', 'z_height', 'z_thick_1000-500', 'u', 'v']
# # var1_month_lists = [[i, i+1, i+2] for i in range(1,11)]  # create list of list [[1,2,3], [2,3,4], ... [10,11,12]]
# var1_level_list = [700, 500]
# var1_region_list = ['cp', 'MeNmAz']

# # var2_list = ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'z_height', 'z_thick_1000-500', 'cape', 'cp']
# # var2_month_lists = [[3, 4, 5], [6, 7, 8]]
# var2_level_list = [700, 500]
# var2_region_list = ['dsw']

# detrend_flag = True
# show_flag = False
# save_flag = True
# overwrite_flag = False


# # specific to each run
# var1_list_master = [
#     ['sd', 'swvl1', 'stl1', '2t', 'tp', 'sf', 'sshf', 'slhf', 'ssr', 'str', 'ssrd', 'strd', 'sro', 'z_height', 'z_thick_1000-500', 'msl', 'u', 'v'],
#     # ['onset', 'retreat', 'length'],
#     # ['sd', 'swvl1'],
#     # ['ssr', 'str'],
#     ['sshf', 'slhf'],
#     ['2t', 'stl1'],
#     ['z_height', 'z_thick_1000-500', 'msl'],
#     # ['u', 'v'],
#     # ['viwvn', 'viwve'],
# ]
# var1_month_lists_master = [
#     [[i, i+1, i+2] for i in range(1,11)],  # create list of list [[1,2,3], [2,3,4], ... [10,11,12]]
#     # [[3, 4, 5]],
#     # [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
#     # [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
#     [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
#     [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
#     [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
#     # [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
#     # [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]],
# ]

# var2_list_master = [
#     ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'z_height', 'z_thick_1000-500', 'msl', 'cape', 'cp'],
#     # ['tp', 'precipitation', 'precipitation-rate'],
#     # ['ssr', 'str', 'sshf', 'slhf'],
#     # ['sshf', 'slhf', '2t', 'stl1'],
#     ['2t', 'stl1', 'z_height', 'z_thick_1000-500', 'msl'],
#     ['z_height', 'z_thick_1000-500', 'msl', 'u', 'v'],
#     ['u', 'v', 'viwvn', 'viwve', 'onset', 'retreat', 'length', 'precipitation', 'precipitation-rate', 'cape', 'cp'],
#     # ['viwvn', 'viwve', 'onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'cape', 'cp'],
#     # ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'cape', 'cp'],
# ]
# var2_month_lists_master = [
#     [[3, 4, 5], [6, 7, 8]],
#     # [[3, 4, 5], [6, 7, 8]],
#     # [[i, i+1, i+2] for i in range(1,11)],
#     # [[i, i+1, i+2] for i in range(1,11)],
#     [[i, i+1, i+2] for i in range(1,11)],
#     [[i, i+1, i+2] for i in range(1,11)],
#     [[i, i+1, i+2] for i in range(1,11)],
#     # [[i, i+1, i+2] for i in range(1,11)],
#     # [[i, i+1, i+2] for i in range(1,11)],
# ]


# len_lists = 0
# for i in range(len(var1_list_master)):
#     len_lists_master = (len(var1_list_master[i]) * len(var1_month_lists_master[i]) * len(var1_level_list) * len(var1_region_list) *
#                         len(var2_list_master[i]) * len(var2_month_lists_master[i]) * len(var2_level_list) * len(var2_region_list))
#     len_lists += len_lists_master

# start_time = time.time()
# cnt = 1

# for i in range(len(var1_list_master)):
#     execute_main_lists(var1_list=var1_list_master[i], var1_month_lists=var1_month_lists_master[i], var1_level_list=var1_level_list, var1_region_list=var1_region_list,
#                        var2_list=var2_list_master[i], var2_month_lists=var2_month_lists_master[i], var2_level_list=var2_level_list, var2_region_list=var2_region_list,
#                        detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# cell to plot moving window correlations
var1_list = ['sd', 'swvl1', 'stl1', '2t', 'tp', 'sf', 'sshf', 'slhf', 'ssr', 'str', 'ssrd', 'strd', 'sro', 'z_height', 'z_thick_1000-500', 'msl', 'u', 'v']
var1_months_list = [[i, i+1, i+2] for i in range(1,11)]  # create list of list [[1,2,3], [2,3,4], ... [10,11,12]]
# var1_region = 'cp'
var1_regions = ['cp', 'MeNmAz']

var2_list = ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'z_height', 'z_thick_1000-500', 'msl', 'cape', 'cp']
# var2_months_list = [[3, 4, 5], [6, 7, 8]]
var2_months_list = [[4, 5, 6], [5, 6, 7]]
var2_region = 'dsw'

var1_levels = [700, 500]
var2_levels = [700, 500]

detrend_flag = True
overwrite_flag = False
show_flag = False
save_flag = True
len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list) * len(var1_regions) * len(var1_levels)
start_time = time.time()
cnt=1
for var1 in var1_list:
    for var2 in var2_list:
        for var2_month_list in var2_months_list:
            for var1_month_list in var1_months_list:
                for var1_region in var1_regions:
                    for i in range(len(var1_levels)):
                        print(f'{var1}:\t{var1_month_list}:\t{var1_region}:\t--\t{var2}:\t{var2_month_list}', end='\t:')
                        main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_levels[i],
                             var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_levels[i],
                             detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
                        print(f'{100*cnt/len_lists:.2f} %\t: time={time.time()-start_time:.2f}')
                        cnt=cnt+1


# In[ ]:


# # cell to plot more moving window correlations
# var1_list = ['onset', 'retreat', 'length']
# var1_month_list = [3, 4, 5]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['tp', 'precipitation', 'precipitation-rate']
# var2_month_list = [[3, 4, 5], [6, 7, 8]]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=False
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var2_month_list in var2_months_list:
#             # print(f'{var1}:\t{var2}:\t{var2_month_list}\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var2}:\t{var2_month_list}\t: ')
#             main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                  var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                  detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt=cnt+1


# In[ ]:


# # cell to plot moving correlations of snow depth/soil moisture and net solar/thermal radiation or sensible/latent heat
# var1_list = ['sd', 'swvl1']
# var1_months_list = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['ssr', 'str', 'sshf', 'slhf']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=True
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_month_list in var1_months_list:
#             for var2_month_list in var2_months_list:
#                 # print(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ', end='')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                      detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#                 # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#                 cnt=cnt+1


# In[ ]:


# # cell to plot moving correlations of net solar/thermal radiation and sensible/latent heat or soil/2m temperature
# var1_list = ['ssr', 'str']
# var1_months_list = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['sshf', 'slhf', '2t', 'stl1']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=True
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_month_list in var1_months_list:
#             for var2_month_list in var2_months_list:
#                 # print(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ', end='')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                      detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#                 # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#                 cnt=cnt+1


# In[ ]:


# # cell to plot moving correlations of sensible/latent heat and soil/2m temperature geopotential
# var1_list = ['sshf', 'slhf']
# var1_months_list = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['2t', 'stl1', 'z_height', 'z_thick_1000-500', 'msl']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=True
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_month_list in var1_months_list:
#             for var2_month_list in var2_months_list:
#                 # print(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ', end='')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                      detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#                 # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#                 cnt=cnt+1


# In[ ]:


# # cell to plot moving correlations of soil/2m temperature and geopotential or u/v wind
# var1_list = ['2t', 'stl1']
# var1_months_list = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['z_height', 'z_thick_1000-500', 'msl', 'u', 'v']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=False
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_month_list in var1_months_list:
#             for var2_month_list in var2_months_list:
#                 # print(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ', end='')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                      detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#                 # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#                 cnt=cnt+1


# In[ ]:


# # cell to plot moving correlations of geopotential and u/v wind or N/E water vapor transport
# var1_list = ['z_height', 'z_thick_1000-500', 'msl']
# var1_months_list = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['u', 'v', 'viwvn', 'viwve', 'onset', 'retreat', 'length', 'precipitation', 'precipitation-rate', 'cape', 'cp']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=False
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_month_list in var1_months_list:
#             for var2_month_list in var2_months_list:
#                 # print(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ', end='')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                      detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#                 # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#                 cnt=cnt+1


# In[ ]:


# # cell to plot moving correlations of u/v wind and N/E water vapor transport or NAM stuff
# var1_list = ['u', 'v']
# var1_months_list = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['viwvn', 'viwve', 'onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'cape', 'cp']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=False
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_month_list in var1_months_list:
#             for var2_month_list in var2_months_list:
#                 # print(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ', end='')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                      detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#                 # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#                 cnt=cnt+1


# In[ ]:


# # cell to plot moving correlations of N/E water vapor transport and NAM stuff
# var1_list = ['viwvn', 'viwve']
# var1_months_list = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# # var1_region = 'cp'
# var1_region = 'MeNmAz'

# var2_list = ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'cape', 'cp']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# var1_level = 700
# var2_level = 700

# detrend_flag=True
# overwrite_flag=False
# show_flag=False
# save_flag=True
# len_lists = len(var1_list) * len(var2_list) * len(var1_months_list) * len(var2_months_list)
# start_time = time.time()
# cnt=0
# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_month_list in var1_months_list:
#             for var2_month_list in var2_months_list:
#                 # print(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ', end='')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{var1}:\t{var1_month_list}:\t{var2}:\t{var2_month_list}\t: ')
#                 main(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region, var1_level=var1_level,
#                      var2=var2, var2_month_list=var2_month_list, var2_region=var2_region, var2_level=var2_level,
#                      detrend_flag=detrend_flag, show_flag=show_flag, save_flag=save_flag, overwrite_flag=overwrite_flag)
#                 # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#                 with open(f'{plot_script_path}plot.txt', 'a') as file:
#                     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#                 cnt=cnt+1


# In[ ]:


# define main funciton to execute plotting based on type
def main_multi_region(var1='sd', var1_month_list=[3, 4, 5], var1_region='cp',
                      var2_list=['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate'],
                      var2_month_list=[6, 7, 8], var2_region_list=['cp', 'mr', 'MeNmAz', 'son', 'moj', 'chi', 'dsw'],
                      detrend_flag=True, overwrite_flag=False, **kwargs):

    # check detrend_flag
    if detrend_flag:
        detrend_str = ''
    else:
        detrend_str = 'NOT-DETRENDED'

    # set var_level string for naming
    if var1 in pl_var_list:
        var1_level_str = kwargs.get('var1_level', 500)
    else:
        var1_level_str = ''

    # make sure regions are in a list for iterating
    var2_list = ensure_var_list(var2_list)
    var2_region_list = ensure_var_list(var2_region_list)

    # months list
    var1_months = month_num_to_name(var=var1, months=var1_month_list)
    var2_months = []  # initialize var2_months
    for var2 in var2_list:
        var2_months.append(month_num_to_name(var=var2, months=var2_month_list))

    # input filename and paths
    in_fn_list = []  # initialize in_fn_list
    in_fp_list = []  # initialize in_fp_list
    for i in range(len(var2_list)):
        in_fn, in_fp, fn_core = get_in_fn_fp_multi(var1, var1_level_str, var1_months, var1_region,
                                                   var2_list[i], var2_months[i], 'dsw',
                                                   detrend_str, **kwargs)
        in_fn_list.append(in_fn)
        in_fp_list.append(in_fp)
    # output filename and path
    out_fn, out_fp, fn_core = get_out_fn_fp_multi(var1, var1_level_str, var1_months, var1_region,
                                                  var2_list, month_num_to_name(var=None, months=var2_month_list), 'regions',
                                                  detrend_str, **kwargs)

    # check existence of input files
    for in_fp in in_fp_list:
        if not os.path.exists(in_fp):
            print(f'corr file not found : {in_fp}')
            return

    # check existence of file already
    if os.path.exists(out_fp):
        print(f'File already exists for: {out_fn}', end=' - ')
        if not overwrite_flag:
            print('overwrite_flag is False. Skipping . . .')
            return
        else:
            print('overwrite_flag is True. Overwriting . . .')

    # open datasets
    corr_da_list = []  # initialize corr_da_list
    pval_da_list = []  # initialize pval_da_list
    for in_fp in in_fp_list:
        ds = xr.open_dataset(in_fp)
        corr_da_list.append(ds['pearson_r'])
        pval_da_list.append(ds['p_value'])

    # check if var is a flux and need to be flipped
    if ((var1 in flux_var_list) or (var2 in flux_var_list)):
        print(f'plotting flux\t-\tvar1: {var1}\t-\tvar2:{var2}')
        corr_da_list = [corr_da * -1 for corr_da in corr_da_list]

    # calculate % of positive/negative correlation values - return dict
    pos_dict = {}  # initialize pos_dict
    neg_dict = {}  # initialize neg_dict
    for region in var2_region_list:
        pos_dict[region], neg_dict[region] = calc_pos_neg_correlation(corr_da_list, pval_da_list, region, **kwargs)

    # plot the results

    plot_multi_region_correlation(pos_dict, neg_dict, var2_list)

    # plot features
    plt.title(f'Percentage of pos/neg correlations with p-values <= 0.1 \n {var_dict[var1]}')
    plt.tight_layout()
    plt.savefig(out_fp, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

    # return ds


# In[ ]:


# define a function to create and return the filename and
# file path for the input file
def get_in_fn_fp_multi(var1, var1_level_str, var1_months, var1_region,
                       var2, var2_months, var2_region,
                       detrend_str, **kwargs):

    # create core of the input file name.
    fn_list = [str(var1), str(var1_level_str), str(var1_months), str(var1_region),
               str(var2), str(var2_months), 'dsw',
               str(detrend_str)]
    fn_core = '_'.join([i for i in fn_list if i != ''])
    # create input filename and path
    in_fn = f'corr_{fn_core}.nc'
    in_fp = os.path.join(my_era5_path, 'correlations/dsw', in_fn)

    return in_fn, in_fp, fn_core


# In[ ]:


# define a function to create and return the filename and
# file path for the output plot
def get_out_fn_fp_multi(var1, var1_level_str, var1_months, var1_region,
                        var2, var2_months, var2_region,
                        detrend_str, **kwargs):

    # create core of output file
    fn_list = [str(var1), str(var1_level_str), str(var1_months), str(var1_region),
               'NAM', str(var2_months), 'regions',
               str(detrend_str)]
    fn_core = '_'.join([i for i in fn_list if i != ''])
    # output filename and path
    out_fn = f'corr_{fn_core}.png'
    out_fp = os.path.join(plot_out_path, 'correlations/regions', out_fn)

    return out_fn, out_fp, fn_core


# In[ ]:


# define a function to calculate pos/neg correlations based
# on p value and return dict for a region
def calc_pos_neg_correlation(corr_da_list, pval_da_list, region, **kwargs):

    # create list of correlations and pvalues
    pos_corr_list = []  # initialize pos_corr_list
    neg_corr_list = []  # initialize neg_corr_list

    for i in range(len(corr_da_list)):
        if region in region_avg_list:
            corr_da_sub = subset_da_region(corr_da_list[i], region_avg_coords[region])
            pval_da_sub = subset_da_region(pval_da_list[i], region_avg_coords[region])
        else:
            corr_da_sub = subset_da_region(corr_da_list[i], [240, 260, 40, 20])
            pval_da_sub = subset_da_region(pval_da_list[i], [240, 260, 40, 20])

        pos_corr = ((corr_da_sub > 0) & (pval_da_sub <= 0.1)).sum().values
        neg_corr = ((corr_da_sub < 0) & (pval_da_sub <= 0.1)).sum().values * -1  #  -- MULTIPLY BY -1 TO SHOW NEGATIVES

        da_size = corr_da_sub.size  # total number of grid points

        # percentage of pos/neg corr values
        pos_corr_list.append((pos_corr / da_size)*100)
        neg_corr_list.append((neg_corr / da_size)*100)

    return pos_corr_list, neg_corr_list


# In[ ]:


# define a function to subset data arrays by regions
def subset_da_region(da, coords, **kwargs):

    # slice ds into region coords
    west, east, north, south = coords[0], coords[1], coords[2], coords[3]
    da_sub = da.sel(latitude=slice(north, south), longitude=slice(west, east))

    return da_sub


# In[ ]:


# define a function to plot the multi region correlations
def plot_multi_region_correlation(pos_dict, neg_dict, var2_list):

    fig, ax = plt.subplots(figsize=(12,6))

    x = np.arange(len(var2_list))  # the label locations
    width = 0.1  # the width of the bars

    # plot positive correlations
    multiplier = 0  # multiplier for each iteration of plotting
    for region, corr_values in pos_dict.items():
        if region in region_avg_dict:
            label = region_avg_dict[region]
        else:
            label = 'Desert Southwest'
        offset = width * multiplier
        rects = ax.bar(x + offset, corr_values, width, label=label, color=region_colors_dict[region])
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # plot negative correlations
    multiplier = 0  # multiplier for each iteration of plotting
    for region, corr_values in neg_dict.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, corr_values, width, color=region_colors_dict[region])
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    # add text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('( % )')
    ax.set_xticks(x+width)
    ax.set_xticklabels(var2_list)
    ax.legend(loc='upper right', ncol=int(np.ceil(len(pos_dict)/2)))
    ax.set_ylim(-100, 100)


# In[ ]:


# # cell to plot bar correlations for regions
# var1_list = ['swvl1', 'sd', 'tp', '2t', 'sshf', 'slhf', 'ssr', 'str']
# var1_month_list=[3, 4, 5]
# var1_region='cp'
# var2_list=['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate']
# var2_month_list=[6, 7, 8]
# var2_region_list=['cp', 'mr', 'MeNmAz', 'son', 'moj', 'chi', 'dsw']
# detrend_flag=True
# overwrite_flag=False

# for var1 in var1_list:
#     main_multi_region(var1=var1, var1_month_list=var1_month_list, var1_region=var1_region,
#                       var2_list=var2_list, var2_month_list=var2_month_list, var2_region_list=var2_region_list,
#                       detrend_flag=detrend_flag, overwrite_flag=overwrite_flag)


# In[ ]:


# define a function to show multiple pre-made plots in a panel
def show_plots_multi(var1, var1_months_list, var1_region,
                     var2, var2_months_list, var2_region,
                     detrend_flag=True, show_flag=True, save_flag=False, overwrite_flag=False, **kwargs):

    # check detrend_flag
    if detrend_flag:
        detrend_str = ''
    else:
        detrend_str = 'NOT-DETRENDED'

    # set var_level string for naming
    if var1 in pl_var_list:
        var1_level_str = kwargs.get('var1_level', 700)
    else:
        var1_level_str = ''
    if var2 in pl_var_list:
        var2_level_str = kwargs.get('var2_level', 700)
    else:
        var2_level_str = ''

    # get month names and fn_core name for var months
    var1_month_names, var1_month_fn_name = get_var_month_fn_name(var1, var1_months_list)
    var2_month_names, var2_month_fn_name = get_var_month_fn_name(var2, var2_months_list)

    # get output file name/paths for new plot
    fn_list = [str(var1), str(var2_level_str), str(var1_month_fn_name), str(var1_region),
               str(var2), str(var2_level_str), str(var2_month_fn_name), str(var2_region),
               str(detrend_str)]
    fn_core = '_'.join([i for i in fn_list if i != ''])

    out_fn = f'corr_{fn_core}.png'
    out_fp = os.path.join(plot_out_path, 'correlations/dsw/moving-window', out_fn)

    # check existence of output file
    if (os.path.exists(out_fp) & save_flag):
        print(f'output file exists for {out_fn}', end='')
        if not overwrite_flag:  # DON'T overwrite
            print('overwrite_flag is set to False. Set to True to overwrite.')
            return
        else:  # overwrite
            print('overwrite_flag is set to True. overwriting . . .')

    # get input file name/paths for plots
    filenames = []
    filepaths = []

    # vor var1 moving window
    if isinstance(var1_months_list[0], list):
        var2_months = var2_month_names
        for var1_months in var1_month_names:
            _, _, filename, filepath, fn_core = get_fn_fp(var1, var1_months, var1_region,
                                                          var2, var2_months, var2_region,
                                                          detrend_flag, **kwargs)
            # append name and path lists
            filenames.append(filename)
            filepaths.append(filepath)

    # for var2 moving window
    elif isinstance(var2_months_list[0], list):
        var1_months = var1_month_names
        for var2_months in var2_month_names:
            _, _, filename, filepath, fn_core = get_fn_fp(var1, var1_months, var1_region,
                                                          var2, var2_months, var2_region,
                                                          detrend_flag, **kwargs)
            # append name and path lists
            filenames.append(filename)
            filepaths.append(filepath)

    # check existence of input files
    for filepath in filepaths:
        if not os.path.exists(filepath):
            print(f'\ncorr file not found : {filename}\n')
            return
    else:
        print('all filepaths verified')

    if (os.path.exists(out_fp) & show_flag & ~overwrite_flag):
        print('plot exitsts, showing . . .')
        img = mpimg.imread(out_fp)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        plt.close()
        return
    else:
        # create figure with 5 columns
        fig, axes = plt.subplots(int(np.ceil(len(filepaths))/5),5, figsize=(24,12))
        axes = axes.flatten()

        # loop through filepaths and plot
        if isinstance(var1_months_list[0], list):
            var_month_names = var1_month_names
        elif isinstance(var2_months_list[0], list):
            var_month_names = var2_month_names

        for ax, filepath, var_months in zip(axes, filepaths, var_month_names):
            img = mpimg.imread(filepath)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{var_months}', fontsize=20)

        # adjust layout
        fig.suptitle(f'Correlation Between \n{var_dict[var1]} ({var1_region}, {var1_month_fn_name}) \n& {var_dict[var2]} ({var2_region}, {var2_month_fn_name})', fontsize=25)
        plt.tight_layout()

        # show or save plot
        if show_flag:
            print('showing plot')
            plt.show()
            plt.close()
        if save_flag:
            print('saving plot')
            plt.savefig(out_fp)
            plt.close()


# In[ ]:


# define a function to get var month names for fn_core
def get_var_month_fn_name(var, var_months_list, **kwargs):

    # check if var_months is a list of lists, i.e., [[1,2,3], [2,3,4], ...]
    if isinstance(var_months_list[0], list):
        var_month_names = []
        # get var1 month names in a list
        for var_month_list in var_months_list:
            var_month_names.append(month_num_to_name(var=var, months=var_month_list))
        # fn_core names for moving window
        var_month_fn_name = f'{var_month_names[0]}-{var_month_names[-1]}'

    # if not list of lists, just take return of function
    else:
        var_month_names = month_num_to_name(var=var, months=var_months_list)
        var_month_fn_name = var_month_names

    return var_month_names, var_month_fn_name


# In[ ]:


# define a function to open and show pre-made plots
def show_plot_png(filepath, **kwargs):

    img = mpimg.imread(filepath)

    plt.imshow(img)
    plt.axis('off')
    plt.show()


# In[ ]:


# # test cell for showing a panel of plots with function
# var1 = 'sd'
# var1_months_list = [3, 4, 5]
# var1_region = 'cp'

# var2 = 'z_thick_1000-500'
# var2_months_list = [[i, i+1, i+2] for i in range(3,7)]  # create list of list [[1,2,3], [2,3,4], ... [10,11,12]]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=True
# save_flag=False
# overwrite_flag=False

# show_plots_multi(var1, var1_months_list, var1_region,
#                  var2, var2_months_list, var2_region,
#                  detrend_flag, show_flag, save_flag, overwrite_flag)


# In[ ]:


# # cell to show panel plot moving correlations of snow depth/soil moisture and
# # net solar/thermal radiation or sensible/latent heat
# var1_list = ['sd', 'swvl1']
# var1_months_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# var1_region = 'cp'

# var2_list = ['ssr', 'str', 'sshf', 'slhf']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var1_months_lists) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_months_list in var1_months_lists:
#             # print(f'{var1}:\t{var1_months_list}:\t{var2}:\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var1_months_list}:\t{var2}:\n')

#             show_plots_multi(var1, var1_months_list, var1_region,
#                              var2, var2_months_list, var2_region,
#                              detrend_flag, show_flag, save_flag, overwrite_flag)

#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt = cnt+1


# In[ ]:


# # cell to show panel plot moving correlations of net solar/thermal radiation
# # and sensible/latent heat or soil/2m temperature
# var1_list = ['ssr', 'str']
# var1_months_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# var1_region = 'cp'

# var2_list = ['sshf', 'slhf', '2t', 'stl1']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var1_months_lists) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_months_list in var1_months_lists:
#             # print(f'{var1}:\t{var1_months_list}:\t{var2}:\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var1_months_list}:\t{var2}:\n')

#             show_plots_multi(var1, var1_months_list, var1_region,
#                              var2, var2_months_list, var2_region,
#                              detrend_flag, show_flag, save_flag, overwrite_flag)

#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt = cnt+1


# In[ ]:


# # cell to show panel plot moving correlations of sensible/latent heat
# # and soil/2m temperature and geopotential
# var1_list = ['sshf', 'slhf']
# var1_months_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# var1_region = 'cp'

# var2_list = ['2t', 'stl1', 'z']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var1_months_lists) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_months_list in var1_months_lists:
#             # print(f'{var1}:\t{var1_months_list}:\t{var2}:\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var1_months_list}:\t{var2}:\n')

#             show_plots_multi(var1, var1_months_list, var1_region,
#                              var2, var2_months_list, var2_region,
#                              detrend_flag, show_flag, save_flag, overwrite_flag)

#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt = cnt+1


# In[ ]:


# # cell to show panel plot moving correlations of soil/2m temperature
# # and geopotential or u/v wind
# var1_list = ['2t', 'stl1']
# var1_months_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# var1_region = 'cp'

# var2_list = ['z', 'u', 'v']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var1_months_lists) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_months_list in var1_months_lists:
#             # print(f'{var1}:\t{var1_months_list}:\t{var2}:\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var1_months_list}:\t{var2}:\n')

#             show_plots_multi(var1, var1_months_list, var1_region,
#                              var2, var2_months_list, var2_region,
#                              detrend_flag, show_flag, save_flag, overwrite_flag)

#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt = cnt+1


# In[ ]:


# # cell to show panel plot moving correlations of geopotential
# # and u/v wind or N/E water vapor transport
# var1_list = ['z']
# var1_months_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# var1_region = 'cp'

# var2_list = ['u', 'v', 'viwvn', 'viwve']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var1_months_lists) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_months_list in var1_months_lists:
#             # print(f'{var1}:\t{var1_months_list}:\t{var2}:\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var1_months_list}:\t{var2}:\n')

#             show_plots_multi(var1, var1_months_list, var1_region,
#                              var2, var2_months_list, var2_region,
#                              detrend_flag, show_flag, save_flag, overwrite_flag)

#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt = cnt+1


# In[ ]:


# # cell to show panel plot moving correlations of u/v wind
# # and N/E water vapor transport or NAM stuff
# var1_list = ['u', 'v']
# var1_months_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# var1_region = 'cp'

# var2_list = ['viwvn', 'viwve', 'onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'cape', 'cp']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var1_months_lists) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_months_list in var1_months_lists:
#             # print(f'{var1}:\t{var1_months_list}:\t{var2}:\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var1_months_list}:\t{var2}:\n')

#             show_plots_multi(var1, var1_months_list, var1_region,
#                              var2, var2_months_list, var2_region,
#                              detrend_flag, show_flag, save_flag, overwrite_flag)

#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt = cnt+1


# In[ ]:


# # cell to show panel plot moving correlations of N/E water vapor transport
# # and NAM stuff
# var1_list = ['viwvn', 'viwve']
# var1_months_lists = [[3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]]
# var1_region = 'cp'

# var2_list = ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate', 'cape', 'cp']
# var2_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var1_months_lists) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         for var1_months_list in var1_months_lists:
#             # print(f'{var1}:\t{var1_months_list}:\t{var2}:\t: ', end='')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{var1}:\t{var1_months_list}:\t{var2}:\n')

#             show_plots_multi(var1, var1_months_list, var1_region,
#                              var2, var2_months_list, var2_region,
#                              detrend_flag, show_flag, save_flag, overwrite_flag)

#             # print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#             with open(f'{plot_script_path}plot.txt', 'a') as file:
#                 file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#             cnt = cnt+1


# In[ ]:


# # cell to show panel plot moving correlations of N/E water vapor transport
# # and NAM stuff
# var1_list = ['nino-3']
# var1_months_list = [[i, i+1, i+2] for i in range(1,11)]
# var1_region = ''

# var2_list = ['onset', 'retreat', 'length', 'tp', 'precipitation', 'precipitation-rate']
# var2_months_list = [6, 7, 8]
# var2_region = 'dsw'

# detrend_flag=True
# show_flag=False
# save_flag=True
# overwrite_flag=True

# len_lists = len(var1_list) * len(var2_list)
# cnt = 1
# start_time = time.time()

# for var1 in var1_list:
#     for var2 in var2_list:
#         print(f'{var1}:\t{var2}:\t: ', end='')
#         # with open(f'{plot_script_path}plot.txt', 'a') as file:
#         #     file.write(f'{var1}:\t{var2}:\n')

#         show_plots_multi(var1, var1_months_list, var1_region,
#                          var2, var2_months_list, var2_region,
#                          detrend_flag, show_flag, save_flag, overwrite_flag)

#         print(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}')
#         # with open(f'{plot_script_path}plot.txt', 'a') as file:
#         #     file.write(f'{100*cnt/len_lists} %\t: time={time.time()-start_time}\n')
#         cnt = cnt+1


# In[ ]:




