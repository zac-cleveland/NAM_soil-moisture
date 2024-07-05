#!/usr/bin/env python
# coding: utf-8

# This script plots 2-D plots of a given variable corresponding to a given year relative to another variable. 
# For example, if specified, plot the average June, July, August, and September 500 mb geopotential heights for 
# the year in which the most precipitation fell in those same months over the Colorado Plateau.  The user can 
# specify if they want to plot multiple years.  All months in that year will be plotted in 1 figure.  The user can 
# also choose if they want to plot the averaged years or months, or plot them individually.  

# In[11]:


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

my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
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
if 'order_years' in sys.modules:
    importlib.reload(sys.modules['order_years'])

# import common functions that I've created
from get_var_data import get_var_data, get_var_files, open_var_data, subset_var_data, time_to_year_month_avg, time_to_year_month_sum, time_to_year_month
from my_functions import month_num_to_name, ensure_var_list
from order_years import *  # order_years(var, months, **kwargs)

# import lists and dictionaries
import my_dictionaries
# my lists
sfc_instan_list = my_dictionaries.sfc_instan_list  # instantaneous surface variables
sfc_accumu_list = my_dictionaries.sfc_accumu_list  # accumulated surface variables
pl_var_list = my_dictionaries.pl_var_list  # pressure level variables
invar_var_list = my_dictionaries.invar_var_list  # invariant variables
NAM_var_list = my_dictionaries.NAM_var_list  # NAM-based variables
region_avg_list = my_dictionaries.region_avg_list  # region IDs for regional averages
flux_var_list = my_dictionaries.flux_var_list  # flux variables that need to be flipped (e.g., sensible heat so that it's positive up instead of down
misc_var_list = my_dictionaries.misc_var_list  # misc variables
# my dictionaries
var_dict = my_dictionaries.var_dict  # variables and their names
var_units = my_dictionaries.var_units  # variable units
region_avg_dict = my_dictionaries.region_avg_dict  # region IDs and names
region_avg_coords = my_dictionaries.region_avg_coords  # coordinates for regions
region_colors_dict = my_dictionaries.region_colors_dict  # colors to plot for each region


# In[73]:


def main(var1, var1_months, var1_region, var2, var2_months, var2_region, rank_years=[1, 2, 3, 4, 5], **kwargs):
    """
    Synchronizes the data in var1 and var2 by sorting the yearly data (summed or averaged over specified months)
    in descending order. e.g., most -> least precipitation in MJJASO at some lat/lon combo = (1981, 1990, ...2004).
    SFC accumulated variables are summed and all others are averaged.

    Parameters
    ----------
    var1 : str
        The variable used to do the sorting of years.
    var1_months : list, int
        The months to consider in summing or averaging var1.
    var1_region : str
        The region over which to average latitude/longitude points of var1.
    var2 : str
        The variable to plot relative to the sorted order of var1.
    var2_months : list, int
        The months to consider in summing or averaging var2.
    var2_region : str
        The region over which to average latitude/longitude points of var2.
    rank_years : list, int, optional
        A list of the number of year to plot by rank order. i.e., 1 is most and 40 is least. default is 5 max years.
    """
    # get sorted years info for var1
    sorted_years = get_sorted_years(var1, var1_months, var1_region)  # years in descending order by var1 mean or sum
    # var_years = sorted_years.sel(sorted_years=rank_years)  # just the ranked years
    var_years_mode = sorted_years.sel(sorted_years=rank_years).values  # just the ranked years

    # subset var_years by lat/lon
    # var_years_subset = subset_lat_lon(var_years, var1_region)

    # get rank ordered list of years on regional average
    # var_years_mode = apply_region_mode(var_years_subset)
    # if len(np.isfinite(np.unique(var_years_mode))) != len(rank_years):
    #     print(f'len(var_years_mode): {len(var_years_mode)} != len(rank_years): {len(rank_years)}')

    # get var1 and var2 data
    var1_data = get_var_data(var1, region=var1_region, level=kwargs.get('var1_level', None))
    var2_data = get_var_data(var2, region=var2_region, level=kwargs.get('var2_level', None))

    # get var1 and var2 monthly data in years corresponding to var_years_mode
    var1_data_yearly = extract_var_yearly(var1, var1_data, var_years_mode, var2_months)
    var2_data_yearly = extract_var_yearly(var2, var2_data, var_years_mode, var2_months)

    # compute stats
    var2_min = var2_data_yearly.min().compute()
    var2_max = var2_data_yearly.max().compute()
    cf_levels = np.linspace(var2_min, var2_max, 50)
    cf_norm = plt.Normalize(var2_min, var2_max)
    cf_cmap = 'turbo'

    # plot results
    for year, rank in zip(var_years_mode, rank_years):
        fn_core, out_fn = get_out_fn(var1, var1_months, var1_region, var2, var2_months, var2_region, year, rank, **kwargs)
        out_fp = f'{plot_out_path}spatial/dsw/{out_fn}'
        if os.path.exists(out_fp):
            return

        projection = ccrs.PlateCarree()
        nrows = math.ceil(len(var2_months)/4)
        ncols = 4 if len(var2_months) >= 4 else len(var2_months)
        fig_length_scale = 4 if len(var2_months) >= 4 else len(var2_months)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8*fig_length_scale,10), subplot_kw=dict(projection=projection))
        axes = axes.flatten()

        for i, month in enumerate(var2_months):
            # get month name from my own function
            month_name = month_num_to_name(var2, month)
            if 'month' in var1_data_yearly.dims:
                var1_ax_text = f'{var1_data_yearly.sel(year=year, month=month).values:.4f}'
                var1_fig_text = ''
            else:
                var1_ax_text = ''
                var1_fig_text = f'{var1_data_yearly.sel(year=year).values:.4f}'
            var2_data_monthly = var2_data_yearly.sel(year=year, month=month)
            cf = axes[i].contourf(var2_data_monthly.longitude, var2_data_monthly.latitude,
                                  var2_data_monthly, levels=cf_levels, norm=cf_norm,
                                  cmap=cf_cmap, extend='both')
            axes[i].set_title(month_name)
            axes[i].text(0.5, -0.05, f'{var_dict[var1]}\n{var1_ax_text} {var_units[var1]}', ha='center', va='center', transform=axes[i].transAxes)

            # add coastlines, state borders, and other features
            axes[i].coastlines(linewidth=0.5)
            axes[i].add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
            axes[i].add_feature(cfeature.STATES, linewidth=0.5)

        for j in range(len(var2_months), len(axes)):
            fig.delaxes(axes[j])
        if var1_region in region_avg_list:
            var1_region_title = region_avg_dict[var1_region]
        else:
            var1_region_title = var1_region
        fig.suptitle(f'{var_dict[var2]} ({year})\nRank {rank} for {month_num_to_name(var1, var1_months)} {var1_region_title}\n{var_dict[var1]}\n{var1_fig_text} {var_units[var1]}',
                     fontsize=18)
        plt.tight_layout(rect=[0.05, 0.15, 0.95, 0.9], pad=0.05)
        cbar_ax = fig.add_axes([0.15, 0.01, 0.7, 0.04])
        fig.colorbar(cf, cax=cbar_ax, orientation='horizontal', label=f'{var_units[var2]}')
        plt.savefig(out_fp, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close()


# In[66]:


def get_out_fn(var1, var1_months, var1_region, var2, var2_months, var2_region, year, rank, **kwargs):
    var1_level = kwargs.get('var1_level', '')
    var2_level = kwargs.get('var2_level', '')

    var1_month_names = month_num_to_name(var1, var1_months)
    var2_month_names = month_num_to_name(var2, var2_months)

    # create core of output file name
    fn_list = [str(var1), str(var1_level), str(var1_month_names), str(var1_region),
               'rank', str(rank), str(year),
               str(var2), str(var2_level), str(var2_month_names), str(var2_region)]
    fn_core = '_'.join([i for i in fn_list if i != ''])

    out_fn_png = f'ranked_{fn_core}.png'

    return fn_core, out_fn_png


# In[3]:


def get_sorted_years(var, months, region, **kwargs):
    """
    Sort the order of var1 data. e.g., most -> least precipitation in MJJASO at some lat/lon combo = (1981, 1990, ...2004).

    Parameters
    ----------
    var : str
        The variable used to do the sorting of years.
    months : list, int
        The months to consider when sorting.
    region : str
        The region over which to average data

    Returns
    -------
    numpy array
        Array of years sorted in descending order of var data.
    """
    sorted_years = order_years(var, months, region)  # years sorted in descending order by var sum or mean
    return sorted_years


# In[4]:


def calc_mode(arr, mode_years, **kwargs):

    # calculate mode
    mode, counts = np.unique(arr.flatten(), return_counts=True)
    sorted_counts = np.argsort(-counts)
    for count in sorted_counts:
        if mode[count] not in mode_years:
            return mode[count]
    else:
        return np.nan


def apply_region_mode(da, **kwargs):

    mode_years = []
    for year in da.sorted_years.values:
        mode = calc_mode(da.sel(sorted_years = year).values, mode_years)
        mode_years.append(mode)

    return np.array(mode_years)


# In[5]:


def subset_lat_lon(da, region, **kwargs):

    if not {'latitude', 'longitude'}.issubset(da.dims):
        return da  # only subset by lat/lon if those dimensions exist
    else:
        # subset by latitude and longitude
        if region in region_avg_list:
            coords = region_avg_coords[region]
        else:
            coords = kwargs.get('coords', [240, 260, 40, 20])  # default to whole dsw
        lats = slice(coords[2], coords[3])  # (North, South)
        lons = slice(coords[0], coords[1])  # (West, East)

        return da.sel(latitude=lats, longitude=lons)


# In[6]:


def extract_var_yearly(var, da, years, months, **kwargs):

    if var in NAM_var_list:
        return da.sel(year=years)
    # calc monthly means dims=(year, month, ...)
    return time_to_year_month(var, da).sel(year=years, month=months)


# In[ ]:


# # cell to call main() and plot data
# var1 = 'precipitation'
# var1_months = [6, 7, 8, 9]
# var1_region = 'cp'

# var2 = 'z_height'
# var2_months = [6, 7, 8]
# var2_region = 'dsw'

# # rank_years = [1, 2, 3, 4, 5, 36, 37, 38, 39, 40]
# rank_years = [1]

# var_kwargs = {
#     'var1_level': 500,
#     'var2_level': 500,
# }

# main_kwargs = {

# }

# if __name__ == '__main__':
#     kwargs = main_kwargs.copy()
#     if var1 in pl_var_list:
#         kwargs.update({'var1_level': var_kwargs['var1_level']})
#     if var2 in pl_var_list:
#         kwargs.update({'var2_level': var_kwargs['var2_level']})

#     main(var1, var1_months, var1_region, var2, var2_months, var2_region, rank_years, **kwargs)


# In[ ]:


# cell to call main() and plot data
var1 = 'precipitation'
var1_months = [6, 7, 8, 9]
var1_region_list = ['cp', 'mr', 'MeNmAz', 'son', 'moj', 'chi', 'dsw']

var2 = 'z_height'
var2_months = [6, 7, 8]
var2_region = 'dsw'

rank_years = [1, 2, 3, 4, 5, 36, 37, 38, 39, 40]

var_kwargs = {
    'var1_level': 500,
    'var2_level': 500,
}

main_kwargs = {

}
for var1_region in var1_region_list:
    if __name__ == '__main__':
        kwargs = main_kwargs.copy()
        if var1_region == 'dsw':
            kwargs.update({'dim_means': ['latitude', 'longitude']})
        if var1 in pl_var_list:
            kwargs.update({'var1_level': var_kwargs['var1_level']})
        if var2 in pl_var_list:
            kwargs.update({'var2_level': var_kwargs['var2_level']})

        main(var1, var1_months, var1_region, var2, var2_months, var2_region, rank_years, **kwargs)


# In[ ]:




