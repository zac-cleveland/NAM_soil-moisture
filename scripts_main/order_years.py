#!/usr/bin/env python
# coding: utf-8


# import functions
import sys
import importlib
import numpy as np
import xarray as xr

# paths to various locations on my NCAR account
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
plot_out_path = '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/'  # path to generated plots
scripts_main_path = '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/'  # path to my dicts, lists, and functions

# import variable lists and dictionaries I've created
if scripts_main_path not in sys.path:
    sys.path.insert(0, scripts_main_path)  # path to file containing these lists/dicts
if 'get_var_data' in sys.modules:
    importlib.reload(sys.modules['get_var_data'])
if 'my_functions' in sys.modules:
    importlib.reload(sys.modules['my_functions'])
if 'my_dictionaries' in sys.modules:
    importlib.reload(sys.modules['my_dictionaries'])

# import common functions that I've created
from get_var_data import get_var_data, get_var_files, open_var_data, subset_var_data, time_to_year_month_avg, time_to_year_month_sum


# import lists and dictionaries
import my_dictionaries

# my lists
sfc_accumu_list = my_dictionaries.sfc_accumu_list  # accumulated surface variables
NAM_var_list = my_dictionaries.NAM_var_list  # NAM-based variables

# my dictionaries
var_dict = my_dictionaries.var_dict  # variables and their names
var_units = my_dictionaries.var_units  # variable units
region_avg_dict = my_dictionaries.region_avg_dict  # region IDs and names
region_avg_coords = my_dictionaries.region_avg_coords  # coordinates for regions
region_colors_dict = my_dictionaries.region_colors_dict  # colors to plot for each region


def calc_mean_sum_yearly(var, da, **kwargs):
    """
    Calculate the mean for accumulated variables and sum for all else over the
    month dimension. If month is not a dimension, don't calculate it.

    Parameters
    ----------
    var : str
        The variable being operated on.
    da : xarray.Dataset or xarray.DataArray
        The data being operated on.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        The monthly mean or sum of the data.
    """
    if 'month' not in da.dims:
        return da  # only execute this code if month is actually a dimension

    if var in sfc_accumu_list:  # sum accumulated variables
        return da.groupby('year').sum('month')
    else:  # average all other variables
        return da.groupby('year').mean('month')


def sort_years_descending(arr_data, arr_years, **kwargs):
    """
    Order the years of some data in descending order of the data value.

    Parameters
    ----------
    arr_data : numpy array
        Input array from xarray.apply_ufunc() of data values. e.g., precipitation.

    arr_years : numpy array
        Years corresponding to the data values.

    Returns
    -------
    Sorted numpy array in descending order
    """
    sorted_data_indices = np.argsort(arr_data)[::-1]  # index of sorted data in descending order
    sort_years_descending = arr_years[sorted_data_indices]  # arr_years sorted in descending order
    return sort_years_descending


def apply_sort_years(var, da, **kwargs):
    """
    Order years by total of variable for certain months.

    Parameters
    ----------
    var : str
        The variable that is being sorted. e.g., 'tp', 'sshf', 'sd', ...
    da : xr.DataArray
        Monthly means of given variable with dimensions (year, month, ...).

    Returns
    -------
    list
        Years ordered by total of variable for certain months from most to least.
    """
    years = da.year

    sorted_years = xr.apply_ufunc(
        sort_years_descending, da, years,
        input_core_dims=[['year'], ['year']],
        output_core_dims=[['sorted_years']],
        vectorize=True,
        dask='parallelized',
        dask_gufunc_kwargs={'allow_rechunk': True,
                           'output_sizes': {'sorted_years': len(years)}
                           },
        output_dtypes=[int],
    )
    return sorted_years.assign_coords({'sorted_years': np.arange(1, len(years)+1)})


def order_years(var, months, region, var_data=None, **kwargs):
    """
    Determines the years, in descending order, of data for a given variable within a specified
    range of months. e.g., determine which years had the most to least precipitation in MJJASO
    at each grid point.

    Parameters
    ----------
    var : str
        The variable to be sorted.
    months : list, int
        List of months to consider. e.g., [3, 4, 5] for March, April, May
    region : str
        The region over which to average data
    var_data : xarray.Dataset or xarray.DataArray, default: None
        If var_data is passed into the function, skip opening it and just pass da along.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
        Dimensions (sorted_years, ...) in descending order.
    """
    if var_data is not None and isinstance(var_data, xr.DataArray):
        # get var data
        var_data = get_var_data(var, region=region, **kwargs)

    if var not in NAM_var_list:
        # get monthly means or sums dims=(year, month, ...)
        if var in sfc_accumu_list:
            var_monthly = time_to_year_month_sum(var_data)  # monthly sums
        else:
            var_monthly = time_to_year_month_avg(var_data)  # monthly averages

        # average or sum data dims=(year, ...)
        var_yearly = calc_mean_sum_yearly(var, var_monthly.sel(month=months))
    else:
        var_yearly = var_data  # NAM_var_list contains data with dims=(year, latitude, longitude)

    # get array of ordered years
    return apply_sort_years(var, var_yearly)
