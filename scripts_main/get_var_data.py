"""
This script contains functions to retrieve data for a given variable and return it as an xarray dataset or data array.
"""

# modules needed
import os
import sys
import glob
import numpy as np
import xarray as xr
import pandas as pd

# needed directories
my_era5_path = '/glade/u/home/zcleveland/scratch/ERA5/'  # path to subset ERA5 data
misc_data_path = '/glade/u/home/zcleveland/scratch/misc_data/'  # path to misc data
my_esa_path = '/glade/u/home/zcleveland/scratch/ESA_data/'  # path to subset ESA data

# needed dictionaries and lists
if '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/' not in sys.path:
    sys.path.insert(0, '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/')  # path to file containing these lists/dicts
import my_dictionaries

# needed lists
sfc_instan_list = my_dictionaries.sfc_instan_list  # instantaneous surface variables
sfc_accumu_list = my_dictionaries.sfc_accumu_list  # accumulated surface variables
pl_var_list = my_dictionaries.pl_var_list  # pressure level variables
invar_var_list = my_dictionaries.invar_var_list  # invariant variables
NAM_var_list = my_dictionaries.NAM_var_list  # NAM-based variables
region_avg_list = my_dictionaries.region_avg_list  # region IDs for regional averages
misc_var_list = my_dictionaries.misc_var_list  # misc variables

# needed dictionaries
var_dict = my_dictionaries.var_dict  # variables and their names
region_avg_coords = my_dictionaries.region_avg_coords  # coordinates for regions


def get_var_data(var, region='dsw', months=[i for i in range(1,13)], **kwargs):
    """
    Retrieves the data for a given variable from my subet ERA5 dataset.  User can choose to return a dataset or data array
    and whether to subset that data based on a region or time.  Any subset data is returned as a data array.

    Parameters
    ----------
    var : str
            The variable desired
    region : str, optional
            The region desired
    months : list of int, optional
            A list of months desired [1, 2, ..., 12]

    Kwargs
    ------
    subset_flag : bool, optional
            True or False.  Whether to subset the data or not
    var_type : str, optional
            Specify whether to return a dataset or data array
    level : int, optional
            The pressure level desired.  Only applied for pressure level data
    coords : list of int, optional
            [west, east, north, south] longitude and latitude coordinates to subset the data.

    Returns
    -------
    xarray.DataArray
            A data array containing the desired data, either in full or subset based on user input

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
    if kwargs.get('subset_flag', True):
        return subset_var_data(var_data, var, region, months, **kwargs)
    return var_data


def get_var_files(var, region, **kwargs):
    """
    Retrieves files for the given variable and region.

    Parameters
    ----------
    var : str
        The desired variable.
    region : str
        The desired region.

    Returns
    -------
    list
        Sorted list of file paths.
    """

    path_map = {
        'sfc': sfc_instan_list + sfc_accumu_list,
        'pl': pl_var_list,
        'NAM': NAM_var_list,
        'misc': misc_var_list,
        'invar': invar_var_list
    }
    
    if var in path_map['sfc']:
        pattern = f'{my_era5_path}dsw/*/{var.lower()}_*_dsw.nc' if region != 'global' else f'{my_era5_path}global/*/{var.lower()}_*_dsw.nc'
    elif var in path_map['pl']:
        pattern = f'{my_era5_path}dsw/*/pl/{var.lower()}_*_dsw.nc'
    elif var in path_map['NAM']:
        pattern = f'{my_era5_path}dsw/NAM_{var}.nc'
    elif var in path_map['misc']:
        pattern = f'{misc_data_path}{var}/{var}*.nc'
    elif var in path_map['invar']:
        pattern = f'{my_era5_path}invariants/{var}_invariant.nc'
    else:
        return []

    files = glob.glob(pattern)
    files.sort()
    return files


def open_var_data(files, var, **kwargs):
    """
    Opens datasets for the given variable.

    Parameters
    ----------
    files : list of str
        List of full file paths.
    var : str
        The desired variable.

    Keyword Args
    ------------
    var_type : str, optional
        Specify whether to return a dataset ('ds') or data array ('da'). Defaults to 'da'.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Data array or dataset containing the variable data.
    """
    var_type = kwargs.get('var_type', 'da')  # default to returning a data array
    ds = xr.open_mfdataset(files)

    if var_type == 'ds':  # return dataset if specified
        return ds

    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v.upper()][0]

    # if var is onset or retreat, convert to day of year
    if var.lower() in ['onset', 'retreat']:
        return ds[var_name].dt.dayofyear  # convert to dayofyear (datetime -> integer)

    return ds[var_name]


def subset_var_data(var_data, var, region, months, **kwargs):
    """
    Subsets the input data by latitude/longitude, time, and averages. Defaults to returning full dataset
    for region = dsw or global, and the latitude/longitude mean if region is in region_avg_list.  If time
    is specified in dim_means, time_group must be specified. If level is specified in dim_means, level must
    be input as a list of levels in kwargs.

    Parameters
    ----------
    var_data : xarray.DataArray or xarray.Dataset
        Input data.
    var : str
        The desired variable.
    months : list of int
        List of months.
    region : str
        The desired region.

    Keyword Args
    ------------
    level : int, optional
        The pressure level desired. Only applied for pressure level data.
    dim_means : list of str, optional
        Dimensions to average over, e.g., ['time', 'latitude', 'longitude', 'level'].

    Returns
    -------
    xarray.DataArray
        Subsetted data array.
    """
    # subset to level if var is a pl var
    if var.lower() in pl_var_list:
        level = kwargs.get('level', var_data.level)  # default to returning all levels
        var_data = var_data.sel(level=level)

    # subset by latitude and longitude
    if region in region_avg_list and {'latitude', 'longitude'}.issubset(var_data.dims):
        coords = region_avg_coords[region]
        dim_means = kwargs.get('dim_means', ['latitude', 'longitude'])
    else:
        coords = kwargs.get('coords', [240, 260, 40, 20])  # default to whole dsw
        dim_means = kwargs.get('dim_means', [])
    lats = slice(coords[2], coords[3])  # (North, South)
    lons = slice(coords[0], coords[1])  # (West, East)

    # only subset if latitude and longitude in var_data
    if {'latitude', 'longitude'}.issubset(var_data.dims):
        var_data = var_data.sel(latitude=lats, longitude=lons)

    # remove latitude and longitude from dim_means if they don't actually exist as dimensions
    if not {'latitude', 'longitude'}.issubset(var_data.dims):
        dim_means = [dim for dim in dim_means if dim not in ['latitude', 'longitude']]
        
    if not dim_means:
        return var_data
    return var_data.mean(dim=dim_means, skipna=True)


def time_to_year_month_avg(ds, **kwargs):
    """
    Converts an xarray.Dataset or xarray.DataArray time dimension from time to year,month with monthly averages.
    e.g., (time: 14610), where time represents daily values over 40 years will be converted to (year:40, month:12).

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
            The dataset or data array to be manipulated.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
            the monthly averaged dataset or data array.
    """
    years = np.unique(ds.time.dt.year)
    months = np.unique(ds.time.dt.month)

    # make a pandas MultiIndex that is years x months
    midx = pd.MultiIndex.from_product([years, months], names=("year","month"))
    ds_temp = ds.resample(time='1M').mean(dim='time', skipna=True)

    return ds_temp.assign_coords({'time':midx}).unstack()


def time_to_year_month_sum(ds, **kwargs):
    """
    Converts an xarray.Dataset or xarray.DataArray time dimension from time to year, month with monthly sum.
    e.g., (time: 14610), where time represents daily values over 40 years will be converted to (year:40, month:12).

    Parameters
    ----------
    ds : xarray.Dataset or xarray.DataArray
            The dataset or data array to be manipulated.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
            the monthly summed dataset or data array.
    """
    years = np.unique(ds.time.dt.year)
    months = np.unique(ds.time.dt.month)

    # make a pandas MultiIndex that is years x months
    midx = pd.MultiIndex.from_product([years, months], names=("year","month"))
    ds_temp = ds.resample(time='1M').sum(dim='time', skipna=True)

    return ds_temp.assign_coords({'time':midx}).unstack()


def time_to_year_month(var, ds, **kwargs):
    """
    Converts an xarray.Dataset or xarray.DataArray time dimension from time to year, month with monthly sum or mean based on
    var type. e.g., (time: 14610), where time represents daily values over 40 years will be converted to (year:40, month:12).
    SFC accumulation variables are summed and all others are averaged.

    Parameters
    ----------
    var : str
        The variable in question.
    ds : xarray.Dataset or xarray.DataArray
            The dataset or data array to be manipulated.

    Returns
    -------
    xarray.Dataset or xarray.DataArray
            the monthly summed or averaged dataset or data array.
    """
    if var in NAM_var_list:
        return ds
    if 'time' not in ds.dims:
        return ds
    years = np.unique(ds.time.dt.year)
    months = np.unique(ds.time.dt.month)

    # make a pandas MultiIndex that is years x months
    midx = pd.MultiIndex.from_product([years, months], names=("year","month"))
    if var in sfc_accumu_list:
        ds_temp = ds.resample(time='1M').sum(dim='time', skipna=True)
    else:
        ds_temp = ds.resample(time='1M').mean(dim='time', skipna=True)

    return ds_temp.assign_coords({'time':midx}).unstack()