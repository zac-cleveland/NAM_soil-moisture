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


def get_var_data(var, region='WestUS_Mexico', time_type='monthly', months=[i for i in range(1,13)], **kwargs):
    """
    Retrieves the data for a given variable from my subet ERA5 dataset.  User can choose to return a dataset or data array
    and whether to subset that data based on a region or time.  Any subset data is returned as a data array.

    Parameters
    ----------
    var : str
            The variable desired. List of options available in my_dictionaries.
    region : str, optional, default: 'WestUS_Mexico'
            The region desired. List of options available in my_dictionaries.
    time_type : str, optional, default: 'monthly'.
            Whether to return daily or monthly data.
    months : list of int, optional, default: [1, ..., 12]
            A list of months desired [1, 2, ..., 12].

    Kwargs
    ------
    subset_flag : bool
            True or False. Whether to subset the data or not. Defaults to True.
    var_type : str
            Specify whether to return a dataset (ds) or data array (da).
    level : int
            The pressure level desired [1000, ..., ].  Only applied for pressure level data.
    coords : list of int
            [west, east, north, south] longitude and latitude coordinates to subset the data. Note ERA5 data is on a 0-360 longitude grid.

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

    files = get_var_files(var, region, time_type, **kwargs)
    var_data = open_var_data(files, var, **kwargs)
    if kwargs.get('subset_flag', True):
        return subset_var_data(var_data, var, region, months, **kwargs)
    return var_data


def get_var_files(var, region, time_type='monthly', **kwargs):
    """
    Retrieves files for the given variable and region.

    Returns sorted list of file paths.
    """

    var_map = {  # help map to directories based on var type
        'base': sfc_instan_list + sfc_accumu_list + pl_var_list,
        'NAM': NAM_var_list,
        'misc': misc_var_list,
        'invar': invar_var_list,
    }    

    # create file path based on pattern of files for var, region, and time_type
    data_dir = 'dsw' if region == 'dsw' else 'WestUS_Mexico'  # for regional averages, the data is averaged later on
    if var in var_map['base']:
        if time_type == 'monthly':
            year_dir = ''
            time_str = '????_????'
        else:
            year_dir = '*'
            time_str = '??????' if var in pl_var_list else '??????_??????'
        pattern = os.path.join(my_era5_path, data_dir, year_dir, f'{var}_{time_str}_{data_dir}.nc')
    elif var in var_map['NAM']:
        pattern = os.path.join(my_era5_path, data_dir, f'NAM_{var}.nc')
    elif var in var_map['misc']:
        pattern = os.path.join(misc_data_path, var, f'{var}*.nc')
    elif var in var_map['invar']:
        pattern = os.path.join(my_era5_path, 'invariants', f'{var}_invariant.nc')
    elif var.lower() == 'ESA_sm'.lower():
        pattern = os.path.join(my_esa_path, 'global', f'{var}_*_dsw.nc')
    else:
        return []

    files = sorted(glob.glob(pattern))
    return files


def open_var_data(files, var, **kwargs):
    """
    Opens datasets for the given variable.

    Returns Data Array or Dataset containing the variable data.
    """
    var_type = kwargs.get('var_type', 'da')  # default to returning a data array
    ds = xr.open_mfdataset(files, parallel=True)

    if var_type == 'ds':  # return dataset if specified
        return ds
    # pull out actual variable name in the dataset since they can be different names/capitalized
    var_name = [v for v in ds.data_vars.keys() if f'{var.upper()}' in v.upper()][0]

    # if var is onset or retreat, return the day of year (datetime -> integer)
    if var.lower() in ['onset', 'retreat']:
        return ds[var_name].dt.dayofyear
    return ds[var_name]


def subset_var_data(var_data, var, region, months, **kwargs):
    """
    Subsets the input data 

    Returns subsetted data array.
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
        if region != 'global':
            if region == 'WestUS_Mexico':
                coords = kwargs.get('coords', [230, 270, 50, 10])  # default to West US and Mexico region
            else:
                coords = kwargs.get('coords', [240, 260, 40, 20])  # default to whole dsw
        else:
            coords = kwargs.get('coords', [0, 360, 90, -90])  # dafault to global
        dim_means = kwargs.get('dim_means', [])
    lats = slice(coords[2], coords[3])  # (North, South)
    lons = slice(coords[0], coords[1])  # (West, East)

    # only subset if latitude and longitude in var_data
    if {'latitude', 'longitude'}.issubset(var_data.dims):
        var_data = var_data.sel(latitude=lats, longitude=lons)

    # remove latitude and longitude from dim_means if they don't actually exist as dimensions
    if not {'latitude', 'longitude'}.issubset(var_data.dims):
        dim_means = [dim for dim in dim_means if dim not in ['latitude', 'longitude']]

    # mask out terrain by elevation
    elevation_value = kwargs.get('elevation_mask', None)
    if elevation_value:
        elevation = xr.open_dataset(os.path.join(my_era5_path, 'invariants/elevation_invariant.nc'))['elevation'].sel(latitude=lats, longitude=lons)
        mask = elevation > elevation_value
        var_data = var_data.where(mask, other=np.nan)
        
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
    if var in NAM_var_list or 'time' not in ds.dims:
        return ds
    years = np.unique(ds.time.dt.year)
    months = np.unique(ds.time.dt.month)

    # make a pandas MultiIndex that is years x months
    midx = pd.MultiIndex.from_product([years, months], names=("year","month"))
    midx_coords = xr.Coordinates.from_pandas_multiindex(midx, 'time')
    ds_temp = ds.resample(time='1ME').mean(dim='time', skipna=True)

    return ds_temp.assign_coords(midx_coords).unstack()


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
    if var in NAM_var_list or 'time' not in ds.dims:
        return ds
    years = np.unique(ds.time.dt.year)
    months = np.unique(ds.time.dt.month)

    # make a pandas MultiIndex that is years x months
    midx = pd.MultiIndex.from_product([years, months], names=("year","month"))
    midx_coords = xr.Coordinates.from_pandas_multiindex(midx, 'time')
    ds_temp = ds.resample(time='1ME').sum(dim='time', skipna=True)

    return ds_temp.assign_coords(midx_coords).unstack()


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
    if var in NAM_var_list or 'time' not in ds.dims:
        return ds

    years = np.unique(ds.time.dt.year)
    months = np.unique(ds.time.dt.month)

    # make a pandas MultiIndex that is years x months
    midx = pd.MultiIndex.from_product([years, months], names=("year","month"))
    midx_coords = xr.Coordinates.from_pandas_multiindex(midx, 'time')
    ds_res = ds.resample(time='1ME')
    ds_out = ds_res.sum(dim='time', skipna=True) if var in sfc_accumu_list else ds_res.mean(dim='time', skipna=True)
    ds_out = ds_out.assign_coords(midx_coords).unstack()
    return ds_out