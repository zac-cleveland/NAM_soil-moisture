"""
This script contains miscillaneous functions that I regulargly use.
"""

# needed modules
import os
import sys
import calendar
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# needed directories

# needed dictionaries and lists
if '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/' not in sys.path:
    sys.path.insert(0, '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/')  # path to file containing these lists/dicts

# import my function to grab my subset data
from get_var_data import (
get_var_data, get_var_files, open_var_data, subset_var_data,
time_to_year_month_avg, time_to_year_month_sum, time_to_year_month
)

# import lists and dictionaries
from my_dictionaries import (
sfc_instan_list, sfc_accumu_list, pl_var_list, derived_var_list, invar_var_list,
NAM_var_list, region_avg_list, flux_var_list, vector_var_list, misc_var_list,
var_dict, var_units, region_avg_dict, region_avg_coords, region_colors_dict
)


# test function
def my_test_function():
    print('updated')


# define a function to turn a list of integers into months
def month_num_to_name(var, months, **kwargs):
    # make sure months is a list
    months = ensure_var_list(months)

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


# define a function to check if inputs are list or not
def ensure_var_list(x):
    if not isinstance(x, list):
        return [x]
    return x


# define a function to plot regional boundaries
def plot_region_boundaries(region, type='subplot', **kwargs):
    if region not in region_avg_list:
        coords = kwargs.get('coords', [240, 260, 40, 20])
        color = kwargs.get('color', 'k')
    else:
        coords = region_avg_coords[region]
        color = kwargs.get('color', region_colors_dict[region])

    lons = [coords[0], coords[1], coords[1], coords[0], coords[0]]
    lats = [coords[2], coords[2], coords[3], coords[3], coords[2]]
    plt.plot(lons, lats, color=color, linewidth=1)