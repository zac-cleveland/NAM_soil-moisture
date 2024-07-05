"""
This script contains miscillaneous functions that I regulargly use.
"""

# needed modules
import os
import sys
import calendar
# needed directories

# needed dictionaries and lists
if '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/' not in sys.path:
    sys.path.insert(0, '/glade/u/home/zcleveland/NAM_soil-moisture/scripts_main/')  # path to file containing these lists/dicts
import my_dictionaries

# needed lists
NAM_var_list = my_dictionaries.NAM_var_list  # NAM-based variables

# needed dictionaries


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


