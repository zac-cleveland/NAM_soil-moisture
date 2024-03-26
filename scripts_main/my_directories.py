"""
This script contains the directories that may be needed.  To save space and make for easier viewing in other scripts, directory paths will be imported from this script, instead of individually specifying each time.
TO UPDATE DIRECTORIES: run the update_directories script located in /glade/u/home/zcleveland/NAM_soil-moisture/scripts_main with specified paths, or manually edit them here.
TO CHECK DIRECTORIES: run the check_directories script located in /glade/u/home/zcleveland/NAM_soil-moisture/scripts_main, which will loop through and check if each directory exists and notify the user if not.
"""

# My directories
my_dirs = {
    'scratch': '/glade/u/home/zcleveland/scratch/',  # my scratch directory
    'work': '/glade/u/home/zcleveland/work/',  # my work directory
    'home': '/glade/u/home/zcleveland/',  # my home directory
    'git_repo': '/glade/u/home/zcleveland/NAM_soil-moisture/',  # base path to my git repo on NCAR HPC
    'era5_analysis': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/',  # ERA5_analysis
    'era5_figures': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/figures/',  # my ERA5 figures (nice)
    'era_plots': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/',  # my ERA5 plots (quick)
    'era5_plots_corr': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/correlations/',  # my ERA5 correlations
    'era5_plots_corr_cp': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/correlations/cp/',  # colorado plateau correlation plots
    'era5_plots_corr_dsw': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/plots/correlations/dsw/',  # desert southwest correlation plots
    'era5_scripts': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/',  # my ERA5 scripts
    'era5_scripts_derived': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/derived/', # derived parameters from era5 data
    'era5_scripts_examples': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/examples', # example/test scripts
    'era5_scripts_figures': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/figures/',  # figures (nice)
    'era5_scripts_plotting': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/plotting/',  # plotting scripts (quick)
    'era5_scripts_subsetting': '/glade/u/home/zcleveland/NAM_soil-moisture/ERA5_analysis/scripts/subsetting/',  # subsetting scripts
    'era5_data': '/glade/u/home/zcleveland/scratch/ERA5/',  # base path to my subset ERA5 data
    'era5_dsw': '/glade/u/home/zcleveland/scratch/ERA5/dsw/',  # my subset ERA5 data over the desert southwest
    'era5_cp': '/glade/u/home/zcleveland/scratch/ERA5/cp/',  # my ERA5 colorado plateau averaged values
    'era5_regrid': '/glade/u/home/zcleveland/scratch/ERA5/regrid-to-esa/',  # my ERA5 data regridded to the ESA dataset 
    'era5_corr': '/glade/u/home/zcleveland/scratch/ERA5/',  # my era5 correlations
    'era5_corr_cp': '/glade/u/home/zcleveland/scratch/ERA5/correlations/cp/',  # my colorado plateau correlations
    'era5_corr_dsw': '/glade/u/home/zcleveland/scratch/ERA5/correlations/dsw/',  # my desert southwest correlations
    'era5_avg_min_max': '/glade/u/home/zcleveland/scratch/ERA5/avg_min_max/',  # pre-combined  ERA5 subset data for parameters with avg, min, and max
    'esa_data': '/glade/u/home/zcleveland/scratch/ESA_data/',  # base path to my subset ESA data
    'esa_dsw': '/glade/u/home/zcleveland/scratch/ESA_data/dsw/',  # ESA data subset to the desert southwest
    'esa_cp': '/glade/u/home/zcleveland/scratch/ESA_data/cp/',  # ESA colorado plateau averaged values
}