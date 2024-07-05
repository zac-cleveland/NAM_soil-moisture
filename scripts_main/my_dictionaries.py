"""
This code contains the lists and dictionaries for variables used in the NAM_soil-moisture analysis project.
"""

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
    'cin',  # convective inhibition (J kg^-1)
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

# invariant data
invar_var_list = [
    'cl',  # lake cover (0-1)
    'dl',  # lake depth (m)
    'cvl',  # low vegetation cover (0-1)
    'cvh',  # high vegetation cover (0-1)
    'tvl',  # type of low vegetation ~
    'tvh',  # type of high begetation ~
    'slt',  # soil type ~
    'sdfor',  # standard deviation of filtered subgrid orography (m)
    'z_sfc',  # geopotential of surface (m^2 s^-2)
    'sdor',  # standard deviation of orography ~
    'isor',  # anisotropy of subgridscale orography ~
    'anor',  # angle of subgridscale orography (radians)
    'slor',  # slope of subgridscale orography ~
    'lsm',  # land-sea mask (0-1)
    'elevation',  # elevation of terrain (m)
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
    'cin': 'Convective Inhibition',
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
    'nino-3': r'Nino-3 Index',
}

# variable units in latex format for plotting
var_units = {
    'sd': r'(m of water equivalent)',
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
    'cin': r'$(J kg^{-1})$',
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
    'sf': r'(m of water equivalent)',
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
    'nino-3': r'(Nino-3 Index Anomaly)',
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















