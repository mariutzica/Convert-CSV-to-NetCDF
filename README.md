This repository contains very simple Python script that uses Unidata's netcdf4 library
<http://unidata.github.io/netcdf4-python/> to create a NetCDF4 file for
the given csv column format for tomography grid data. Data is divided up by the third column (depth)--treated as the ‘levels’ dimension. The
longitudes/latitudes do not create a complete grid, so those are kept
as coordinate pairs; dVp data is saved as a variable with dimensions num_levels x num_lonlat_coordinates.
