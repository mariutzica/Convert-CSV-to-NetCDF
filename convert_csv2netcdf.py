#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 16:55:32 2018

@author: mariutzica
@description: Script to convert csv to NETCDF4
"""

#pandas for reading in csv
import pandas as pd
import numpy as np

#netcdf4 for creating NETCDF4 data group
from netCDF4 import Dataset
import time

# read in csv data, sort by long, lat
data = pd.read_csv('jgrb51645-sup-0002-supplementary.csv')
cols = data.columns.values
data = data.sort_values(by=[cols[0],cols[1]])
# determine the number of levels from third column
num_levels = len(np.unique(data[cols[2]]))
lonlat = data[[cols[0],cols[1]]].drop_duplicates().reset_index(drop=True)
num_lonlat = len(data[[cols[0],cols[1]]].drop_duplicates())
#cleck if data is evenly divided by levels
#print(num_levels*num_lonlat == len(data))    

# create dataset
rootgrp = Dataset("dvp.nc", "w", format="NETCDF4")

# create dimensions; lon and lat are the same dimension and create coordinate pairs
level = rootgrp.createDimension("level", num_levels)
lat = rootgrp.createDimension("lat", num_lonlat)
lon = rootgrp.createDimension("lon", num_lonlat)

# create variables for levels, lat, lon, and dvp
levels = rootgrp.createVariable("level","i4",("level",))
latitudes = rootgrp.createVariable("lat","f4",("lat",))
longitudes = rootgrp.createVariable("lon","f4",("lon",))
dvp = rootgrp.createVariable("dvp","f4",("level","lat",))

# store unit attributes
latitudes.units = 'degrees north'
longitudes.units = 'degrees east'
levels.units = 'km'
dvp.units = '%'
rootgrp.description = "example script convert csv to netcdf"
rootgrp.history = "Created " + time.ctime(time.time())

#assign values to variables
levels[:] = np.unique(data[cols[2]])
latitudes[:] = lonlat[cols[0]].tolist()
longitudes[:] = lonlat[cols[1]].tolist()
# reshape dvp variable into [level x lonlat_coordinate] array
i = 0
for n in np.unique(data[cols[2]]):
    #check if lonlat data is the same for all levels
#    dvp_lonlat = data.loc[data[cols[2]]==n,[cols[0],cols[1]]]
#    if dvp_lonlat.reset_index(drop=True).equals(lonlat):
#        print('Equal.')
#    else:
#        print(dvp_lonlat[cols[0]],lonlat[cols[0]])
    dvp[i,:]=data.loc[data[cols[2]]==n,cols[3]].tolist()
    i+=1


rootgrp.close()