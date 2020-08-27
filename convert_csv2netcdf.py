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
#from scipy.interpolate import interp2d
from scipy.interpolate import griddata
import math
#import pyproj

#netcdf4 for creating NETCDF4 data group
from netCDF4 import Dataset
import time

import sys
import matplotlib.pyplot as plt
from matplotlib import cm
#deprecated
#from matplotlib.mlab import griddata
from numpy import matlib as mb

def main(argv):
    if len(argv)==2:
        input_file = argv[0]
        output_file = argv[1]
        wordy = False
    else:
        print('Usage -- convert_csv2netcdf.py inputfilename outputfilename')
        sys.exit(0)
    
    # read in csv data, sort by long, lat
    data = pd.read_csv(input_file)
    cols = data.columns.values
    data = data.sort_values(by=[cols[2],cols[0],cols[1]])
    
    #lambert_azimuthal_eq_area=pyproj.Proj("+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
    #wgs84=pyproj.Proj("+init=EPSG:4326")
    
    # determine the number of levels from third column
    num_levels = len(np.unique(data[cols[2]]))
    level_vals = list(np.unique(data[cols[2]]))
    if wordy:
        print('Number of levels is ',num_levels)
    
    # get unique pairs of lon, lat coordinates
    lonlat = data[[cols[0],cols[1]]].drop_duplicates().reset_index(drop=True)

    # plot a scatter of the lon, lat coordinates
    #plt.scatter(lonlat[cols[0]],lonlat[cols[1]])
    
#    num_lonlat = len(data[[cols[0],cols[1]]].drop_duplicates())
#    num_lon = len(np.unique(data[cols[0]]))
#    num_lat = len(np.unique(data[cols[1]]))
    
    #check if data is evenly divided by levels
    # If any of these values are true then levels are not similarly laid out
    same = True
    if wordy:
        print('Checking if levels have similar lonlat grids.')
    for level in level_vals:
        level_df_lonlat = data.loc[data[cols[2]]==level]
        level_df_lonlat = level_df_lonlat[[cols[0],cols[1]]].reset_index(drop=True)
        is_equal = ~((level_df_lonlat != lonlat).any(1)).any(0)
        same = same and is_equal
        if wordy:
            print('Level = '+str(level)+', lon lat are '+('' if is_equal else 'not ')+'the same.')
    
    if not same:
        print('Can\'t create even grids.')
        return lonlat

    dvp = []
    hq = []
    for level in level_vals:
        lonlat = data.loc[data[cols[2]]==level,[cols[0],cols[1],cols[3],cols[4]]]
        #check to see if wrap around is necessary and create regular grid
        #current default spacing = .05 degrees
        # ASSUMPTION: if map wraps at 180, it does not wrap at 0, so coordinates can be shifted
        lon_min = lonlat[cols[0]].min()
        lon_max = lonlat[cols[0]].max()
        lat_min = lonlat[cols[1]].min()
        lat_max = lonlat[cols[1]].max()
        wrap = False
        if lon_min < -175 or lon_max > 175:
            wrap = True
            lon_shift_left = 180+lonlat.loc[lonlat[cols[0]]<0,cols[0]].max()
            lon_min = lonlat.loc[lonlat[cols[0]]>0,cols[0]].min()-lon_shift_left
            lon_max = lonlat.loc[lonlat[cols[0]]<0,cols[0]].max() + 360 - lon_shift_left
            
        # plot a scatter of the lon, lat coordinates
        #plt.scatter(lonlat[cols[0]],lonlat[cols[1]]) 
    
        # use pyproj to map the inputted coordinates to cartesian, store in lonlat
        for i in lonlat.index:
            lon = lonlat.loc[i,cols[0]]
            lat = lonlat.loc[i,cols[1]]
            if wrap:
                if lon < 0:
                    lon = lon + 360 
                lon -= lon_shift_left

            #x,y= pyproj.transform(wgs84, lambert_azimuthal_eq_area, lon, lat)
            #        lonlat.loc[i,'proj_x'] = x
            #        lonlat.loc[i,'proj_y'] = y
            lonlat.loc[i,'x'] = lon
            lonlat.loc[i,'y'] = lat

        # create regular grid -- assumption that it is approx square .. change this!
        spacing = math.floor(len(lonlat)**.5)*2
        #print((lon_max - lon_min)/spacing)
        lon_reg = np.linspace(lon_min, lon_max, num = spacing)
        lat_reg = np.linspace(lat_min, lat_max, num = spacing)
        x = lon_reg
        y = lat_reg
        x = mb.repmat(x,len(lat_reg),1)
        x = x.T.reshape(1,len(lon_reg)*len(lat_reg))
        x = x[0]
        y = mb.repmat(y,1,len(lon_reg))
        y = y[0]
        
        #calculate mask around data:
        # opt #1 -- based on point density
        #min_dist = []
        distances = []
        for x2, y2 in zip(x, y):
            dist = (((lonlat['x']-x2)**2 + (lonlat['y']-y2)**2 ) ** 0.5).tolist()
            #min_dist.append(min(dist))
            dist.sort()
            distances.append(sum(dist[0:10]))
        
        dist_sorted = distances.copy()
        dist_sorted.sort()
        num_discard = round(len(dist_sorted)*.2)
        dist_cutoff = dist_sorted[-num_discard]
        mask = [1 if d<dist_cutoff else 0 for d in distances]
        mask = [mask[i] if x[i]<(175) else 0 for i in range(len(mask))]
        #plt.hist(dist_sorted)
        #plt.show()
        
        #opt # 2 - get four edge functions ...
        
        #left side function ..
        
        # interp2d doesn't work ...
        #interp2d(data.loc[data[cols[2]]==60,cols[0]], data.loc[data[cols[2]]==60,cols[1]], data.loc[data[cols[2]]==60,cols[3]], kind='linear')

        #need to shift everything before interpolation, then shift back ...
        # linear interpolation of data; switch to scipy version
        lonlat_cols = lonlat.columns.values
        zi = griddata((lonlat['x'], lonlat['y']), lonlat[lonlat_cols[2]], (x, y), method='linear')
        hqi = griddata((lonlat['x'], lonlat['y']), lonlat[lonlat_cols[3]], (x, y), method='linear')
        lon_reg = [ xi+lon_shift_left if (xi+lon_shift_left)<180 else xi-360+lon_shift_left for xi in lon_reg]
        # create z array for plotting
        #print(type(zi))
        z = list(zi)
    
        #create x, y array
    
        #x = [xi+lon_shift_left if (xi+lon_shift_left)<180 else xi-360+lon_shift_left for xi in x]
        z = [z[i] if mask[i] == 1 else float('NaN') for i in range(len(z))]
        print('Level ',level,' successfully interpolated. Original data(left) and interpolated data(right):')
        #plot original data
        plt.figure(1,figsize=(12,4))
        plt.subplot(121)
        plt.xlim(140,180);plt.scatter(lonlat[lonlat_cols[0]],lonlat[lonlat_cols[1]],s=20,c=lonlat[lonlat_cols[2]], marker = 'o', cmap = cm.jet )
#        
#        n=0
#        for k in z:
#            if k != k:
#                n +=1
#        print(n)
#        
        #plot interpolated data
        plt.subplot(122)
        plt.xlim(140,180);plt.scatter(x,y,s=20,c=z, marker = 'o', cmap = cm.jet )
        plt.show()

        
#        
        #time.sleep(4)
        #lon_reg = []
        #lat_reg = []
    #    for i in lon_val:
    #        for j in lat_val:
    #            #x,y=pyproj.transform(wgs84, lambert_azimuthal_eq_area, i, j)
    #            lon_reg.append(x)
    #            lat_reg.append(y)
        zi = np.array(z)    
        dvp.append(zi.reshape(len(lon_reg),len(lat_reg)))
        
#        plt.imshow(zi.reshape(len(lon_reg),len(lat_reg)));
#        plt.colorbar()
#        plt.show()
        hq.append(hqi.reshape(len(lon_reg),len(lat_reg))) 
    # create dataset
    fill_value = -50000
    rootgrp = Dataset(output_file, "w", format="NETCDF4" )
    
    # create dimensions; lon and lat are the same dimension and create coordinate pairs
    rootgrp.createDimension("level", num_levels)
    rootgrp.createDimension("lat", len(lat_reg))
    rootgrp.createDimension("lon", len(lon_reg))
    
    # create variables for levels, lat, lon, and dvp
    levels = rootgrp.createVariable("level","i4",("level",))
    latitudes = rootgrp.createVariable("lat","f4",("lat",))
    longitudes = rootgrp.createVariable("lon","f4",("lon",))
    dvp_rg = rootgrp.createVariable("dvp","f4",("level","lon","lat"), \
                                 fill_value = fill_value )
    hq_rg = rootgrp.createVariable("hq","i4",("level","lon","lat"), \
                                 fill_value = fill_value )
    
    # store unit attributes
    latitudes.units = 'degrees_north'
    longitudes.units = 'degrees_east'
    levels.units = 'km'
    dvp_rg.units = '%'
    rootgrp.description = "example script convert csv to netcdf"
    rootgrp.history = "Created " + time.ctime(time.time())
    
    #assign values to variables
    z = list(np.unique(data[cols[2]]))
    x = lon_reg
    x = [xi+lon_shift_left if (xi+lon_shift_left)<180 else xi-360+lon_shift_left for xi in x]
    #x = [xi+lon_shift_left for xi in x]
    y = lat_reg
#    print(len(z),len(x),len(y))
#    print(num_levels,num_lat,num_lon)
    #print(num_levels*num_lat, num_levels*num_lon)
    levels[:] = z
    latitudes[:] = y
    longitudes[:] = x
    # reshape dvp variable into [level x lat x lon] array
    for (level,lon,lat), value in np.ndenumerate(dvp):
        dvp_rg[level,lon,lat] = value 
    for (level,lon,lat), value in np.ndenumerate(hq):
        hq_rg[level, lon, lat] = value
    
    rootgrp.close()
    return [dvp, [lon_reg,lat_reg]]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv[1:])
        main(sys.argv[1:])
    else:
        lonlat, lonlat_reg=main(["jgrb51645-sup-0002-supplementary.csv", "output.nc"])
        
