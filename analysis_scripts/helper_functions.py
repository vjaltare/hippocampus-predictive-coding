"""
This file contains helper functions for analyzing the mutual information between spiking of hippocampal and entorhinal cortical 
neurons.
The data are taken from CRCNS hc-3 dataset.

"""

import os
import numpy as np
import pandas as pd

#######################################################################################################################

#### Set the data paths ######
BASE_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3"
METADATA_PATH = "/nadata/cnl/data/Vikrant/hc3/hc3-metadata-tables"
SESSIONS_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3/sessions_data"

# set the last directory name --> we'll use os.path.join later
CA3_SESSION_PATH = "ca3_sessions"
CA1_SESSION_PATH = "ca1_sessions"
EC_SESSION_PATH = "ec_sessions"

##################################################################################################################

## read the metadata files in case required
cell_type_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-cell.csv'), header=None, names=['id', 'topdir', 'animal', 'ele', 'clu', 'region', 'ne', 'ni', 'eg', 'ig', 'ec', 'idd', 'fireRate', 'totalFireRate', 'type'])
session_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-session.csv'), header=None, names = ['id', 'topdir', 'session', 'behavior', 'familiarity', 'duration'])
electrode_place_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-epos.csv'), header = None, names = ['topdir', 'animal', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16'])

#################################################################################################################

## define a function to get the track length traveresed by the rat (may be less than the one mentioned in the metadata) in cm

def find_track_length(region, topdir, session):
    """
    OUTPUT:
    1. track_length: float. length of the track travelled by the rat (cm)
    """

    # find the correct region directory
    if region == 'CA1':
        region_path = CA1_SESSION_PATH
    elif region == 'CA3':
        region_path = CA3_SESSION_PATH
    elif region == 'EC':
        region_path = EC_SESSION_PATH
    else:
        raise Exception("Invalid region name. Input must be either of 'CA1', 'CA3' or 'EC'")

    # access the location-time file
    data_filename = f"location-time-speed-{session}.csv"
    loc_time_file_path = os.path.join(SESSIONS_DATA_PATH, region_path, topdir, session, data_filename)

    # check if the the file exists
    if os.path.isfile(loc_time_file_path):
        loc_time_df = pd.read_csv(loc_time_file_path)
        track_length = loc_time_df['x1'].max() - loc_time_df['x1'].min()
        return track_length
    else:
        raise FileNotFoundError(f"Could not find the file {loc_time_file_path}")
    


## define a function to normalize and subsequently circularize the location


def circularize_location(region, topdir, session):
    """
    This function circularizes location based on the speed

    OUTPUT:
    1. circularized_loc_df:
    """

    # find the correct region directory
    if region == 'CA1':
        region_path = CA1_SESSION_PATH
    elif region == 'CA3':
        region_path = CA3_SESSION_PATH
    elif region == 'EC':
        region_path = EC_SESSION_PATH
    else:
        raise Exception("Invalid region name. Input must be either of 'CA1', 'CA3' or 'EC'")

    # access the location-time file
    data_filename = f"location-time-speed-{session}.csv"
    loc_time_file_path = os.path.join(SESSIONS_DATA_PATH, region_path, topdir, session, data_filename)

    # check if the the file exists and read the file if it does
    if os.path.isfile(loc_time_file_path):
        loc_time_df = pd.read_csv(loc_time_file_path)
    else:
        raise FileNotFoundError(f"Could not find the file {loc_time_file_path}")

    # find the min and max x-points
    [x_min, x_max] = [loc_time_df['x1'].max(), loc_time_df['x1'].min()]

    # create a new column with normalized x1 entries
    loc_time_df['normalized_x'] = (loc_time_df['x1'] - x_min)/(x_max - x_min)

    # use the circularizing operation and add circularized location column
    loc_time_df['circularized_location'] = loc_time_df.apply(circularizing_operation, axis=1)

    # return dataframe with required columns
    return loc_time_df[['circularized_location', 'time', 'speed']]


## define the circularizing operation
def circularizing_operation(loc_time_df):
    """
    Use this function in the .apply() method in pandas
    """

    if loc_time_df['speed'] >= 0:
        return loc_time_df['normalized_x']*100
    else:
        return (200 - 100*loc_time_df['normalized_x'])


## define the master function that returns firing rate as a function for location
## this function essentially gives p(x) and lambda(x) for a given delay

def firing_rate_location_data(region, topdir, session, place_fld_size = 4, speed_threshold = 4, delay_time = 0, celltype = 'p'):
    """
    This function essentially gives p(x) and lambda(x) for a given delay.

    INPUT:

    OUTPUT:
    1. rx (lambda(x))
    2. px
    3. bins_x
    """

    # find the correct region directory
    if region == 'CA1':
        region_path = CA1_SESSION_PATH
    elif region == 'CA3':
        region_path = CA3_SESSION_PATH
    elif region == 'EC':
        region_path = EC_SESSION_PATH
    else:
        raise Exception("Invalid region name. Input must be either of 'CA1', 'CA3' or 'EC'")
    
    # extract the circularized location
    circularized_loc_df = circularize_location(region, topdir, session)

    # access the cell-spiketime file
    cell_spt_filename = f"cell-spiketime-file-{session}.csv"
    cell_spt_file_path = os.path.join(SESSIONS_DATA_PATH, region_path, topdir, session, cell_spt_filename)

    # check if the the file exists and read the file if it does
    if os.path.isfile(cell_spt_file_path):
        cell_spt_df = pd.read_csv(cell_spt_file_path)
        # drop nan's if any (there shouldn't be)
        cell_spt_df = cell_spt_df.dropna()
    else:
        raise FileNotFoundError(f"Could not find the file : {cell_spt_file_path}")
    
    ## now we'll find the cell-spiketime-location raw datapoints

    # sort the spiketimes and location
    cell_spt_df = cell_spt_df.sort_values(by=['spiketime'], ignore_index=True)
    circularized_loc_df = circularized_loc_df.sort_values(by=['time'], ignore_index=True)








