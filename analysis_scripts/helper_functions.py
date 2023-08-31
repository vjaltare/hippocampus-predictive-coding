"""
This file contains helper functions for analyzing the mutual information between spiking of hippocampal and entorhinal cortical 
neurons.
The data are taken from CRCNS hc-3 dataset.

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#######################################################################################################################

#### Set the data paths ######
BASE_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3"
METADATA_PATH = "/nadata/cnl/data/Vikrant/hc3/hc3-metadata-tables"
SESSIONS_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3/sessions_data"

# set the last directory name --> we'll use os.path.join later
CA3_SESSION_PATH = "ca3_sessions"
CA1_SESSION_PATH = "ca1_sessions"
EC_SESSION_PATH = "ec_sessions"

# Tolerance for merging
TOL_ = 1e-2

# framerate
FRAME_RATE = 39.06 # Hz

# sampling rate for spikes
SAMPLE_RATE = 20e3 # Hz

##################################################################################################################

## read the metadata files in case required
cell_type_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-cell.csv'), header=None, names=['id', 'topdir', 'animal', 'ele', 'clu', 'region', 'ne', 'ni', 'eg', 'ig', 'ec', 'idd', 'fireRate', 'totalFireRate', 'type'])
session_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-session.csv'), header=None, names = ['id', 'topdir', 'session', 'behavior', 'familiarity', 'duration'])
electrode_place_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-epos.csv'), header = None, names = ['topdir', 'animal', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16'])

#################################################################################################################

#################################################################################################################
#################                       Auxillary functions
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
    

## define a function to return bin edges for location given a bin width in cm
def find_loc_bin_edges(place_field_size, region, topdir, session):
    '''
    Use this function to get the bin edges for location given a bin width
    Input:
    1. place_field_size (cm)
    '''

    # find the track length
    track_len = find_track_length(region, topdir, session)

    # find the number of divisions
    num_div = np.round(track_len/place_field_size + 1)

    # construct the bin edges array
    loc_bin_edges = np.linspace(0., 200., int(num_div*2))

    return loc_bin_edges
    


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

#######################################################################################################################
###                         PART 1 : Generating the files
#######################################################################################################################

## define a function to give cell-spiketime-location dataframe

def cell_spiketime_location_file_generator(region, topdir, session, delay_time = 0, celltype = 'p'):
    """
    This function essentially gives cell-spiketime-location dataframe

    INPUT:
    1. region: "CA1", "CA3" or "EC"
    2. topdir: topdirectories
    3. session: session name
    4. delay time: in seconds, delay time for spike train

    OUTPUT:
    1. cell_spiketime_location_df (cell_spt_loc_df)
    2. circularized_loc_df
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

        # consider the appropriate celltype and the region
        cell_spt_df = (cell_spt_df.loc[cell_spt_df['region'] == region]
                       .loc[cell_spt_df['celltype'] == celltype]
                       )
    else:
        raise FileNotFoundError(f"Could not find the file : {cell_spt_file_path}")
    
    ## now we'll find the cell-spiketime-location raw datapoints

    # add the time delay to spiketimes
    cell_spt_df['spiketime'] = cell_spt_df['spiketime'] + delay_time

    # sort the spiketimes and location
    cell_spt_df = cell_spt_df.sort_values(by=['spiketime'], ignore_index=True)
    circularized_loc_df = circularized_loc_df.sort_values(by=['time'], ignore_index=True)

    # round off spiketimes and location times to 3 decimal places
    cell_spt_df['spiketime'] = cell_spt_df['spiketime'].round(3)
    circularized_loc_df['time'] = circularized_loc_df['time'].round(3)

    # merge the dataframes along spiketime, time axes
    cell_spt_loc_df = pd.merge_asof(cell_spt_df, circularized_loc_df, left_on='spiketime', right_on='time', tolerance=TOL_)

    # drop unnecessary columns
    cell_spt_loc_df = cell_spt_loc_df.drop(columns = ['time', 'region', 'celltype'])

    return cell_spt_loc_df, circularized_loc_df


#######################################################################################################################
###                         PART 2 : Generating the distributions
#######################################################################################################################


## define the master function that returns firing rate as a function for location
## this function essentially gives p(x) and lambda(x) for a given delay

def firing_rate_location_distributions_generator(cell_spiketime_location_df, circularized_location_df, region, topdir, session, neuron_number = None, bin_width = 4):
    """
    This function gives p(x) and lambda(x) for a given delay

    INPUT:
    [Will populate this later. place field size is in cm]

    OUTPUT:
    1. lambda(x) (spikes/s)
    2. p(x) (normalized distribution)
    3. location_bin_edges: well...location bin edges. used to plot histogram using stairs()
    """


    ## finding the location bin edges
    location_bin_edges = find_loc_bin_edges(bin_width, region, topdir, session)

    ## generate p(x)
    px, bins_px = np.histogram(circularized_location_df['circularized_location'], bins = location_bin_edges)
    px_norm, bins_px = np.histogram(circularized_location_df['circularized_location'], bins = location_bin_edges, density = True)

    ## generating lambda(x)

    # check if the mentioned neuron exists
    neuron_list = cell_spiketime_location_df['cell'].unique()
    
    if neuron_number not in neuron_list:
        raise Exception(f"Neuron # {neuron_number} does not exist")
    else:
        # consider the cell_spiketime_location dataframe only for the mentioned neuron
        spt_loc_df = cell_spiketime_location_df.loc[cell_spiketime_location_df['cell'] == neuron_number]
                      

    # first, find the #spikes in each location bin. This is done by counting the number of times a location point occurs
    num_spikes, bins_spikes = np.histogram(spt_loc_df['circularized_location'], bins = location_bin_edges)
    
    # rate = num_spikes(s)*frame rate/p(x)
    rate_x = np.divide(num_spikes*FRAME_RATE, px, where = (px != 0))

    ## removing the endpoint bins

    # creating masks
    mask1 = location_bin_edges < 10

    mask2 = (location_bin_edges > 90) & (location_bin_edges < 110)

    mask3 = location_bin_edges > 190

    final_mask = mask1 | mask2 | mask3

    # extract indices where final_mask is true
    idx_drop = np.where(final_mask)[0]

    # make rate and p(x) = 0 from endpoint location
    rate_x[idx_drop[:-1]] = 0
    px_norm[idx_drop[:-1]] = 0

    return rate_x, px_norm, location_bin_edges


    
########################################################################################################
###                     Part 3: Mutual information functions
########################################################################################################

# Define a function to compute Skaggs information
def skaggs_MI(rx, px):
    '''
    This function computes Skaggs information (Skaggs..Markus, NeurIPS 1992)
    Inputs: 
    1. rx: np.ndarray. Firing rate as a function of location (normalized by max firing rate)
    2. px: np.ndarray. Probability of animal visiting a location.
    Output:
    1. I: float. MI between firing rate and location
    '''

    # compute the mean firing rate
    rx_mean = np.sum(rx * px)
    # print(f'Mean firing rate= {rx_mean:.02f} spikes/s')

    # define a small epsilon
    eps = 1e-8

    # compute the MI
    I = np.sum(rx * px * np.log2((rx + eps)/(rx_mean + eps)))

    return I



