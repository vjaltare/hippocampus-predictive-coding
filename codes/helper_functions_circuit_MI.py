import numpy as np
from numpy.random import shuffle, default_rng, normal, choice
import scipy.io as sio
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import glob
import sys
from sklearn.mixture import GaussianMixture
from sklearn import linear_model
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import KernelDensity
from scipy.interpolate import interp1d
from scipy.stats import gamma, invgamma, gaussian_kde, lognorm
from scipy.stats import norm as gauss
from numpy.linalg import norm
from scipy.special import jv
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from helper_functions_hc3_preprocessing import * 

# Global Constants
FIL_DATA_PATH = '/nadata/cnl/home/vjaltare/Documents/datasets/hc3/processed_files/final_processed_files/'
DATA_PATH = '/cnl/data/Vikrant/hc3/gor/gor01-6-7/2006-6-7_11-26-53/'
METADATA_PATH = '/home/vjaltare/Documents/datasets/hc3/metadata/'
CA1_DATA_PATH = '/home/vjaltare/Documents/datasets/hc3/ec014.29/ec014.468/'
FIG_PATH = '/home/vjaltare/Documents/figures/'
LOC_SAMPLING_FREQUENCY = 39.06 # Hz

###############################################################################################
################# Potentially construct a class if this method works ##########################
###############################################################################################

###############################################################################################
##################################### Utility functions ########################################
################################################################################################

## Define a function to return location data NOT histogram: return return the output of the circularize_location function
def location_datapoints(threshold, animal):
    '''
    Use this function to get the datapoints for location. Specify speed threshold using the threshold 
    argument
    Input: velocity threshold (in cm/s), animal (string): 'gor01' for CA3 and 'ec014' for CA1
    Output: pandas dataframe containing datapoints of vx, vy, x1, y1, t, xc.
    NB: We only focus on the central 80% of the track.
    '''
    # setting the percentile flag to zero to ensure that speed threshold is in cm/s
    prc_flag = False

    # load the xyt_dataframe
    if animal == 'gor01':
        xyt_df = pd.read_csv(FIL_DATA_PATH + 'gor01_2006-6-7_11-26-53_loc_time.csv')
        # print(f'Read the location time dataframe for gor01')
    elif animal == 'ec014':
        xyt_df = pd.read_csv(FIL_DATA_PATH + 'ec014_29_468_loc_time.csv')
        # print(f'Read the location time dataframe for ec014')
    else:
        raise Exception('Animal not found')
    
    # call the get_speed_df function to compute the speed_df
    speed_df = get_speed_df(xyt_df)

    # circularize the location
    xyt_cir = circularize_location(speed_df, threshold, prc_flag)

    return xyt_cir['xc'].to_numpy()

## define a function to find the track length traversed in the session
def track_length_traversed(animal):
    '''
    Use this function to find the track length traversed by the animal during the trial
    '''
    # load the xyt_dataframe
    if animal == 'gor01':
        xyt_df = pd.read_csv(FIL_DATA_PATH + 'gor01_2006-6-7_11-26-53_loc_time.csv')
        # print(f'Read the location time daataframe for gor01')
    elif animal == 'ec014':
        xyt_df = pd.read_csv(FIL_DATA_PATH + 'ec014_29_468_loc_time.csv')
        # print(f'Read the location time daataframe for ec014')
    else:
        raise Exception('Animal not found')
    
    x_length = np.max(xyt_df['x1']) - np.min(xyt_df['x1'])
    return x_length

## define a function to return bin edges for location given a bin width in cm
def loc_bin_edges(bin_width, animal):
    '''
    Use this function to get the bin edges for location given a bin width
    Input:
    1. bin_width: in cm
    2. animal: 'gor01', 'eco14'...
    Output:
    1. loc_bin_edges: numpy array with location bin edges
    '''

    # find the track length
    track_len = track_length_traversed(animal)

    # find the number of divisions
    num_div = np.round(track_len/bin_width + 1)

    # construct the bin edges array
    loc_bin_edges = np.linspace(0., 200., int(num_div*2))

    return loc_bin_edges

def total_pyramidal_neurons(animal):
    '''
    Use this function to get the total number of pyramidal neurons form and animal
    Input: 
    1. animal (str): 'gor01', 'ec014' etc..
    Output:
    1. tot_neurons (float): total number of pyramidal neurons 
    '''

    # load the cellid dataframe for an animal
    # load the xyt_dataframe
    if animal == 'gor01':
        cell_id = pd.read_csv(FIL_DATA_PATH + 'gor01_ca3_ele_cell_id.csv')
        # print(f'Read the location time daataframe for gor01')
    elif animal == 'ec014':
        cell_id = pd.read_csv(FIL_DATA_PATH + 'ec014_ca1_ele_cell_id.csv')
        # print(f'Read the location time daataframe for ec014')
    else:
        raise Exception('Animal not found')    
    
    return np.amax(cell_id['id'])

## Function to pick 'k' neurons at a time
def pick_random_neurons(fraction_neurons, animal):
    '''
    Use this function to pick random combinations of neurons from the dataset.
    Input: fraction of neurons (number <= 1), animal
    Output: numpy array with a list of neurons to be selected
    '''

    # load the cellid dataframe for an animal
    # load the xyt_dataframe
    if animal == 'gor01':
        cell_id = pd.read_csv(FIL_DATA_PATH + 'gor01_ca3_ele_cell_id.csv')
        # print(f'Read the location time daataframe for gor01')
    elif animal == 'ec014':
        cell_id = pd.read_csv(FIL_DATA_PATH + 'ec014_ca1_ele_cell_id.csv')
        # print(f'Read the location time daataframe for ec014')
    else:
        raise Exception('Animal not found')
    
    # determine the number of neurons
    tot_neurons = np.amax(cell_id['id'])
    # print(f'Total neurons = {tot_neurons}')

    # pick random id's for a fraction of the neurons
    k = int(round(tot_neurons*fraction_neurons)) - 1


    # pick the random neurons
    rng = default_rng()
    rand_neurons = rng.choice(np.arange(1, tot_neurons), k, replace=False)

    return rand_neurons

## Function to sort the place cells by location
def sort_place_cells(rate_loc_arr):
    '''
    Use this function to find the location sorted place cells
    Input: 
    1. rate_loc_mat: np.ndarray (neurons x location) with place fields
    Output:
    1. sorted_place_cell_mat: np.ndarray (neurons x location) with location sorted place cells
    '''
    # find the id with max value in every row of the rate_x_mat
    id_max = rate_loc_arr.argmax(axis=1)

    # create a linspace of the legth of id_max
    cell_id = np.arange(id_max.shape[0])

    # concatenate the two arrays
    sort_max = np.stack((id_max, cell_id))

    # sort sort_max by row 1
    sort_r1 = np.argsort(sort_max[0, :])

    # sort the matrix
    sort_max = sort_max[:, sort_r1]

    # extract the second row of the sort_max
    sort_idx = sort_max[1, :]

    # sort the rate_x_mat using these keys
    sorted_place_cell_mat = rate_loc_arr[sort_idx, :]

    return sorted_place_cell_mat


##################################################################################################
############################### Functions to get distributions ###################################
##################################################################################################

## define a function to give datapoints for Pk(firing rate) = Pk(rate)

def firing_rate_datapoints(rand_neurons, animal, speed_threshold, delay_time = 0):
    '''
    Use this function to find the firing rate datapoints. This function can be used to find Pk(r).
    This function is also used to return datapoints for the distribution Pk(r, x). The function returns a matrix
    rate_loc_data (#neurons x location bins) -- PENDING
    Inputs:
    1. rand_neuron_id: np.ndarray with id's of random neurons
    2. animal: 'gor01', 'ec014' ...
    3. speed_threshold: in cm/s
    4. delay_time: in seconds, default=0
    Outputs:
    rate_data: numpy array of firing rates datapoints for a fraction of neurons
    rate_location_data: numpy array of 
    '''

    # # pick random neurons
    # rand_neurons = pick_random_neurons(frac_neurons, animal)

    # initialize an empty list to append all the firing rates
    rate_data = []


    # get their firing rate histograms
    if animal == 'gor01':
        for n in rand_neurons:
            rate_dist, nx, bins = get_ca3_firing_rate_hist(delay_time, n, speed_threshold, False)
            rate_data.append(np.divide(rate_dist, np.max(rate_dist), where= np.max(rate_dist)!=0))
    elif animal == 'ec014':
        for n in rand_neurons:
            rate_dist, nx, bins = get_ca1_firing_rate_hist(delay_time, n, speed_threshold, False)
            rate_data.append(np.divide(rate_dist, np.max(rate_dist), where= np.max(rate_dist)!=0))

    else:
        raise Exception('Animal not found')
    
    # convert the list to a 1D numpy array
    rate_data = np.concatenate(rate_data, axis=0)

    return rate_data


############ FUTURE DIRECTION: define a function to return the data (0s and 1s) for spiking of k-neurons #####

############ redefine the function get_cax_firing_rate_hist ##################

def firing_rate_location_data(animal, id, loc_bin_width, speed_threshold, delay_time = 0):
    '''
    Use this function to get the data for firing rate and firing rate as a function of animal's location
    Inputs:
    1. animal: 'gor01', 'ec014' ...
    2. loc_bin_width: size of location bin width in cm.
    3. speed_threshold: in cm/s
    4. delay_time: in seconds, default=0

    Output:
    1. rate_loc_data: numpy array with firing rate as a function of location
    2. location_bins: numpy array with edges of location bins
    '''

    # get the necessary dataframes
    if animal == 'gor01':
        cell_id = pd.read_csv(FIL_DATA_PATH +'gor01_ca3_ele_cell_id.csv')
        ## extract the electrode and cell    
        ele = cell_id.loc[cell_id['id'] == id, 'electrode'].to_numpy()[0]
        cell = cell_id.loc[cell_id['id'] == id, 'cluster'].to_numpy()[0]
        xyt_df = pd.read_csv(FIL_DATA_PATH +'gor01_2006-6-7_11-26-53_loc_time.csv')
        df = delay_ca3_spiketrain_speed(delay_time, ele, cell)

    elif animal == 'ec014':
        cell_id = pd.read_csv(FIL_DATA_PATH + 'ec014_ca1_ele_cell_id.csv')
        ele = cell_id.loc[cell_id['id'] == id, 'electrode'].to_numpy()[0]
        cell = cell_id.loc[cell_id['id'] == id, 'cluster'].to_numpy()[0]
        xyt_df = pd.read_csv(FIL_DATA_PATH + 'ec014_29_468_loc_time.csv')
        df = delay_ca1_spiketrain_speed(delay_time, id)
    else:
        raise Exception('Animal not found')
    

    # get delayed spike train for circularized locations
    shift_df = filter_spiketimes_by_speed(df, speed_threshold, False)

    # compute the bin_edges
    loc_bin_edge_arr = loc_bin_edges(loc_bin_width, animal)

    # get the data for location
    loc_data = location_datapoints(speed_threshold, animal)

    # circularize the location
    speed_df = get_speed_df(xyt_df)
    circ_loc_df = circularize_location(speed_df, speed_threshold, False)

    # find compute the number of spikes in every bin
    spikes_x, bins_spikes_x = np.histogram(shift_df['x_mean'], bins=loc_bin_edge_arr)

    ## extract the indices of datapoints where location is at the track endpoints

    ### create a mask for values less than 10
    mask1 = bins_spikes_x < 10

    ### create a mask for values between 90 and 100
    mask2 = (bins_spikes_x > 90) & (bins_spikes_x < 110)

    ### create a mask for values more than 190
    mask3 = bins_spikes_x > 190

    ### combine all masks
    final_mask = mask1 | mask2 | mask3

    ### use final_mask to find the indices of elements whose
    idx = np.where(final_mask)[0]

    ### make the spikes_x zero at idx locations
    spikes_x[idx[:-1]] = 0
    # plt.stairs(spikes_x, bins_spikes_x)



    # compute the occupancy histogram for the same bin edges
    p_x, bins_p_x = np.histogram(loc_data, bins=loc_bin_edge_arr)

    # compute the normalized version of p_x
    p_x_norm, bins_p_x = np.histogram(loc_data, bins=loc_bin_edge_arr, density=True)
    # print(np.where(p_x==0)[0])

    # divide spikes_x/(p_x*sampling rate) to get firing rate data
    rate_x = np.divide(spikes_x*LOC_SAMPLING_FREQUENCY, p_x, where=(p_x != 0))

    # normalize rate_x by its peak firing rate
    # rate_x = np.divide(rate_x, np.max(rate_x), where=np.max(rate_x)!=0)
    # plt.stairs(rate_x, bins_spikes_x)

    return rate_x, p_x_norm, bins_spikes_x


## define a function that takes in a list of randomly chosen neuron #'s and 
## returns the matrix with rate vs location for all neurons
def rate_location_matrix(rand_neuron_id, animal, loc_bin_width, speed_threshold, delay_time = 0):
    '''
    Use this function to get the matrix with rate vs location for all neurons
    Input:
    1. rand_neuron_id: randomly generated neuron IDs
    2. animal: 'gor01', 'ec014' ...
    3. loc_bin_width: size of location bin width in cm.
    4. speed_threshold: in cm/s
    5. delay_time: in seconds, default=0
    Output:
    1. rate_loc_mat: numpy.ndarray (#loc_bins x #neurons)
    '''

    # declare an empty list
    rate_loc_mat = []

    # iterate over these neurons and keep appending the result to the rate_x_mat
    for i in rand_neuron_id:
        rate_i, p_x, bins_x = firing_rate_location_data(animal, i, loc_bin_width, speed_threshold, delay_time)
        rate_loc_mat.append([rate_i])

    # concatenate the rate_loc_mat to a numpy array
    rate_loc_mat = np.concatenate((rate_loc_mat), axis=0)

    return rate_loc_mat


## define a function that computes the distribution Pk(r, x)
def joint_distribution(rate_loc_mat, r_bins, x_bins):
    '''
    This function returns a 2D histogram of the Pk(x, r) distribution.
    Inputs:
    1. rate_loc_matrix: np.ndarray (#loc_bins x #neurons) output of the rate_location_matrix function
    2. r_bins: #bins for firing rate
    3. x_bins: #bins for location
    Output:
    1. p_x_r: np.ndarray (r_bins x x_bins)
    '''

    # compute the span of location matrix
    x_span = np.linspace(0., 200., rate_loc_mat.shape[1])

    # create a matrix with repeats of x_span, number of neurons times
    x_mat = np.tile(x_span, (rate_loc_mat.shape[0], 1))

    # compute the distribution
    p_r_x, x_edges, y_edges = np.histogram2d(rate_loc_mat.ravel(), x_mat.ravel(), bins=[r_bins, x_bins], density=True)

    return p_r_x, x_edges, y_edges



##########################################################################################################
###################################### MI Calculation Functions ##########################################
##########################################################################################################

## define a function to compute MI between location and rate
def find_MI(p_r_x, p_r, p_x):
    '''
    This function computed MI between the rate and location variables using the KL divergence definition
    Inputs:
    1. p_r_x: np.ndarray (rate_bins x location_bins) P(r, x)
    2. p_r: P(r), np.ndarray (rate_bins, )
    3. p_x: P(x), np.ndarray, (location_bins, )
    Output:
    1. mi: mutual information (float number)
    '''

    # extract the number of x and rate bins
    rate_bins = p_r.shape[0]
    x_bins = p_x.shape[0]

    # smoothen the distributions
    eps = 1e-8


    # check if the number of bins in p_r_x are consistent with p_r and p_x
    if (p_r_x.shape[0] != rate_bins) or (p_r_x.shape[1] != x_bins):
        raise Exception('Inconsistent binning')
    else:
        pass

    # start the for loop and keep track of the sum
    mi = 0

    mi = np.sum(np.nan_to_num(p_r_x * (np.log2(p_r_x + eps) - np.log2(np.outer(p_r, p_x) + eps))))

    return mi
    
    ######################################################################################################

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


# define a function for bootstrap
