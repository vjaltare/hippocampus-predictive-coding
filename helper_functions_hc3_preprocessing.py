import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import os
from numpy.linalg import inv as inv
from numpy.linalg import norm
from sklearn import linear_model
from scipy.signal import find_peaks
from scipy.special import jv


# Global Constants
FIL_DATA_PATH = '/nadata/cnl/home/vjaltare/Documents/datasets/hc3/processed_files/final_processed_files/'
DATA_PATH = '/cnl/data/Vikrant/hc3/gor/gor01-6-7/2006-6-7_11-26-53/'
METADATA_PATH = '/home/vjaltare/Documents/datasets/hc3/metadata/'
LOC_SAMPLING_FREQUENCY = 39.06 # Hz
# get bin size for easy computations
BIN_SIZE =  np.pi/100
# location bins for consistency (each bin is of size pi/200)
BINS_X= np.arange(0., 2*np.pi, BIN_SIZE)



def ch_dir(path):
	'''easily switch between directories'''
	if os.getcwd() != path:
		os.chdir(path)
	# print(f'pwd: {os.getcwd()}')

def delay_spiketrain(t_shift, ele, cell):
    '''
    a function that takes in a time delay in seconds, electrode and cell 
    outputs a dataframe that has columns 'electrode', 'cell', 'x_mean', 'y_mean'
    USE t_shift WITH SIGN: - means time advance, + means time delay
    when t_shift = 0, you get the location-spiketrain without any time-delay
    '''
    # Make sure we're in the right directory
    ch_dir(FIL_DATA_PATH)

    # Load the x-y-t dataframe
    xyt_df = pd.read_csv('gor01_2006-6-7_11-26-53_loc_time.csv')

    # Load the ele-clu-spiketime file
    cell_spt_df = pd.read_csv('gor01_filtered_ca3_spiketimes.csv')

    # consider only a given electrode and cluster
    cell_spt_df = cell_spt_df.loc[cell_spt_df['electrode'] == ele].loc[cell_spt_df['cluster'] == cell]
    # sort the spt file by spiketimes

    cell_spt_df.sort_values(by=['spiketime'], inplace=True)
    cell_spt_df.reset_index(drop=True, inplace=True)

    # time bins: 1/39.06 s
    t_arr = xyt_df['t'].to_numpy()

    # shift the spiketime array
    cell_spt_df['spiketime'] = cell_spt_df['spiketime'] + t_shift

    # drop negative spiketimes
    cell_spt_df = cell_spt_df.loc[cell_spt_df['spiketime'] > 0]

    # extract spiketimes onto a numpy array
    spt = cell_spt_df['spiketime'].to_numpy()

    for i in range(t_arr.shape[0]-1):
        idx = np.where((spt >= t_arr[i]) & (spt < t_arr[i+1]))[0]
        #print(idx)
        if idx.shape[0] > 0:
            # avg x-location
            avg_locx = xyt_df.iloc[i:i+2, -1].to_numpy()
            #print(avg_locx)

            # select the appropriate mean location
            if (avg_locx[0] <= np.pi) & (avg_locx[1] <= np.pi):
                avg_locx = np.mean(avg_locx)
            if (avg_locx[0] > np.pi) & (avg_locx[1] > np.pi):
                avg_locx = np.mean(avg_locx)
            else:
                avg_locx = avg_locx[-1]

            # add the locations to the spiketime file at appropriate indices
            cell_spt_df.loc[idx[0]:idx[-1]+1, 'x_mean'] = avg_locx
            # cell_spt_df.loc[idx[0]:idx[-1], 'y_mean'] = avg_locy
        
    print(cell_spt_df.head())

    return cell_spt_df

def delay_spiketrain_circularized_loc(t_shift, ele, cell, prc):
    '''
    Returns a dataframe with columns: 
    Inputs:
    1. t_shift: time shift in seconds. Negative -> lead, positive -> lag
    2. ele: electrode #
    3. cell: cell #
    4. prc: percentile of threshold speed
    Output:
    pandas dataframe.
    '''
    # Make sure we're in the right directory
    ch_dir(FIL_DATA_PATH)

    # Load the x-y-t dataframe
    xyt_df = pd.read_csv('gor01_2006-6-7_11-26-53_loc_time.csv')

    # compute the speed dataframe
    speed_df = get_speed_df(xyt_df)

    # compute the circularized and speed filtered location dataframe
    filter_speed_df = circularize_location(speed_df, prc, True)
    filter_speed_df.sort_values(by=['t'], inplace=True)
    # print(filter_speed_df.head())

    # Load the ele-clu-spiketime file
    cell_spt_df = pd.read_csv('gor01_filtered_ca3_spiketimes.csv')

    # consider only a given electrode and cluster
    cell_spt_df = cell_spt_df.loc[cell_spt_df['electrode'] == ele].loc[cell_spt_df['cluster'] == cell]
    #print(f"ele cell filtered cell_spt_df: {cell_spt_df.head()}")
    # sort the spt file by spiketimes
    cell_spt_df.sort_values(by=['spiketime'], inplace=True)
    cell_spt_df.reset_index(drop=True, inplace=True)

    # time bins: 1/39.06 s
    t_arr = filter_speed_df['t'].to_numpy()
    # print(f't_array: {t_arr}')

    # shift the spiketime array
    cell_spt_df['spiketime'] = cell_spt_df['spiketime'] + t_shift

    # drop negative spiketimes
    cell_spt_df = cell_spt_df.loc[cell_spt_df['spiketime'] > 0]
    #print(f"cell_spt_df['spiketime']: {cell_spt_df['spiketime']}")

    # extract spiketimes onto a numpy array
    spt = cell_spt_df['spiketime'].to_numpy()
    # print(f'spt: {spt}')

    for i in range(t_arr.shape[0]-1):
        idx = np.where((spt >= t_arr[i]) & (spt < t_arr[i+1]))[0]
        print(f'idx: {idx}')
        if idx.shape[0] > 1:
            # avg x-location
            avg_locx = filter_speed_df.iloc[i:i+2, -1].to_numpy()
            #print(avg_locx)
            #print(avg_locx)
            # select the appropriate mean location
            if (avg_locx[0] <= np.pi) & (avg_locx[1] <= np.pi):
                avg_locx = np.mean(avg_locx)
            elif (avg_locx[0] > np.pi) & (avg_locx[1] > np.pi):
                avg_locx = np.mean(avg_locx)
            else:
                avg_locx = avg_locx[-1]


            # add the locations to the spiketime file at appropriate indices
            print(avg_locx)
            cell_spt_df.loc[idx[0]:idx[-1]+1, 'x_mean'] = avg_locx
            # cell_spt_df.loc[idx[0]:idx[-1], 'y_mean'] = avg_locy
        
    print(cell_spt_df.head())

    return cell_spt_df

def unique_ele():
    '''Call this function to find a list of unique electrodes'''
    ch_dir(FIL_DATA_PATH)
    spt_df = pd.read_csv('gor01_filtered_ca3_spiketimes.csv')
    unique_ele = np.array(spt_df['electrode'].unique())
    # print(f'Unique electrodes: {unique_ele}')
    return np.sort(unique_ele)

def unique_cells(ele):
    '''Call this function to find the list of unique cells in an electrode
    Input: 
    1. ele: electrode # 
    '''
    ch_dir(FIL_DATA_PATH)
    spt = pd.read_csv('gor01-ca3_spiketime-location.csv')
    spt = spt.loc[spt['electrode'] == ele]
    cells = np.array(spt['cluster'].unique())
    return np.sort(cells)



def mi(lx, px, bin_loc):
    '''function to output mutual information'''
    bin_size = np.diff(bin_loc)[0] # assuming equispaced bins
    # remove the zeros in lambda(x) so that the log doesn't blow
    ## make a dummy dataframe
    dummy = pd.DataFrame(
        {
            'lx': lx,
            'px': px
        }
    )
    ## remove the rows with lx=0
    dummy = dummy.loc[dummy['lx'] > 0]
    # find overall firing rate 
    l_overall = exp_value(dummy['lx'], dummy['px'], bin_loc)
    # split up the sum into two parts
    l_log = np.multiply(dummy['lx'], dummy['px'])
    mi = np.dot(l_log, np.log2(dummy['lx']/l_overall))

    return mi

def mi_spike(lx, px, bin_loc):
    '''function to output mutual information'''
    bin_size = np.diff(bin_loc)[0] # assuming equispaced bins
    # remove the zeros in lambda(x) so that the log doesn't blow
    ## make a dummy dataframe
    dummy = pd.DataFrame(
        {
            'lx': lx,
            'px': px
        }
    )
    ## remove the rows with lx=0
    dummy = dummy.loc[dummy['lx'] > 0]
    # find overall firing rate 
    l_overall = exp_value(dummy['lx'], dummy['px'], bin_loc)
    # split up the sum into two parts
    l_log = np.multiply(dummy['lx'], dummy['px'])
    mi = np.dot(l_log, np.log2(dummy['lx']/l_overall))

    return mi/l_overall

def exp_value(lx, px, bins):
    ''''Function gives expectation value given lambda(x) and p(x) = lambda'''
    # bin size 
    bin_size = np.diff(bins)[0]

    # dot product of the arrays* bin size == discrete integral
    l = np.dot(lx,px)*bin_size

    return l



def get_speed_df(xyt_df):
	'''
	Function returns a dataframe with columns: x1, y1, vx, vy, t
	Procedure:
	1. Find the dx, dy using difference between successive elements
	2. divide by the dt
	3. Create a consolidated dataframe. 
	4. Drop NaNs
	'''


	# finding speeds in x and y directions
	vx = xyt_df['x1'].diff()
	# print(f'vx: {vx}')
	vy = xyt_df['y1'].diff()
	# print(f'vy: {vy}')
	dt = xyt_df['t'].diff()
	# print(f'dt: {dt}')

	# make a consolidated dataframe
	speed_df = pd.DataFrame(
	{
	    'vx' : vx/dt,
	    'vy' : vy/dt,
	    'x1' : xyt_df['x1'],
	    'y1' : xyt_df['y1'],
	    't' : xyt_df['t'] 
	}
	)

	speed_df.dropna(inplace=True)
	# print(f'size of speed_df = {speed_df.shape[0]}')
	# speed_df.head()
	return speed_df


def filter_loc_by_speed(speed_df, prc, prc_flag):
    '''
    Function returns a dataframe with speed filtered locations.
    1. Columns: x1, y1, vx, vy, t
    2. prc: Percentile off speed. Locations at which speed falls below this 
        speed will be truncated.
    '''

    try:
        type(prc_flag) == bool
    except TypeError:
        raise TypeError('The last argument should be bool')

    if prc_flag:
        v_threshold = np.percentile(abs(speed_df['vx']), np.floor(prc))

    else:
        v_threshold = prc 

    #np.percentile(abs(speed_df['vx']), prc)

    # truncate the locations where speed < vx_prc
    speed_df = speed_df.loc[abs(speed_df['vx']) > v_threshold]

    return speed_df


def save_df(df, path, filename):
	'''Function to save dataframes as .csv files'''
	df.to_csv(path + filename, index=False)


def normalize_location(speed_df):
	'''Normalize locations w.r.t. x. 
	Input: speed_filtered location dataframe for better results
	'''
	### translate the x location to start at zero
	speed_df.loc[:, 'y1'] = speed_df.loc[:,'y1'] - np.amin(speed_df['y1'])
	### Divide by the largest location value to compress the locations from 0-1
	speed_df.loc[:,'y1'] = speed_df.loc[:,'y1']/np.amax(speed_df['x1'])


	### translate the y location to start at zero
	speed_df.loc[:, 'x1'] = speed_df.loc[:,'x1'] - np.amin(speed_df['x1'])
	### Divide by the largest x location value to compress the locations from 0-1
	speed_df.loc[:,'x1'] = speed_df.loc[:,'x1']/np.amax(speed_df['x1'])

	return speed_df


def circularize_location(speed_df, prc, prc_flag):
    '''
    Use this function to get speed-filtered, circularized location dataframe.
    Function takes in a dataframe with columns: vx, vy, x1, y1, t 
    Output: a similar dataframe with circularized location (0-2*pi)
    '''


    df = filter_loc_by_speed(normalize_location(speed_df), prc, prc_flag)

    # separate the speeds in positive and negative and then concat the circularized locations
    pos_speed = df.loc[df['vx'] > 0]
    neg_speed = df.loc[df['vx'] < 0]

    pos_speed['xc'] = pos_speed.loc[:,'x1']*np.pi 
    neg_speed['xc'] = (2 - neg_speed.loc[:,'x1'])*np.pi 

    # concat the dataframes
    df = pd.concat([pos_speed, neg_speed], axis=0)
    df.sort_values(by=['t'], inplace=True)
    df.reset_index(drop=True, inplace=True)


    return df

def get_occupany_hist(circularize_loc_df):
    '''
    Input: Output of the circularize_location() function
    Easier way to get p(x).
    Make sure to NORMALIZE after executing this function to get pdf.
    '''
    nx, binsx = np.histogram(circularize_loc_df['xc'], bins=BINS_X)
    return nx, binsx[:-1]

# define von Mises template function

def von_Mises_template(theta, k, theta_0):
    '''Returns a von Mises function b(theta) = exp(k cos(theta - theta_0))/(2*pi*I0(k))'''

    return np.exp(k*np.cos(theta - theta_0))#/(2*np.pi*jv(0, k))

def span_x_von_Mises(num_functions, x_length):
    '''Returns a matrix X with every column having a von Mises function with different theta_0 spanning the track.
    x_length is the length of bin array for p(x) and lambda(x)'''
    # create a theta_0 array
    theta_0_arr = np.linspace(0., 2*np.pi, num_functions+1)

    # arbitrarily define k
    k =  100#400#5/np.diff(theta_0_arr)[0] #1/np.pi 

    # define a location (x) array
    x = np.linspace(0., 2*np.pi, x_length)

    # define empty X-matrix
    X = np.empty([x.shape[0], theta_0_arr.shape[0]])

    # create the X-matrix
    for i, t in enumerate(theta_0_arr):
        X_column = von_Mises_template(x, k, t)
        X[:,i] = X_column

    return X, theta_0_arr

def pseudo_inverse_coefficients(X, y):
    '''Returns a vector of length of the number of von Mises functions used containing coefficients for each of them.
    Pseudo-inverse used.'''
    return inv(X.T@X)@X.T@y

def ridge_cv(X, y, alpha):
    '''Performs ridge regression with cross-validation using scikit learn RidgeCV
    Input: 1. X: matrix whose columns are the von Mises functions tiling the track
           2. y: probability/frequency infered from histograms of the data
           3. alpha: vector of alpha parameters
    Output: 1. coefficients
            2. alphas
    '''
    reg = linear_model.RidgeCV(alphas=alpha)
    return reg.fit(X,y)


def get_firing_rate_hist(delay_time, ele, cell, prc, prc_flag):
    '''
    Returns discrete lambda(x) distribution in Hz for a given cell and delay-time
    Inputs:
    1. delay_time (float): time in seconds by which the spiketrain needs to be delayed
    2. ele (int): electrode #
    3. cell (int): cell #
    4. prc (int): velocity threshold percentile 
    5. prc_flag: bool. True for percentile threshold, False for float threshold

    Outputs:
    n, bins (np.ndarray): firing rate and bins to plot histogram
    LIVE FUNCTION!
    '''
    # set the correct path
    ch_dir(FIL_DATA_PATH)

    # read the xyt_df
    xyt_df = pd.read_csv('gor01_2006-6-7_11-26-53_loc_time.csv')

    # get delayed spike train for circularized locations
    df = delay_spiketrain_speed(delay_time, ele, cell)
    shift_df = filter_spiketimes_by_speed(df, prc, prc_flag)

    # find the PDF histogram of spike locations
    n_loc, bins_loc = np.histogram(shift_df['x_mean'], bins=BINS_X)

    # find the speed dataframe
    speed_df = get_speed_df(xyt_df)

    # compute the number of times the rat visited the location
    circ_loc_df = circularize_location(speed_df, prc, prc_flag)

    # get the occypancy histogram
    n_x, bins_x = get_occupany_hist(circ_loc_df)
    # print(n_x)

    # remove points where nx = 0
    ## create a dummy dataframe

    dummy = pd.DataFrame(
        {
            'n_loc': n_loc,
            'n_x': n_x,
            'bins_x': bins_x
        }
    )

    ## filter out places where n_x = 0
    dummy = dummy.loc[dummy['n_x'] != 0]

    ## get numpy arrays back
    n_loc = dummy['n_loc'].to_numpy()
    n_x = dummy['n_x'].to_numpy()
    bins_x = dummy['bins_x'].to_numpy()

    # compute the firing rate
    lambda_loc = (n_loc/n_x)*LOC_SAMPLING_FREQUENCY
    lambda_loc = np.nan_to_num(lambda_loc, nan=0)
    

    return lambda_loc, bins_x



def delay_spiketrain_speed(t_shift, ele, cell):
    '''
    This function takes time delay (signed; in seconds), electrode #, cell # as inputs.
    Output: Dataframe with columns electrode, cluster, x_mean, vx_mean.
    The column x_mean can be used to get histogram for firing locations of the cell. 
    vx_mean can be used to filter the output of this dataframe to the desired speed threshold
    '''

    # make sure that the electrode and cluster are CA3
    clu_flag = False
    ele_flag = False

    if (ele in unique_ele()) & (cell in unique_cells(ele)):
        ele_flag = True
        clu_flag = True
    else:
        pass

    if not clu_flag & ele_flag:
        raise Exception('Entered electrode-cell combination is not CA3')
    
    # Make sure we're in the right directory
    ch_dir(FIL_DATA_PATH)

    # Load the x-y-t dataframe
    xyt_df = pd.read_csv('gor01_2006-6-7_11-26-53_loc_time.csv')

    # compute speed dataframe
    speed_df = get_speed_df(xyt_df)

    # circularize with NO threshold
    circ_loc_df = circularize_location(speed_df, 0, True)

    # Load the ele-clu-spiketime file
    cell_spt_df = pd.read_csv('gor01_filtered_ca3_spiketimes.csv')

    # consider only a given electrode and cluster
    cell_spt_df = cell_spt_df.loc[cell_spt_df['electrode'] == ele].loc[cell_spt_df['cluster'] == cell]
    # sort the spt file by spiketimes

    cell_spt_df.sort_values(by=['spiketime'], inplace=True)
    cell_spt_df.reset_index(drop=True, inplace=True)

    # time bins: 1/39.06 s
    t_arr = circ_loc_df['t'].to_numpy()

    # shift the spiketime array
    cell_spt_df['spiketime'] = cell_spt_df['spiketime'] + t_shift

    # drop negative spiketimes
    cell_spt_df = cell_spt_df.loc[cell_spt_df['spiketime'] > 0]

    # extract spiketimes onto a numpy array
    spt = cell_spt_df['spiketime'].to_numpy()

    for i in range(t_arr.shape[0]-1):
        idx = np.where((spt >= t_arr[i]) & (spt < t_arr[i+1]))[0]
        #print(idx)
        if idx.shape[0] > 0:
            # avg x-location
            avg_locx = circ_loc_df.iloc[i:i+2, -1].to_numpy()
            # avg x-velocity
            avg_vx = circ_loc_df.iloc[i:i+2, 0].to_numpy()
            #print(avg_locx)

            # select the appropriate mean location and append the mean x-velocity
            if (avg_locx[0] <= np.pi) & (avg_locx[1] <= np.pi):
                avg_locx = np.mean(avg_locx)
            elif (avg_locx[0] > np.pi) & (avg_locx[1] > np.pi):
                avg_locx = np.mean(avg_locx)
            else:
                avg_locx = avg_locx[-1]

            # select appropriate mean velocity
            if (avg_vx[0] <= 0) & (avg_vx[1] <= 0):
                avg_vx = np.mean(avg_vx)
            elif (avg_vx[0] > 0) & (avg_vx[1] > 0):
                avg_vx = np.mean(avg_vx)
            else:
                avg_vx = avg_vx[-1]

            # add the locations to the spiketime file at appropriate indices
            cell_spt_df.loc[idx[0]:idx[-1]+1, 'x_mean'] = avg_locx
            cell_spt_df.loc[idx[0]:idx[-1]+1, 'vx_mean'] = avg_vx
            # cell_spt_df.loc[idx[0]:idx[-1], 'y_mean'] = avg_locy
        
    # print(cell_spt_df.head())
    cell_spt_df.dropna(inplace=True)

    return cell_spt_df

def filter_spiketimes_by_speed(cell_spt_df, v_threshold, prc_flag):
    '''
    This function takes in the output of the delay_spiketrain_speed function and returns the speed_filtered version of it.
    1. if v_threshold is a percentile, set the prc_flag to True. 
    2. if v_threshold is a speed, set the prc_flag to False. 
    '''

    try:
        type(prc_flag) == bool
    except TypeError:
        raise TypeError('The last argument should be bool')

    if prc_flag:
        v_threshold = np.percentile(abs(cell_spt_df['vx_mean']), np.floor(v_threshold))

    else:
        pass

    cell_spt_df = cell_spt_df.loc[abs(cell_spt_df['vx_mean']) > v_threshold]

    return cell_spt_df


# define a function to get optimal number of von Mises functions required to fit a curve
def get_optimal_basis_functions(y_data, x_data):
    '''
    This function outputs the optimal number of basis functions to fit the data.
    Input: y_data, x_data: distribution of p(x) and lambda(x) or any periodic function for that matter.
    All the possible number of functions with R^2 > 0.9 will be returned
    Output: pd.DataFrame with columns N_functions, r_squared
    LIVE!!
    '''
    # define an array for number of von Mises functions
    n_func = np.arange(1, 101, 1)

    # alphas for ridgecv
    alphas = np.logspace(-3, 6, 10)

    # empty lists to hold num functions and r
    num_func_list = []
    r_list = []

    # initiate for loop
    for n, i in enumerate(n_func):
        f, theta = span_x_von_Mises(n, y_data.shape[0])
        # normalize f: gives better results
        f = f/np.amax(f[:,0])

        # determine the ridge regression fit
        fit = ridge_cv(f, y_data, alphas)
        # exctract weights
        W = fit.coef_

        # reconstruct the function
        y_fit = f@W
        plt.plot(x_data, y_fit)
        plt.plot(x_data, y_data)

        # find the coefficient of determination
        # res = (y_data - y_fit)**2
        # res = res.sum()
        # print(res)
        # var = np.var(y_data)
        # print(var)
        # r = 1 - (res/var)
        # print(r)

        # cosine similarity
        r = np.dot(y_fit, y_data)/(norm(y_fit)*norm(y_data))
        #print(r)

        # append to a list if r > 0.9
        if r > 0.8:
            num_func_list.append(n)
            r_list.append(r)
        else:
            pass

        # make a dataframe
        df = pd.DataFrame(
            {
                'num_functions': num_func_list,
                'cosine_sim': r_list 
            }
        )

    return df

def is_place_cell(ele, cell, prc, prc_flag):
    '''
    This function takes in electrode #, cell #, speed threshold (prc), and prc_flag as inputs and returns
    True if the cell meets the criteria for place cell. Similar to Senzai et al.
    '''
    # define condition flags
    peak_firing_rate_flag = False
    num_peaks_flag = False
    firing_spread_flag = False
    # call the delay_spiketrain_circularized_loc function
    fr_distr, bins_fr = get_firing_rate_hist(0, ele, cell, prc, prc_flag)

    # Peak firing rate condition
    if np.amax(fr_distr) > 1:
        peak_firing_rate_flag = True

    # Find the number of peaks > 50% of the 
    peak_idx, _ = find_peaks(fr_distr, height=0.5*np.amax(fr_distr))
    if peak_idx.shape[0] < 2:
        num_peaks_flag = True
    else:
        pass

    # Spread i.e. length of track for which the cluster was firing.
    # should be > 5 bins; < 50 bins
    nnz = np.nonzero(fr_distr)
    nnz_arr = nnz[0]
    l = nnz_arr.shape[0]

    if 5 < l < BINS_X.shape[0]/8:
        firing_spread_flag = True
    else:
        pass

    
    return peak_firing_rate_flag & num_peaks_flag & firing_spread_flag


def find_place_cells(prc, prc_flag):
    '''
    This function returns a dataframe with columns 'electrode', 'cell'. These are the clusters that satisfy the place cell criteria
    from the is_place_cell() function.
    Input: Mention the speed threshold in prc (set prc_flag = True) or cm/s (set prc_flag=False)
    Output: pd.DataFrame with columns ['electrode', 'cell']
    '''

    # get all the unique CA3 electrodes
    ca3_ele = unique_ele()

    # initialize empty lists
    ele_list = []
    clu_list = []

    # initiate for loop and find if the specific clusters are place cells
    for e in ca3_ele:
        ca3_clu = unique_cells(e)
        for c in ca3_clu:
            if is_place_cell(e, c, prc, prc_flag):
                ele_list.append(e)
                clu_list.append(c)
            else:
                pass
    
    # dump the lists onto a dataframe
    df = pd.DataFrame(
        {
            'electode' : ele_list,
            'cell' : clu_list
        }
    )

    return df