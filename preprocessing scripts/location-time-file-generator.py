"""
Preprocessing CRCNS hc-3 linear track sessions.
This script is used to generate the the location-time file for a given session.
The script uses .whl files provided on the hc-3 dataset website for location.
The time series is evaluated using the duration of task and the frame rate for capturing the location (39.06 Hz)
The processed file will be stored in the same directory as the session.

OUTPUT:
location-time-speed-{session}.csv file
Columns = [x1, y1, x2, y2, time, speed]
"""


import os
import numpy as np
import pandas as pd

#### Set the data paths ######
BASE_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3"
METADATA_PATH = "/nadata/cnl/data/Vikrant/hc3/hc3-metadata-tables"
SESSIONS_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3/sessions_data"

# set the last directory name --> we'll use os.path.join later
CA3_SESSION_PATH = "ca3_sessions"
CA1_SESSION_PATH = "ca1_sessions"
EC_SESSION_PATH = "ec_sessions"

##################################################################################################################

## read the metadata files required
cell_type_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-cell.csv'), header=None, names=['id', 'topdir', 'animal', 'ele', 'clu', 'region', 'ne', 'ni', 'eg', 'ig', 'ec', 'idd', 'fireRate', 'totalFireRate', 'type'])
session_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-session.csv'), header=None, names = ['id', 'topdir', 'session', 'behavior', 'familiarity', 'duration'])
electrode_place_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-epos.csv'), header = None, names = ['topdir', 'animal', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16'])
# print(f"electrode placement : {electrode_place_df.head()}")

##################################################################################################################

## set the frame rate
FRAME_RATE = 39.06 # Hz. Can be found in the CRCNS hc-2 data description

##################################################################################################################

## iterate over the directories to access the .whl files

for region in os.listdir(SESSIONS_DATA_PATH):

    # enter the region directory
    region_dir = os.path.join(SESSIONS_DATA_PATH, region)

    # pick out which region we're in
    if 'ca3' in region:
        region_name = 'CA3'
    elif 'ca1' in region:
        region_name = 'CA1'
    elif 'ec' in region:
        region_name = 'EC'

    print("***********************************************")
    print(f"Region name : {region_name}") 
    print("***********************************************")

    ## check if this is a valid directory
    if os.path.isdir(region_dir):

        ## write code for accessing topdirectory
        for topdir in os.listdir(region_dir):
            topdir_path = os.path.join(region_dir, topdir)

            ## check of the topdir_path exists
            if os.path.isdir(topdir_path):
                for session in os.listdir(topdir_path):
                    session_path = os.path.join(topdir_path, session)

                    ## check if the session_path exists
                    if os.path.isdir(session_path):

                        ## Now we're inside the session folder
                        print("----------------------------------------------")
                        print(f"Session under progress: {session}. Topdirectory : {topdir}")
                        print("----------------------------------------------")

                        ## set the whl file path
                        whl_file_path = os.path.join(session_path, f"{session}.whl")

                        ## see if the path exists

                        if os.path.exists(whl_file_path):

                            # read the whl file using pandas
                            whl_file = pd.read_csv(whl_file_path, sep='\t', header=None)
                            whl_file.columns = ['x1', 'y1', 'x2', 'y2']

                            ## create a time series for the whl file
                            ## read the task duration from session_info_df (last column)
                            ## create a linearly spaced array from (0, task_duration (including the last point), step size = 1/frame_rate)

                            # find the task_duration
                            task_duration = (session_info_df.loc[session_info_df['topdir'] == topdir]
                                             .loc[session_info_df['session'] == session]['duration'].to_numpy()[0]
                                             ) # s
                            # print(f"Duration for {session} : {task_duration} s")

                            # create the time series
                            time_series = np.arange(0, task_duration + 1/FRAME_RATE, 1/FRAME_RATE)

                            # print("----------------------------------------------")

                            # print(f" BEFORE dropping -1: \n length(time series) : {len(time_series)}, length(location) : {len(whl_file)}")

                            # print("----------------------------------------------")

                            ## in general there is a small discrepency between the whl and time_series size
                            ## whl file is a larger than the time_series file by a few entries (about 10-ish)
                            ## to align, the code will first remove the additional rows in the whl file. Then the time_series and whl file will be concatenated and then the -1 locations will be dropped

                            # find the difference between lengths of location and time_series
                            diff_len = len(whl_file) - len(time_series)

                            if diff_len > 0:

                                # remove the first diff_len rows from the whl_file
                                whl_file = whl_file.iloc[diff_len :]

                                # print('files aligned') if len(whl_file) == len(time_series) else print(f'NOT aligned: location : {len(whl_file)}, time_series : {len(time_series)}')

                                # add the time column to whl_file
                                whl_file['time'] = time_series

                                # drop all the rows with -1 entries
                                whl_file = whl_file.drop(whl_file[whl_file.isin([-1]).any(axis = 1)].index)

                                # reset indices
                                whl_file = whl_file.reset_index(drop=True)

                                # find the velocity
                                velocity = whl_file['x1'].diff()*FRAME_RATE # cm/s

                                print("----------")
                                print(f"first 5 elements of velocity array = \n {velocity[:5]}")
                                print("----------")

                                # append the velocity array to whl file
                                whl_file['speed'] = velocity

                                # drop the NaN row
                                whl_file = whl_file.dropna()

                                # save the whl_file
                                filename = f"location-time-speed-{session}.csv"
                                data_file_path = os.path.join(session_path, filename)
                                whl_file.to_csv(data_file_path, index = False)

                                print('\n')

                                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                                print(f"Successfully saved: {filename}")
                                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")


                            else:
                                print(f"MISSING data (time series > location) for {session}")
                                break

                            # print("----------------------------------------------")

                            # print(f" AFTER dropping -1: \n length(time series) : {len(time_series)}, length(location) : {len(whl_file)}")

                            # print("----------------------------------------------")










