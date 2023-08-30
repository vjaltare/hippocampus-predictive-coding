"""
This script generates cell-spiketime-location file which is used to generate the distributions p(x) and lambda(x)

OUTPUT:
cell-spiketime-file-{session}.csv
Columns = [cell, spiketime, celltype, region]
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

## set a tolerance for merging dataframes
TOL_ = 1e-1

#############################################################################################################

## start iterating over directories
for region in os.listdir(SESSIONS_DATA_PATH):

    ## enter the region directory
    region_dir = os.path.join(SESSIONS_DATA_PATH, region)

    ## pick out what region we're in
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
                        print(f"Session under progress: {session_path}")
                        # print("----------------------------------------------")

                        # check if the processed cell-spiketime and location-time files exist
                        cell_spiketime_filename = f"cell-spiketime-file-{session}.csv"
                        location_time_speed_filename = f"location-time-speed-{session}.csv"

                        cell_spiketime_file_path = os.path.join(session_path, cell_spiketime_filename)
                        location_time_speed_file_path = os.path.join(session_path, location_time_speed_filename)

                        if os.path.isfile(cell_spiketime_file_path) and os.path.isfile(location_time_speed_file_path):

                            # read the cell-spiketime-file
                            cell_spiketime_file = pd.read_csv(cell_spiketime_file_path)

                            # sort the spiketimes 
                            cell_spiketime_file = cell_spiketime_file.sort_values(by=['spiketime'], ignore_index=True)

                            # read the location-time-file
                            location_time_file = pd.read_csv(location_time_speed_file_path)

                            # merge the two files along spiketime-time axes
                            cell_spiketime_location_file = pd.merge_asof(cell_spiketime_file, location_time_file, left_on="spiketime", right_on="time", tolerance=TOL_)

                            print("----------------")
                            print(f"Last two entries in cell-spiketime-location-file \n {cell_spiketime_location_file.iloc[-2:]}")
                            print("----------------")

                            # drop the time column
                            cell_spiketime_location_file = cell_spiketime_location_file.drop(columns=['y1', 'x2', 'y2', 'time'])

                            # save the cell spiketime location file
                            filename = f"cell-spiketime-location-{session}.csv"
                            data_file_path = os.path.join(session_path, filename)
                            cell_spiketime_location_file.to_csv(data_file_path, index=False)


                            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                            print(f"Successfully saved: {filename}")
                            print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")



                        else:
                            raise Exception(f"Processed files do not exist for {session}")
                           
