"""
This script generates a file delay_times.csv containing spiketime delays.
The delay_time arrays will be stored under: /nadata/cnl/data/Vikrant/hc3/processed_data_files


"""

import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv

from helper_functions import *

#### Set the data paths ######
BASE_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3"
METADATA_PATH = "/nadata/cnl/data/Vikrant/hc3/hc3-metadata-tables"
SESSIONS_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3/sessions_data"
PROCESSED_DATA_FILES = "/nadata/cnl/data/Vikrant/hc3/processed_data_files"

# set the last directory name --> we'll use os.path.join later
CA3_SESSION_PATH = "ca3_sessions"
CA1_SESSION_PATH = "ca1_sessions"
EC_SESSION_PATH = "ec_sessions"

## create the file to store the data 
data_file_path = os.path.join(PROCESSED_DATA_FILES, f"delay_times-region-mi_max-p_value.csv")

with open(data_file_path, "w", newline='') as f:
    csv_writer = csv.writer(f)
    first_line = ['delay_time', 'region', 'mi_max', 'p-value']
    csv_writer.writerow(first_line)

print("Created initial blank file")

##################################################################################################################

## defining the delay times
dt = 50e-3
t_delays = np.arange(-1, 1 + dt, dt)

# resamples
num_resamples = 1000


##################################################################################################################
## Take inputs
##################################################################################################################
[region, topdir, session] = [sys.argv[1], sys.argv[2], sys.argv[3]]


print("------------------------------------------------------")
print(f"Running analysis for: {region} -> {topdir} -> {session}")
print("------------------------------------------------------")

##################################################################################################################
###             Bootstrap analysis
##################################################################################################################

## for a given session, ennumerate the number of cells
cell_indices = get_celltype_indices(region, topdir, session, 'p')

print("--------------------------------------------------------------------------")

for id in tqdm(cell_indices, desc = f"Running for {topdir} - {session}", unit = "item"):
    bootstrap_mi = []
    cell_spiketime_loc_df, loc_df = cell_spiketime_location_file_generator(region=region, topdir=topdir, session=session, delay_time=0, celltype='p')
    rx0, px0, binsx0 = firing_rate_location_distributions_generator(cell_spiketime_loc_df, loc_df, region, topdir, session, id)

    if np.amax(rx0) > 1:

        for _ in range(num_resamples):
            rx0_resampled = np.random.choice(rx0, size = len(rx0), replace = True)
            px0_resampled = np.random.choice(px0, size = len(px0), replace = True)

            mi_resampled = skaggs_MI(rx0_resampled, px0_resampled)
            bootstrap_mi.append(mi_resampled)

        # initialize the mi-delay list
        mi_list = []
        for td in t_delays:

            # compute the new distributions
            cell_spiketime_loc_df, loc_df = cell_spiketime_location_file_generator(region=region, topdir=topdir, session=session, delay_time=td, celltype='p')
            rx, px, binsx = firing_rate_location_distributions_generator(cell_spiketime_loc_df, loc_df, region, topdir, session, id)

            # compute the MI for the delay
            mi = skaggs_MI(rx, px)

            # check if mi is positive
            if mi < 0:
                raise Exception("MI cannot be negative. Check binning!")

            # append mi to the list
            mi_list.append(mi)


        # find the max mi
        mi_list = np.array(mi_list)
        mi_max = np.max(mi_list)

        # find the time delay that gives max mi
        td_max = t_delays[np.argmax(mi_list)]
        
        # check if the obtained mi is significant (LOOK FOR DIFFERET WAYS TO QUANTIFY THIS IN THE FUTURE)
        p_value = np.sum(bootstrap_mi >= mi_max)/len(bootstrap_mi)

        # set significance for the data
        if p_value < 0.05:

            # add the obtained mi and time delay to the file
            with open(data_file_path, "a", newline = '') as f:

                csv_writer = csv.writer(f)
                data = [np.round(td_max, 3), region, mi_max, p_value]
                csv_writer.writerow(data)






