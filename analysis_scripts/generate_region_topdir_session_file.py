"""
Use this file to generate a file containing region-topdir-session to be used a command line inputs
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

## define the output file path
out_file_path = os.path.join(BASE_DATA_PATH, f"region-topdir-session.txt")

# start iterating over directories
with open(out_file_path, "w") as f:
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
                            print(f"Session under progress: {session_path}")

                            f.write(f"{region_name} {topdir} {session} \n")

    f.close()
