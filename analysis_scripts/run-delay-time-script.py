"""
Use this script to run the delays-core-script.py in parallel.
This script takes in inputs from the region-topdir-session.txt file and feeds it into the helper functions.

"""

import concurrent.futures
import multiprocessing
import os

#### Set the data paths ######
BASE_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3"
METADATA_PATH = "/nadata/cnl/data/Vikrant/hc3/hc3-metadata-tables"
SESSIONS_DATA_PATH = "/nadata/cnl/data/Vikrant/hc3/sessions_data"

# set the last directory name --> we'll use os.path.join later
CA3_SESSION_PATH = "ca3_sessions"
CA1_SESSION_PATH = "ca1_sessions"
EC_SESSION_PATH = "ec_sessions"



def run_script(args):
    region, topdir, session = args

    # enter the command
    command = f"python3 delays-core-script.py {region} {topdir} {session}"

    # run the command using os
    result = os.system(command)

    return result


if __name__ == "__main__":

    ## basically the above line ensures that this script is run on its own and not as a module

    # read the input arguments
    input_args = [] # this will be a list of tuples

    input_file_path = os.path.join(BASE_DATA_PATH, "region-topdir-session.txt")

    with open(input_file_path, "r") as f:
        print("Input file accessed successfully")

        for line in f:
            region, topdir, session = line.strip().split()
            input_args.append((region, topdir, session))

    # this part of the script I don't fully understand -- it's courtesy of ChatGPT

    # potentially get #cores
    num_workers = multiprocessing.cpu_count()

    with concurrent.futures.ProcessPoolExecutor(max_workers = num_workers) as executor:
        results = executor.map(run_script, input_args)

    # for result in results:
    #     print(f"Script exit code: {result}")

