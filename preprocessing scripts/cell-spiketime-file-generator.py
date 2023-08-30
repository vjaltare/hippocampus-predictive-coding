"""
Preprocessing linear track sessions from CRCNS hc-3 dataset.
This script is used to generate the cell-spiketime-file for a given session.
The script reads all .clu and .res files to find the active clusters and spiketimes respectively.
It then concatenates them, removes artifacts and also appends celltype and region info.
This way we can potentially have cells from neighboring brain region for simultaneous analysis.

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

##################################################################################################################

## read the metadata files required
cell_type_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-cell.csv'), header=None, names=['id', 'topdir', 'animal', 'ele', 'clu', 'region', 'ne', 'ni', 'eg', 'ig', 'ec', 'idd', 'fireRate', 'totalFireRate', 'type'])
session_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-session.csv'), header=None, names = ['id', 'topdir', 'session', 'behavior', 'familiarity', 'duration'])
electrode_place_info_df = pd.read_csv(os.path.join(METADATA_PATH, 'hc3-epos.csv'), header = None, names = ['topdir', 'animal', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8', 'e9', 'e10', 'e11', 'e12', 'e13', 'e14', 'e15', 'e16'])
# print(f"electrode placement : {electrode_place_df.head()}")

##################################################################################################################

## create a file to store the cell-spiketime-celltype file
data = pd.DataFrame(columns = ['cell', 'spiketime', 'celltype', 'region'])
temp_data_df = pd.DataFrame(columns = ['ele', 'cell', 'spiketime', 'celltype', 'region'])

data_list = []
temp_data_list = []
##################################################################################################################

##################################################################################################################
## defining a function that can process the clu-spiketime

def concat_cell_spiketime(session_path, clu_file_name, res_file_name):

    with open(os.path.join(session_path, clu_file_name), 'r') as f:
        cluster_arr = [int(line.strip()) for line in f.readlines()[1:]]
    
    with open(os.path.join(session_path, res_file_name), 'r') as f:
        spiketime_arr = [float(line.strip()) for line in f.readlines()]

    # filter spiketimes
    fil_spiketimes = np.array(spiketime_arr)[np.array(cluster_arr) > 1]  

    # filter clusters
    fil_cluster_arr = np.array([cell for cell in cluster_arr if cell > 1])

    return fil_spiketimes, fil_cluster_arr

## defining a function to get cell type
def get_cell_type(cell_type_info_df, topdir, ele, clu):
    cell_type = (cell_type_info_df.loc[cell_type_info_df['topdir'] == topdir]
                 .loc[cell_type_info_df['ele'] == ele]
                 .loc[cell_type_info_df['clu'] == clu]['type']
                 )
    if len(cell_type) == 0:
        cell_type = np.nan
    else:
        cell_type = cell_type.values[0]

    return cell_type

##################################################################################################################

# start iterating over directories
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

                        # check how many electrodes are present by counting no. of .clu.N files
                        ## make a list of all the files
                        all_files = os.listdir(session_path)

                        ## count all the files containing .clu in their name
                        num_ele = sum(1 for file in all_files if '.clu' in file) ## really neat line of code, thanks to ChatGPT
                        print(f"No. electrodes = {num_ele}")

                        print("----------------------------------------------")

                        # check if the file already exists
                        if os.path.isfile(os.path.join(session_path, f"cell-spiketime-file-{session}.csv")):
                            print(f"The cell-spiketime-file for {session} already exists")
                            break

                        for ele in range(1, num_ele + 1):
                            print(f"*** ELECTRODE {ele} ***")
                            clu_file_path = os.path.join(session_path, f"{session}.clu.{ele}")
                            res_file_path = os.path.join(session_path, f"{session}.res.{ele}")

                            # check if the clu and res files exist
                            if os.path.exists(clu_file_path) and os.path.exists(res_file_path):
                                # call the function to obtain the spiketimes and cells
                                spiketimes, cells = concat_cell_spiketime(session_path, clu_file_path, res_file_path)
                                # print(spiketimes)
                                spiketimes = spiketimes/20e3 # divide the spiketimes by 20 kHz to get the spiketimes in s

                                # find cell types
                                cell_types = []
                                for c in cells:
                                    cell_type_ = get_cell_type(cell_type_info_df, topdir, ele, c)
                                    cell_types.append(cell_type_)

                                temp_data_df = pd.DataFrame({
                                    'ele' : [ele]*len(spiketimes),
                                    'cell' : cells,
                                    'spiketime' : spiketimes,
                                    'celltype' : cell_types,
                                    'region' : [region_name]*len(spiketimes)
                                })
                                temp_data_list.append(temp_data_df)

                                print(temp_data_df.iloc[-2:])
                            else:
                                pass

                        # concatenate all the temp lists into one dataframe
                        temp_data_df = pd.concat(temp_data_list, ignore_index=True)

                        # drop all the rows with nan in the type
                        temp_data_df = temp_data_df.dropna(subset=['celltype'])

                        ######################################################################################################################################
                        ## concat (electrode, cell) -> cell (a unique ID)

                        ## METHOD1: enumerate all the unique electrodes and cells in temp_data_df (DOESN'T WORK)
                        # unique_electrode = temp_data_df['ele'].unique()

                        # # iterate over all the electrodes to find cells
                        # temp_ec_list = []
                        # for e in unique_electrode:
                        #     c_arr = temp_data_df.loc[temp_data_df['ele'] == e]['cell'].unique()
                        #     ec_dict = {()}
                        #     temp_ec_df = pd.DataFrame(
                        #         {
                        #             'ele' : e*np.ones_like(c_arr),
                        #             'cell' : c_arr
                        #         }
                        #     )
                        #     temp_ec_list.append(temp_ec_df)
                        
                        # temp_ec_df = pd.concat(temp_ec_list, ignore_index=True)

                        # # create a sequential list for cell IDs
                        # ids_arr = np.arange(1, len(temp_ec_list) + 1)

                        # temp_ec_df['ids'] = ids_arr

                        ## METHOD 2: Using the metadata file to create the dictionary that maps the (ele, cell) --> id
                        ec_info_df = cell_type_info_df.loc[cell_type_info_df['topdir'] == topdir]
                        ids = np.arange(1, len(ec_info_df) + 1)

                        ec_dict = {(e, c) : id for e, c, id in zip(ec_info_df['ele'], ec_info_df['clu'], ids)}
                        print("-----------")
                        print(f"Total # neurons : {len(ec_dict)}")
                        print("-----------")




                        # # # create a dictionary that maps the (ele, cell) --> id
                        # # ec_dict = {(electrode, cell) : id for electrode, cell, id in zip(temp_ec_df['ele'].to_numpy(), temp_ec_df['cell'].to_numpy(), ids_arr)}
                        # cell_id_list = []
                        # for e, c in zip(temp_data_df['ele'].to_numpy(), temp_data_df['cell'].to_numpy()):
                        #     cell_id = (temp_ec_df.loc[temp_ec_df['ele'] == e]
                        #                .loc[temp_ec_df['cell'] == c]['ids']
                        #                )
                        #     cell_id = cell_id.to_numpy()
                        #     cell_id_list.append(cell_id)

                        # create the final dataframe with cell-spiketime-celltype-region data = pd.DataFrame(columns = ['cell', 'spiketime', 'celltype', 'region'])
                        data = pd.DataFrame(
                            {
                                'cell' : [ec_dict[(e, c)] for e, c in zip(temp_data_df['ele'], temp_data_df['cell'])],
                                'spiketime' : temp_data_df['spiketime'].to_numpy(),
                                'celltype' : temp_data_df['celltype'].to_numpy(),
                                'region' : temp_data_df['region'].to_numpy()
                            }
                        )

                        # save the dataframe in the session directory
                        filename = f"cell-spiketime-file-{session}.csv"
                        data_file_path = os.path.join(session_path, filename)
                        data.to_csv(data_file_path, index=False)

                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        print(f"Successfully saved: {filename}")
                        print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++")

                        # empty the temp_data_list
                        temp_data_list = []




                        






