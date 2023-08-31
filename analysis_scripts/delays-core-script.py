"""
This script generates a file delay_times_{region}.csv containing spiketime delays.
The delay_time arrays will be stored under: /nadata/cnl/data/Vikrant/hc3/processed_data_files


"""

import sys
import os
import numpy as np
import pandas as pd

##################################################################################################################

## defining the delay times
dt = 50e-3
t_delays = np.arange(-1, 1 + dt, dt)

## define a function to read the 