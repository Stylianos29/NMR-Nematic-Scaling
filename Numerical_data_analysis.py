# DESCRIPTION:
# This program creates as output for each input file the following:
# 1. A recovery pairs dictionary with the temperature values as keys

import os
import pandas as pd
import pprint
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

# Input directory: All NMR raw data are stored in the 
# ".\NMR_Nematic_Scaling\NMR_measurements" directory, specified below
# in a platform-independent way
Numerical_data_relative_path = os.path.join("NMR_Nematic_Scaling",
                "Parameters_scan_magnetization_recovery_from_power_law.csv")
# Full paths are specified user-independently as well
current_working_directory = os.getcwd()
Numerical_data_full_path = os.path.join(
    current_working_directory, Numerical_data_relative_path)

# if os.path.isfile(Numerical_data_full_path):  
#     print("Yey, it is a file!") 

# INPUT DATA PROCESSING
# Passing the content of the .csv file to a Pandas DataFrame
Numerical_data_dataframe = pd.read_csv(
    Numerical_data_full_path, header = [0,1,2,3])

# Initialization of the recovery pairs dictionary
Numerical_data_header_dictionary = {}

# The header of each column of the raw data file contains
# the temperature value for each measurement and needs to be
# extrated initially in a list and then in a dictionary
# sorted necessarily in ascending order. That will be the
# recovery pairs dictionary.

list_header = list( Numerical_data_dataframe.columns)
pprint.pprint(list_header)

########################################################################
# COMPARING NUMERICAL VALUES WITH ACTUAL ONES

recovery_times = np.array( [ (0.25+0.25*np.mod(k,4))*pow(
                    10.0,np.floor_divide(k,4)) for k in range(-40,40) ])

number_of_columns = len(Numerical_data_dataframe.columns)

color = iter(cm.rainbow(np.linspace(0, 1, number_of_columns)))
   

# Calculating numerically the recovery magnetization values
fig, ax = plt.subplots()
ax.grid()
ax.set_xscale('log')
ax.set_title('Magnetization Recovery Curve', pad=15)
ax.set(xlabel='Recovery time (s)', ylabel='Normalized recovery magnetization')
ax.set_xlim(10**(-4),10**2)
for column, c in zip(Numerical_data_dataframe, color):
    # print(Numerical_data_dataframe[column])
    magnetization_curve_numerical = Numerical_data_dataframe[column]

    # plotting
    ax.scatter(recovery_times, magnetization_curve_numerical, s=20,
        color=c, marker=6)
plt.show()

########################################################################