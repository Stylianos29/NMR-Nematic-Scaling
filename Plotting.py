import pandas as pd

import ast
import pprint

import numpy as np
import matplotlib.pyplot as plt

###################################################################################
#TEMPORARY

#Passing the content of the .csv file to a Pandas DataFrame
df = pd.read_csv(r'D:\Dropbox (Personal)\Purdue University\2020_C (Fall Term)\PHYS590 (NMR)\Python Programs\NMR-Nematic-Scaling\NMR_measurements\Raw Recovery Data x=0.05898 11.7T.csv')

#Importing the content of the Magnetization_recovery_dict dictionary to a text file
file = open("NMR-Nematic-Scaling\Output_Files\Text_files\Magnetization_recovery_pairs_dictionary.txt", "r")
contents = file.read()
Magnetization_recovery_dict = ast.literal_eval(contents)
file.close()
#Print statements
pprint.pprint(Magnetization_recovery_dict)

###################################################################################

for temperature_value in Magnetization_recovery_dict.keys():
    x = np.array(df[Magnetization_recovery_dict[temperature_value][0]])
    x = x[np.logical_not(np.isnan(x))]
    y = np.array(df[Magnetization_recovery_dict[temperature_value][1]])
    y = y[np.logical_not(np.isnan(y))]

    #Making sure that all time recovery values are in ascenting order
    arr1inds = x.argsort()
    x = x[arr1inds[::1]]
    y = y[arr1inds[::1]]

    fig1, ax1 = plt.subplots()
    ax1.plot(x, y, 'o', color='black')
    ax1.set(xlabel='Recovery time (s)', ylabel='Magnetization', title='Magnetization Curve at T = '+str(temperature_value)+'K')

    plt.savefig('NMR-Nematic-Scaling\Output_Files\Linear_Plots\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LinearPlot.png')

    ax1.set_xscale('log')
    plt.savefig('NMR-Nematic-Scaling\Output_Files\LogLinear_Plots\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')

    #Normalizing the y values
    y_norm = ( y - min(y) ) / ( max(y) - min(y) )

    fig2, ax2 = plt.subplots()
    ax2.plot(x, y_norm, 'o', color='black')
    ax2.set(xlabel='Recovery time (s)', ylabel='Magnetization', title='Magnetization Curve at T = '+str(temperature_value)+'K')
    plt.savefig('NMR-Nematic-Scaling\Output_Files\Linear_Plots_Normalized\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')

    ax2.set_xscale('log')
    plt.savefig('NMR-Nematic-Scaling\Output_Files\LogLinear_Plots_Normalized\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')

###################################################################################

fig, ax = plt.subplots()

for temperature_value in Magnetization_recovery_dict.keys():
    x = np.array(df[Magnetization_recovery_dict[temperature_value][0]])
    x = x[np.logical_not(np.isnan(x))]
    y = np.array(df[Magnetization_recovery_dict[temperature_value][1]])
    y = y[np.logical_not(np.isnan(y))]

    #Making sure that all time recovery values are in ascenting order
    array_indices = x.argsort()
    x = x[array_indices[::1]]
    y = y[array_indices[::1]]

    ax.set_xscale('log')

    y_norm = ( y - min(y) ) / ( max(y) - min(y) )

    ax.plot(x, y_norm, 'o')#, color= 'black')
    ax.set(xlabel='Recovery time (s)', ylabel='Magnetization', title='Magnetization Curves')

    plt.savefig('NMR-Nematic-Scaling\Output_Files\Magnetization_Recovery_Curves_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')
