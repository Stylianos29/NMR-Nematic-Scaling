import pandas as pd
import ast
import pprint

from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt

#Importing functions for curve fitting
import importlib.util
spec = importlib.util.spec_from_file_location("Fitting_functions", "NMR-Nematic-Scaling\Fitting_functions.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)

import matplotlib.font_manager as fm
import pylab

# Edit the font, font size, and axes width
# plt.rcParams['font.family'] = 'Avenir'
# plt.rcParams['font.size'] = 18
# plt.rcParams['axes.linewidth'] = 2

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
    array_indices = x.argsort()
    x = x[array_indices[::1]]
    y = y[array_indices[::1]]

    #Normalized data
    y = ( y - min(y) ) / ( max(y) - min(y) )

    print("Saving plot for T="+str(temperature_value))
    fig, ax = plt.subplots()
    ax.scatter(x, y, s=20, color='black', label='Data')
    ax.set_xscale('log')
    # pars, cov = curve_fit(f=fit.exponential, xdata=x, ydata=y, p0=[max(y), 1-min(y)/max(y), 100], bounds=(-np.inf, np.inf))
    # stdevs = np.sqrt(np.diag(abs(cov)))
    # res = y - fit.exponential(x, pars[0], pars[1], pars[2])
    # ax.plot(x, fit.exponential(x, pars[0], pars[1], pars[2]), linestyle='--', linewidth=2, color='black')
    
    pars, cov = curve_fit(f=fit.reduced_stretched_exponential, xdata=x, ydata=y, p0=[100, 0.5], bounds=(-np.inf, np.inf))
    ax.plot(x, fit.reduced_stretched_exponential(x, *pars), linestyle=':', linewidth=2, color='red')
    
    ax.set(xlabel='Recovery time (s)', ylabel='Magnetization', title='Magnetization Curve at T = '+str(temperature_value)+'K',xscale = "log")
    ax.set_xlabel('Wavelength (nm)', labelpad=10)
    ax.set_ylabel('Absorbance (O.D.)', labelpad=10)
    plt.legend(['Stretched Exponential with β = {:.2f}'.format(pars[1])], loc="upper left")
    plt.savefig('NMR-Nematic-Scaling\Output_Files\Fits_LogLinear_Plots_Normalized\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')


