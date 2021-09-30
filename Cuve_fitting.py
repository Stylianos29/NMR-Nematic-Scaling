import pandas as pd
import ast
import pprint

from scipy.optimize import curve_fit

import numpy as np
import matplotlib.pyplot as plt

#Importing functions for curve fitting
import importlib.util
spec = importlib.util.spec_from_file_location("Fitting_functions", "NMR_Nematic_Scaling\Fitting_functions.py")
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
df = pd.read_csv(r'D:\Dropbox (Personal)\Purdue University\2020_C (Fall Term)\PHYS590 (NMR)\Python Programs\NMR_Nematic_Scaling\NMR_measurements\Raw Recovery Data x=0.05898 11.7T.csv')

#Importing the content of the Magnetization_recovery_dict dictionary to a text file
file = open("NMR_Nematic_Scaling\Output_Files\Text_files\Magnetization_recovery_pairs_dictionary.txt", "r")
contents = file.read()
Magnetization_recovery_dict = ast.literal_eval(contents)
file.close()
#Print statements
pprint.pprint(Magnetization_recovery_dict)

###################################################################################

fit_parameters_values_dict = {}
fit_parameters_error_dict = {}

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
    
    # Raw data
    # pars, cov = curve_fit(f=fit.stretched_recovery_function, xdata=x, ydata=y, p0=[max(y), 1-min(y)/max(y), 50, 0.5], bounds=(-np.inf, np.inf))
    # Normalized data
    pars, cov = curve_fit(f=fit.stretched_recovery_function, xdata=x, ydata=y, p0=[1, 0.5, 50, 0.5], bounds=(-np.inf, np.inf))
    
    # Standard deviation error for all the parameters
    stdevs = np.sqrt(np.diag(abs(cov)))
    res = np.sum((y - fit.stretched_recovery_function(x, *pars))**2)

    ax.plot(x, fit.stretched_recovery_function(x, *pars), linestyle=':', linewidth=2, color='red')#, label="fit"+str(list(pars))
    ax.set(xlabel='Recovery time (s)', ylabel='Normalized Magnetization Curve', title='Stretched Exponential fit on Magnetization Curve at T = '+str(temperature_value)+'K',xscale = "log")

    plt.legend(['Fit:'+' $W1^*$={:.2f}'.format(pars[2])+'$\pm${:.2f}'.format(stdevs[2])+' ($1/s$), β={:.2f}'.format(pars[3])+'$\pm${:.2f}'.format(stdevs[3])+',\n $M_0$={:.2f}'.format(pars[0])+', $\phi$={:.2f}'.format(pars[1])+", RSS={:.2e}".format(res), 'NMR data'], loc="upper left")
    # plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Fits_LogLinear_Plots\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')
    # Save normalized data
    # plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Fits_LogLinear_Plots_Normalized\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')

    fit_parameters_values_dict[temperature_value] = pars
    fit_parameters_error_dict[temperature_value] = stdevs

pprint.pprint(fit_parameters_values_dict)

###################################################################################

list_of_alpha_values = [ fit_parameters_values_dict[j][2] for j in list(fit_parameters_values_dict.keys()) ]
list_of_alpha_error = [ fit_parameters_error_dict[j][2] for j in list(fit_parameters_error_dict.keys()) ]
list_of_beta_values = [ fit_parameters_values_dict[j][3] for j in list(fit_parameters_values_dict.keys()) ]
list_of_beta_error = [ fit_parameters_error_dict[j][3] for j in list(fit_parameters_error_dict.keys()) ]
temperature_values = list(Magnetization_recovery_dict.keys())

slope_at_inflection = np.array([0.193161, 0.167753, 0.223586, 0.231432, 0.197833, 0.229112, 0.232296, 0.17893, 0.232366, 0.169364, 0.19182, 0.190065, 0.187226, 0.239567, 0.23781, 0.27536, 0.247891, 0.25687, 0.261396, 0.266929, 0.260024, 0.257597, 0.259894, 0.263925, 0.268179, 0.272231, 0.292903, 0.319857, 0.363608, 0.382399, 0.40175, 0.634582, 0.441697, 0.483477, 0.50374, 0.491276, 0.542637, 0.570584, 0.563728, 0.573509])

fig1, ax1 = plt.subplots()

ax1.scatter(np.array(temperature_values), np.array(list_of_alpha_values), s=20, color='red', marker=".")
# ax1.scatter(np.array(temperature_values), np.array(list_of_beta_values), s=20, color='red', marker=".")
# ax1.scatter(np.array(temperature_values), 1/np.array(list_of_alpha_values), s=20, color='red', marker=".")

ax1.grid()
ax1.set_title('Stretched exponential $W_1^*$ values Vs. Temperature', pad=15)
# ax1.set_title('Stretched exponential $\\beta$ values Vs. Temperature', pad=15)
# ax1.set_title('Stretched exponential $T_1^*$ values Vs. Temperature', pad=15)

ax1.set(xlabel='Temperature (K)', ylabel='$W_1^*$ values (1/s)')
# ax1.set(xlabel='Temperature (K)', ylabel='$\\beta$ values')
# ax1.set(xlabel='Temperature (K)', ylabel='$T_1^*$ values (s)')

# for i, txt in enumerate(list_of_beta_values):
#     plt.annotate(txt, (temperature_values[i], list_of_alpha_values[i]))

# Loop for annotation of all points with beta values
for i in range(len(temperature_values)):
    plt.annotate('{:.2f}'.format(list_of_beta_values[i]), (temperature_values[i], list_of_alpha_values[i] *1.10))

plt.errorbar(temperature_values, list_of_alpha_values, yerr=list_of_alpha_error, fmt=".", markersize=5, capsize=3, color='red')
# plt.errorbar(temperature_values, list_of_beta_values, yerr=list_of_beta_error, fmt=".", markersize=5, capsize=3, color='red')#fmt=".",
# plt.errorbar(temperature_values, 1/np.array(list_of_alpha_values), yerr=np.array(list_of_alpha_error)/(np.array(list_of_alpha_values))**2, fmt=".", markersize=5, capsize=3, color='red')

# ax1.set_yscale('log')
# ax1.set_xscale('log')

plt.xlim([-10, 310])
plt.ylim([-5, 130])

# plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_W1_v_Temperature.png')
plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_W1_with_beta_values_v_Temperature.png')
# plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_beta_values_v_Temperature_up_to_50K.png')
# plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_T1_values_v_Temperature.png')

plt.show()

###################################################################################
