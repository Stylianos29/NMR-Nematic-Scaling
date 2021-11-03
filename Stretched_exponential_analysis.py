import pandas as pd
import ast
import pprint
import shutil
import os

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

########################################################################
# FUNCTIONS DEFINITIONS

#Stretched exponential recovery function for I=1/2
def stretched_recovery_function(x, M0, phi, alpha, beta):
    return ( M0*(1 - 2*phi*( (9/10)*np.exp(-pow((6*alpha*x), beta) ) -
                                (1/10)*np.exp(-pow((alpha*x), beta) ) ) ) )

# Plotting
def plot_histogram(histogram_bins, histogram_values, histogram_title = '',
        x_axis_label = '', y_axis_label = '', x_axis_scale = 'linear',
            y_axis_scale = 'linear'):
    fig, ax = plt.subplots()
    ax.scatter(histogram_bins, histogram_values, marker=".")
    ax.grid()
    ax.set_xscale(x_axis_scale)
    ax.set_yscale(y_axis_scale)
    ax.set_title(histogram_title, pad=15)
    ax.set(xlabel = x_axis_label, ylabel = y_axis_label)
    plt.show()

########################################################################
# IMPORT DATA

# *TASK*: Loop over all processed data directories

# Paths formed in a platform-independent way
NMR_processed_data_directory_relative_path = os.path.join(
    "NMR_Nematic_Scaling", "Output_Files", "Processed_raw_data",
                                    "Processed_raw_data_x=0.05898_H=11.7T")
# Full paths are specified user-independently as well
current_working_directory = os.getcwd()
NMR_processed_data_output_directory_full_path = os.path.join(
        current_working_directory, NMR_processed_data_directory_relative_path)

# Importing the sorted magnetization recovery data to a Pandas DataFrame
magnetization_recovery_sorted_data = pd.read_csv(
    os.path.join(NMR_processed_data_output_directory_full_path,
                                    "Sorted_raw_data_x=0.05898_H=11.7T.csv"))

# Importing the normalized magnetization recovery data to a Pandas DataFrame
magnetization_recovery_normalized_data = pd.read_csv(
    os.path.join(NMR_processed_data_output_directory_full_path,
                                "Normalized_raw_data_x=0.05898_H=11.7T.csv"))

# Importing the Magnetization recovery dictionary to a text file
magnetization_pairs_dictionary_full_path = os.path.join(
                    NMR_processed_data_output_directory_full_path,
                        "Recovery_magnetization-time_pairs_"+
                            "per_temperature_dictionary_x=0.05898_H=11.7T.txt")
magnetization_pairs_dictionary_text_file = open(
                            magnetization_pairs_dictionary_full_path, "r")
# Evaluating the text file content as a dictionary
magnetization_pairs_dictionary = ast.literal_eval(
                            magnetization_pairs_dictionary_text_file.read())
magnetization_pairs_dictionary_text_file.close()
pprint.pprint(magnetization_pairs_dictionary)

########################################################################

fit_parameters_for_sorted_data_dictionary = {}
fit_parameters_for_normalized_data_dictionary = {}


output_directory_relative_path = os.path.join("NMR_Nematic_Scaling",
                            "Output_Files", "Stretched_Exponential_Analysis")

output_directory_full_path = os.path.join(current_working_directory,
output_directory_relative_path, "Data_set_x=0.05898_H=11.7T")

if os.path.exists(output_directory_full_path):
    shutil.rmtree(output_directory_full_path)
os.makedirs(output_directory_full_path)

output_directory_full_path_sorted_plots = os.path.join(
        output_directory_full_path, 'Fits_LogLinear_Plots')
if os.path.exists(output_directory_full_path_sorted_plots):
    shutil.rmtree(output_directory_full_path_sorted_plots)
os.makedirs(output_directory_full_path_sorted_plots)


output_directory_full_path_normalized_plots = os.path.join(
        output_directory_full_path, 'Fits_LogLinear_Plots_normalized')
if os.path.exists(output_directory_full_path_normalized_plots):
    shutil.rmtree(output_directory_full_path_normalized_plots)
os.makedirs(output_directory_full_path_normalized_plots)


for temperature_value in magnetization_pairs_dictionary.keys():

    # *TASK:* create functions for DRY code

    ### Data sets for the specific temperature value
    # temperature_value = 25
    temperature_value = float(temperature_value)
    # Exctract the recovery times measurements without any 'nan' values
    recovery_times = np.array(magnetization_recovery_normalized_data[
                        magnetization_pairs_dictionary[temperature_value][0]])
    recovery_times = recovery_times[np.logical_not(np.isnan( recovery_times ))]

    # Exctract the recovery times measurements without any 'nan' values
    sorted_magnetization_curve = np.array(
                    magnetization_recovery_sorted_data[
                        magnetization_pairs_dictionary[temperature_value][1]])
    sorted_magnetization_curve = sorted_magnetization_curve[
                        np.logical_not(np.isnan(sorted_magnetization_curve))]

    # Exctract the recovery times measurements without any 'nan' values
    normalized_magnetization_curve = np.array(
        magnetization_recovery_normalized_data[magnetization_pairs_dictionary[
                                                        temperature_value][1]])
    normalized_magnetization_curve = normalized_magnetization_curve[
                    np.logical_not(np.isnan(normalized_magnetization_curve))]


    # Curve fitting a stretched exponential expresion
    fit_parameters_for_sorted_data, covariant_matrix_for_sorted_data = curve_fit(
        f = stretched_recovery_function, xdata = recovery_times,
            ydata = sorted_magnetization_curve,
                p0=[max(sorted_magnetization_curve), 0.6, 50, 0.5],
                    bounds=(-np.inf, np.inf))
    standard_deviation_values_for_sorted_data = np.sqrt(np.diag(abs(
                                                covariant_matrix_for_sorted_data)))

    fit_parameters_for_normalized_data, covariant_matrix_for_normalized_data = (
        curve_fit(f = stretched_recovery_function, xdata = recovery_times,
            ydata = normalized_magnetization_curve, p0=[1, 0.5, 50, 0.5],
                bounds=(-np.inf, np.inf)))
    standard_deviation_values_for_normalized_data = np.sqrt(np.diag(abs(
                                            covariant_matrix_for_normalized_data)))

    RSS_for_sorted_data = np.sum(
        (sorted_magnetization_curve - stretched_recovery_function(
            recovery_times, *fit_parameters_for_sorted_data))**2)
    RSS_for_normalized_data = np.sum(
        (normalized_magnetization_curve - stretched_recovery_function(
            recovery_times, *fit_parameters_for_normalized_data))**2)

    # numpy.vstack() will return an array of shape (2,4) and .T the tranpose
    fit_parameters_for_sorted_data_dictionary[temperature_value] = np.vstack(
        (fit_parameters_for_sorted_data,
                                standard_deviation_values_for_sorted_data)).T

    fit_parameters_for_normalized_data_dictionary[temperature_value] = np.vstack(
        (fit_parameters_for_normalized_data,
                                standard_deviation_values_for_normalized_data)).T

    # *TASK:* save plots with best fir line and parameter estimates with errors


    fig, ax = plt.subplots()
    ax.scatter(recovery_times, sorted_magnetization_curve, s=20,
                                                    color='black', label='Data')
    ax.plot(recovery_times, stretched_recovery_function(recovery_times,
    *fit_parameters_for_sorted_data), linestyle=':', linewidth=2, color='red')
    ax.set(xlabel='Recovery time (s)', ylabel='Recovery magnetization (arb. units)',
        title='Stretched Exponential fit on Recovery magnetization data at T = '
            +str(temperature_value)+'K')
    ax.set_xscale('log')
    plt.legend(['Fit curve' + ' (RSS={:.2e}'.format(RSS_for_sorted_data)+')'
        +'\n- $W1^*$={:.2f}'.format(fit_parameters_for_sorted_data[2])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_sorted_data[2])
        +' ($Hz$)'
        +',\n- β={:.2f}'.format(fit_parameters_for_sorted_data[3])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_sorted_data[3])
        +',\n- $M_0$={:.2f}'.format(fit_parameters_for_sorted_data[0])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_sorted_data[0])
        +',\n- $\phi$={:.2f}'.format(fit_parameters_for_sorted_data[1])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_sorted_data[1]),
            'NMR data'], loc= 'upper left')
    plot_filename = ('Magnetization_Recovery_Curve_T='+str(temperature_value)
                                    +'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')
    plt.savefig(os.path.join(output_directory_full_path_sorted_plots, plot_filename))
    # plt.show()


    fig, ax = plt.subplots()
    ax.scatter(recovery_times, normalized_magnetization_curve, s=20,
                                                    color='black', label='Data')
    ax.plot(recovery_times, stretched_recovery_function(recovery_times,
    *fit_parameters_for_normalized_data), linestyle=':', linewidth=2, color='red')
    ax.set(xlabel='Recovery time (s)', ylabel='Normalized recovery magnetization (arb. units)',
        title='Stretched Exponential fit on Recovery magnetization data at T = '
            +str(temperature_value)+'K')
    ax.set_xscale('log')
    plt.legend(['Fit curve' + ' (RSS={:.2e}'.format(RSS_for_normalized_data)+')'
        +'\n- $W1^*$={:.2f}'.format(fit_parameters_for_normalized_data[2])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[2])
        +' ($Hz$)'
        +',\n- β={:.2f}'.format(fit_parameters_for_normalized_data[3])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[3])
        +',\n- $M_0$={:.2f}'.format(fit_parameters_for_normalized_data[0])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[0])
        +',\n- $\phi$={:.2f}'.format(fit_parameters_for_normalized_data[1])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[1]),
            'NMR data'], loc="upper left")
    plot_filename = ('Magnetization_Recovery_Curve_T='+str(temperature_value)
                                    +'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')
    plt.savefig(os.path.join(output_directory_full_path_normalized_plots, plot_filename))
    # plt.show()

###############################################################################

# pprint.pprint(fit_parameters_for_sorted_data_dictionary)
# pprint.pprint(fit_parameters_for_normalized_data_dictionary)

# Exporting the contents of the recovery pairs dictionary to
# a .txt file
fit_parameters_for_sorted_data_dictionary_filename = (
    "Fit_parameters_for_sorted_data_dictionary_x=0.05898_H=11.7T.txt")
with open(os.path.join(output_directory_full_path, fit_parameters_for_sorted_data_dictionary_filename ),'w') as data:
    data.write(
        str(fit_parameters_for_sorted_data_dictionary))

fit_parameters_for_normalized_data_dictionary_filename = (
    "Fit_parameters_for_normalized_data_dictionary_x=0.05898_H=11.7T.txt")
with open(os.path.join(output_directory_full_path, fit_parameters_for_normalized_data_dictionary_filename ),'w') as data:
    data.write(
        str(fit_parameters_for_normalized_data_dictionary))

    # print("Saving plot for T="+str(temperature_value))
    

    # ax.set(xlabel='Recovery time (s)', ylabel='Magnetization Curve', title='Stretched Exponential fit on Magnetization Curve at T = '+str(temperature_value)+'K',xscale = "log")

    # Save normalized data
    # plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Fits_LogLinear_Plots_Normalized\Magnetization_Recovery_Curve_T='+str(temperature_value)+'K_x=0.05898_H=11.7T_Curve_LogLinearPlot.png')


import matplotlib.cm as cm

# *TASK:* create plots of the parameter estimates vs. temperature

list_of_alpha_values = [ fit_parameters_values_dict[j][2] for j in list(fit_parameters_values_dict.keys()) ]
list_of_alpha_error = [ fit_parameters_error_dict[j][2] for j in list(fit_parameters_error_dict.keys()) ]
list_of_beta_values = [ fit_parameters_values_dict[j][3] for j in list(fit_parameters_values_dict.keys()) ]
list_of_beta_error = [ fit_parameters_error_dict[j][3] for j in list(fit_parameters_error_dict.keys()) ]
list_of_phi_values = [ fit_parameters_values_dict[j][1] for j in list(fit_parameters_values_dict.keys()) ]
list_of_phi_error = [ fit_parameters_error_dict[j][1] for j in list(fit_parameters_error_dict.keys()) ]
list_of_M0_values = [ fit_parameters_values_dict[j][0] for j in list(fit_parameters_values_dict.keys()) ]
list_of_M0_error = [ fit_parameters_error_dict[j][0] for j in list(fit_parameters_error_dict.keys()) ]


temperature_values = list(magnetization_pairs_dictionary.keys())
recovery_times = np.array([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1])
colors = cm.rainbow(np.linspace(0, 1, len(recovery_times)))

slope_at_inflection = np.array([0.193161, 0.167753, 0.223586, 0.231432, 0.197833, 0.229112, 0.232296, 0.17893, 0.232366, 0.169364, 0.19182, 0.190065, 0.187226, 0.239567, 0.23781, 0.27536, 0.247891, 0.25687, 0.261396, 0.266929, 0.260024, 0.257597, 0.259894, 0.263925, 0.268179, 0.272231, 0.292903, 0.319857, 0.363608, 0.382399, 0.40175, 0.634582, 0.441697, 0.483477, 0.50374, 0.491276, 0.542637, 0.570584, 0.563728, 0.573509])

fig1, ax1 = plt.subplots()

# ax1.scatter(np.array(temperature_values), np.array(list_of_alpha_values), s=20, color='red', marker=".")
# ax1.scatter(np.array(temperature_values), np.array(list_of_beta_values), s=20, color='red', marker=".")
# ax1.scatter(np.array(temperature_values), 1/np.array(list_of_alpha_values), s=20, color='red', marker=".")
# ax1.scatter(np.array(temperature_values), np.array(list_of_phi_values), s=20, color='red', marker=".")

for recovery_time, col in zip(recovery_times, colors):
    specific_magnetization = [ fit.stretched_recovery_function(recovery_time, *fit_parameters_values_dict[temperature_value]) for temperature_value in temperature_values ]
    ax1.scatter(np.array(temperature_values), np.array(specific_magnetization), color=col, marker=".")

plt.legend(recovery_times, loc="upper right", title = "Recovery times (s):")

ax1.grid()
# ax1.set_title('Stretched exponential $W_1^*$ values Vs. Temperature', pad=15)
# ax1.set_title('Stretched exponential $W_1^*$ values (with $\\beta$ values) Vs. Temperature', pad=15)
# ax1.set_title('Stretched exponential $\\beta$ values Vs. Temperature', pad=15)
# ax1.set_title('Stretched exponential $T_1^*$ values Vs. Temperature', pad=15)
# ax1.set_title('Stretched exponential $\phi$ values Vs. Temperature', pad=15)
# ax1.set_title('Stretched exponential $M_0$ values Vs. Temperature', pad=15)
ax1.set_title('Stretched exponential $M(t_i)$ values Vs. Temperature', pad=15)

# ax1.set(xlabel='Temperature (K)', ylabel='$W_1^*$ values (1/s)')
# ax1.set(xlabel='Temperature (K)', ylabel='$\\beta$ values')
# ax1.set(xlabel='Temperature (K)', ylabel='$T_1^*$ values (s)')
# ax1.set(xlabel='Temperature (K)', ylabel='$\phi$ values')
ax1.set(xlabel='Temperature (K)', ylabel='Normalized $M(t_i)$ values')

# for i, txt in enumerate(list_of_beta_values):
#     plt.annotate(txt, (temperature_values[i], list_of_alpha_values[i]))

# Loop for annotation of all points with beta values
# for i in range(len(temperature_values)):
#     plt.annotate('{:.2f}'.format(list_of_beta_values[i]), (temperature_values[i], list_of_alpha_values[i] *1.10))

# plt.errorbar(temperature_values, list_of_alpha_values, yerr=list_of_alpha_error, fmt=".", markersize=5, capsize=3, color='red')
# plt.errorbar(temperature_values, list_of_beta_values, yerr=list_of_beta_error, fmt=".", markersize=5, capsize=3, color='red')#fmt=".",
# plt.errorbar(temperature_values, 1/np.array(list_of_alpha_values), yerr=np.array(list_of_alpha_error)/(np.array(list_of_alpha_values))**2, fmt=".", markersize=5, capsize=3, color='red')

# plt.errorbar(temperature_values, list_of_M0_values, yerr=list_of_M0_error, fmt=".", markersize=5, capsize=3, color='red')

# ax1.set_yscale('log')
# ax1.set_xscale('log')

# plt.xlim([-10, 325])
# plt.ylim([-5, 140])

# plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_W1_v_Temperature.png')
# plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_W1_with_beta_values_v_Temperature.png')
# plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_beta_values_v_Temperature_up_to_50K.png')
# plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_T1_values_v_Temperature.png')
plt.savefig('NMR_Nematic_Scaling\Output_Files\Stretched_Exponential_Analysis\Stretched_exponential_M_for different_times_v_Temperature.png')

plt.show()

###################################################################################