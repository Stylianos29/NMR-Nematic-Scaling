# DESCRIPTION
# This program creates as output for each input file the following:
# 1. 

########################################################################
# DEPENDENCIES

import pandas as pd
import ast
import pprint
import shutil
import os
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
import re

########################################################################
# FUNCTIONS DEFINITIONS

# Stretched exponential recovery function for I=1/2
def stretched_recovery_function(x, M0, phi, alpha, beta):
    return ( M0*(1 - 2*phi*( 0.9*np.exp(-pow((6*alpha*x), beta) ) -
                                0.1*np.exp(-pow((alpha*x), beta) ) ) ) )

def RSS_calculation(magnetization_curve_data, numerical_magnetization_curve):
    return np.sum(
        (magnetization_curve_data - numerical_magnetization_curve)**2)

########################################################################
# LIST DATA SETS

NMR_processed_raw_data_relative_path = os.path.join("NMR_Nematic_Scaling",
    "Output_Files", "Processed_raw_data")
# Full paths are specified user-independently as well
current_working_directory = os.getcwd()
NMR_processed_raw_data_full_path = os.path.join(
    current_working_directory, NMR_processed_raw_data_relative_path)

output_directory_identifier_list = []
for NMR_processed_raw_data_dir in os.scandir(NMR_processed_raw_data_full_path):
    if (NMR_processed_raw_data_dir.is_dir()):
        identifying_parameters_tuple = tuple(
            [float(number) for number in re.findall(r'-?\d+\.?\d*',
                                            str(NMR_processed_raw_data_dir))])        
        output_directory_identifier_list.append( "x={}_H={}T".format(
            identifying_parameters_tuple[0], identifying_parameters_tuple[1]) )

output_directory_identifier = output_directory_identifier_list[0]

# *FUTURE TASK*: Loop over all processed data directories

########################################################################
# IMPORT DATA

# Paths formed in a platform-independent way
NMR_processed_data_directory_relative_path = os.path.join(
    "NMR_Nematic_Scaling", "Output_Files", "Processed_raw_data",
                            "Processed_raw_data_"+output_directory_identifier)

# Full paths are specified user-independently as well
# *FUTURE TASK*: get the full path of script's directory
current_working_directory = os.getcwd()
NMR_processed_data_output_directory_full_path = os.path.join(
        current_working_directory, NMR_processed_data_directory_relative_path)

# Importing the sorted magnetization recovery data to a Pandas DataFrame
magnetization_recovery_sorted_data = pd.read_csv(
    os.path.join(NMR_processed_data_output_directory_full_path,
                        "Sorted_raw_data_"+output_directory_identifier+".csv"))

# Importing the normalized magnetization recovery data to a Pandas DataFrame
magnetization_recovery_normalized_data = pd.read_csv(
    os.path.join(NMR_processed_data_output_directory_full_path,
                    "Normalized_raw_data_"+output_directory_identifier+".csv"))

# Importing the Magnetization recovery dictionary to a text file
magnetization_pairs_dictionary_full_path = os.path.join(
    NMR_processed_data_output_directory_full_path,
        "Recovery_magnetization-time_pairs_"+
            "per_temperature_dictionary_"+output_directory_identifier+".txt")

magnetization_pairs_dictionary_text_file = open(
                            magnetization_pairs_dictionary_full_path, "r")

# Evaluating the text file content as a dictionary
magnetization_pairs_dictionary = ast.literal_eval(
                            magnetization_pairs_dictionary_text_file.read())

magnetization_pairs_dictionary_text_file.close()
pprint.pprint(magnetization_pairs_dictionary)

########################################################################
# EXPORT DATA

fit_parameters_for_sorted_data_dictionary = {}

fit_parameters_for_normalized_data_dictionary = {}

output_directory_relative_path = os.path.join("NMR_Nematic_Scaling",
                            "Output_Files", "Stretched_Exponential_Analysis")

output_directory_full_path = os.path.join(current_working_directory,
    output_directory_relative_path, "Data_set_"+output_directory_identifier)

# Deleting any existing directories
if os.path.exists(output_directory_full_path):
    shutil.rmtree(output_directory_full_path)

# Making a new directory
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

########################################################################
# FITTING DATA

for temperature_value in magnetization_pairs_dictionary.keys():

    temperature_value = float(temperature_value)
    
    # Exctract the recovery times measurements without any 'nan' values
    recovery_times = np.array(magnetization_recovery_normalized_data[
                        magnetization_pairs_dictionary[temperature_value][0]])
    
    recovery_times = recovery_times[np.logical_not(np.isnan( recovery_times ))]

    # Exctract sorted recovery magnetization data excluding 'nan' values
    sorted_magnetization_curve = np.array(
                    magnetization_recovery_sorted_data[
                        magnetization_pairs_dictionary[temperature_value][1]])
    
    sorted_magnetization_curve = sorted_magnetization_curve[
                        np.logical_not(np.isnan(sorted_magnetization_curve))]

    # Exctract normalized recovery magnetization data excluding 'nan' values
    normalized_magnetization_curve = np.array(
        magnetization_recovery_normalized_data[magnetization_pairs_dictionary[
                                                        temperature_value][1]])

    normalized_magnetization_curve = normalized_magnetization_curve[
                    np.logical_not(np.isnan(normalized_magnetization_curve))]

    # Curve fitting a stretched exponential expresion on both sets of data
    fit_parameters_for_sorted_data, covariant_matrix_for_sorted_data = (
        curve_fit( stretched_recovery_function, xdata = recovery_times,
                    ydata = sorted_magnetization_curve,
                        p0=[max(sorted_magnetization_curve), 0.6, 50, 0.5],
                            bounds=(-np.inf, np.inf)))
    
    standard_deviation_values_for_sorted_data = np.sqrt(np.diag(abs(
                                            covariant_matrix_for_sorted_data)))

    fit_parameters_for_normalized_data, covariant_matrix_for_normalized_data =(
        curve_fit(stretched_recovery_function, xdata = recovery_times,
            ydata = normalized_magnetization_curve, p0=[1, 0.5, 50, 0.5],
                bounds=(-np.inf, np.inf)))
    
    standard_deviation_values_for_normalized_data = np.sqrt(np.diag(abs(
                                        covariant_matrix_for_normalized_data)))

    # Calculating the RSS estimate for both sets of data
    RSS_for_sorted_data = RSS_calculation(sorted_magnetization_curve,
        stretched_recovery_function(recovery_times,
            *fit_parameters_for_sorted_data))

    RSS_for_normalized_data = RSS_calculation(normalized_magnetization_curve,
        stretched_recovery_function(recovery_times,
            *fit_parameters_for_normalized_data))

    # numpy.vstack() will return an array of shape (2,4) and .T the tranpose
    fit_parameters_for_sorted_data_dictionary[temperature_value] = np.vstack(
        (fit_parameters_for_sorted_data,
                                standard_deviation_values_for_sorted_data)).T

    fit_parameters_for_normalized_data_dictionary[temperature_value] = (
        np.vstack((fit_parameters_for_normalized_data,
                            standard_deviation_values_for_normalized_data)).T)

    # Saving plots with best fit line and parameter estimates with errors
    fig, ax = plt.subplots()
    ax.scatter(recovery_times, sorted_magnetization_curve, s=20,
                                                color='black', label='Data')
    ax.plot(recovery_times, stretched_recovery_function(recovery_times,
        *fit_parameters_for_sorted_data), linestyle=':', linewidth=2,
            color='red')
    ax.set(xlabel='Recovery time (s)',
            ylabel='Recovery magnetization (arb. units)',
                title='Stretched Exponential fit on Recovery '+
                    'magnetization data at T = '+str(temperature_value)+'K')
    ax.set_xscale('log')
    plt.legend(['Fit curve' + ' (RSS={:.2e}'.format(RSS_for_sorted_data)+')'
        +'\n- $W_1^*$={:.2f}'.format(fit_parameters_for_sorted_data[2])
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
                    +'K'+output_directory_identifier+'Curve_LogLinearPlot.png')
    plt.savefig(os.path.join(output_directory_full_path_sorted_plots,
        plot_filename))


    fig, ax = plt.subplots()
    ax.scatter(recovery_times, normalized_magnetization_curve, s=20,
                                                color='black', label='Data')
    ax.plot(recovery_times, stretched_recovery_function(recovery_times,
        *fit_parameters_for_normalized_data), linestyle=':', linewidth=2,
            color='red')
    ax.set(xlabel='Recovery time (s)', ylabel='Normalized recovery '+
            'magnetization (arb. units)',
            title='Stretched Exponential fit on Recovery magnetization data'+
                'at T = '+str(temperature_value)+'K')
    ax.set_xscale('log')
    plt.legend(['Fit curve' + ' (RSS={:.3f}'.format(RSS_for_normalized_data)
        +')'+'\n- $W_1^*$={:.2f}'.format(fit_parameters_for_normalized_data[2])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[2])
        +' ($Hz$)'
        +',\n- β={:.2f}'.format(fit_parameters_for_normalized_data[3])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[3])
        +',\n- $M_0$={:.2f}'.format(fit_parameters_for_normalized_data[0])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[0])
        +',\n- $\phi$={:.2f}'.format(fit_parameters_for_normalized_data[1])
        +'$\pm${:.2f}'.format(standard_deviation_values_for_normalized_data[1])
            ,'NMR data'], loc="upper left")
    plot_filename = ('Magnetization_Recovery_Curve_T='+str(temperature_value)
                    +'K'+output_directory_identifier+'Curve_LogLinearPlot.png')
    plt.savefig(os.path.join(output_directory_full_path_normalized_plots,
        plot_filename))

###############################################################################
# EXPORTING PARAMETER VALUES

# Exporting the sorted recovery pairs dictionary to a text file
fit_parameters_for_sorted_data_dictionary_filename = (
    "Fit_parameters_for_sorted_data_dictionary"
        +output_directory_identifier+".txt")

fit_parameters_for_sorted_data_dictionary_full_path = os.path.join(
    output_directory_full_path,
        fit_parameters_for_sorted_data_dictionary_filename)

with open(fit_parameters_for_sorted_data_dictionary_full_path,'w') as data:
    data.write(str(fit_parameters_for_sorted_data_dictionary))

# Exporting the normalized recovery pairs dictionary to a text file
fit_parameters_for_normalized_data_dictionary_filename = (
    "Fit_parameters_for_normalized_data_dictionary"
        +output_directory_identifier+".txt")

fit_parameters_for_normalized_data_dictionary_full_path = os.path.join(
    output_directory_full_path,
        fit_parameters_for_normalized_data_dictionary_filename)

with open(fit_parameters_for_normalized_data_dictionary_full_path,'w') as data:
    data.write(str(fit_parameters_for_normalized_data_dictionary)) 

###############################################################################
# PARAMETER PLOTS

temperature_values = np.array(list(magnetization_pairs_dictionary.keys()))

fit_parameter_plot_horizontal_axis_label_list = [
    '$M_0$ values (Hz)',
    '$\\phi$ values (Hz)',
    '$W_1^*$ values (Hz)',
    '$\\beta$ values'
]

fit_parameter_plot_title_list = [
    'Stretched exponential $M_0$ values Vs. Temperature',
    'Stretched exponential $\\phi$ values Vs. Temperature',
    'Stretched exponential $W_1^*$ values Vs. Temperature',
    'Stretched exponential $\\beta$ values Vs. Temperature'
]

# Fit parameters for sorted data
sorted_data_fit_parameter_plot_filename_list = [
    'Stretched_exponential_M0_Vs_Temperature.png',
    'Stretched_exponential_phi_Vs_Temperature.png',
    'Stretched_exponential_W1_Vs_Temperature.png',
    'Stretched_exponential_beta_Vs_Temperature.png'
]

for fit_parameter_index in range(4):

    list_of_fit_parameter_values = np.array([
        fit_parameters_for_sorted_data_dictionary[j][fit_parameter_index][0]
            for j in list(fit_parameters_for_sorted_data_dictionary.keys()) ])

    list_of_fit_parameter_error_values = np.array([
        fit_parameters_for_sorted_data_dictionary[j][fit_parameter_index][1]
            for j in list(fit_parameters_for_sorted_data_dictionary.keys()) ])

    fig, ax = plt.subplots()
    ax.scatter(temperature_values, list_of_fit_parameter_values, s=20,
        color='red', marker=".")
    plt.errorbar(temperature_values, list_of_fit_parameter_values,
        yerr=list_of_fit_parameter_error_values, fmt=".", markersize=5,
            capsize=3, color='red')
    ax.set_title(fit_parameter_plot_title_list[fit_parameter_index], pad=15)
    ax.set(xlabel='Temperature (K)',
        ylabel=fit_parameter_plot_horizontal_axis_label_list[
            fit_parameter_index])
    ax.grid()

    plt.savefig(os.path.join(output_directory_full_path,
        sorted_data_fit_parameter_plot_filename_list[fit_parameter_index]))

# Fit parameters for normalized data
normalized_data_fit_parameter_plot_filename_list = [
    'Stretched_exponential_M0_Vs_Temperature_normalized_data.png',
    'Stretched_exponential_phi_Vs_Temperature_normalized_data.png',
    'Stretched_exponential_W1_Vs_Temperature_normalized_data.png',
    'Stretched_exponential_beta_Vs_Temperature_normalized_data.png'
]

for fit_parameter_index in range(4):

    list_of_fit_parameter_values = np.array([
        fit_parameters_for_normalized_data_dictionary[j][
                fit_parameter_index][0]
            for j in list(
                fit_parameters_for_normalized_data_dictionary.keys()) ])

    list_of_fit_parameter_error_values = np.array([
        fit_parameters_for_normalized_data_dictionary[j][
                fit_parameter_index][1]
            for j in list(
                fit_parameters_for_normalized_data_dictionary.keys()) ])

    fig, ax = plt.subplots()
    ax.scatter(temperature_values, list_of_fit_parameter_values, s=20,
        color='red', marker=".")
    plt.errorbar(temperature_values, list_of_fit_parameter_values,
        yerr=list_of_fit_parameter_error_values, fmt=".", markersize=5,
            capsize=3, color='red')
    ax.set_title(fit_parameter_plot_title_list[fit_parameter_index], pad=15)
    ax.set(xlabel='Temperature (K)',
        ylabel=fit_parameter_plot_horizontal_axis_label_list[
            fit_parameter_index])
    ax.grid()

    plt.savefig(os.path.join(output_directory_full_path,
        normalized_data_fit_parameter_plot_filename_list[fit_parameter_index]))

###############################################################################
# SPECIALIZED PARAMETER PLOTS

# W1* values plot annotated with the corresponding beta values
list_of_W1_star_values = np.array([
    fit_parameters_for_normalized_data_dictionary[j][2][0]
        for j in list(fit_parameters_for_normalized_data_dictionary.keys()) ])

list_of_W1_star_error_values = np.array([
    fit_parameters_for_normalized_data_dictionary[j][2][1]
        for j in list(fit_parameters_for_normalized_data_dictionary.keys()) ])

list_of_beta_values = np.array([
    fit_parameters_for_normalized_data_dictionary[j][3][0]
        for j in list(fit_parameters_for_normalized_data_dictionary.keys()) ])

fig, ax = plt.subplots()
ax.scatter(temperature_values, list_of_W1_star_values, s=20, color='red',
    marker=".")
plt.errorbar(temperature_values, list_of_W1_star_values,
    yerr=list_of_W1_star_error_values, fmt=".", markersize=5, capsize=3,
        color='red')
ax.set_title(fit_parameter_plot_title_list[2]
    +' (with $\\beta$ values)', pad=15)
ax.set(xlabel='Temperature (K)',
    ylabel=fit_parameter_plot_horizontal_axis_label_list[2])
ax.grid()

for i in range(len(temperature_values)):
    plt.annotate('{:.2f}'.format(list_of_beta_values[i]),
        (temperature_values[i], list_of_W1_star_values[i] *1.10))

plt.savefig(os.path.join(output_directory_full_path,
    'Stretched_exponential_W1_Vs_Temperature_'
        +'with_beta_values_normalized_data.png'))

plt.show()


# T1* values plot
list_of_T1_star_values = 1/np.array([
    fit_parameters_for_normalized_data_dictionary[j][2][0]
        for j in list(fit_parameters_for_normalized_data_dictionary.keys()) ])

list_of_W1_star_error_values = np.array([
    fit_parameters_for_normalized_data_dictionary[j][2][1]
        for j in list(fit_parameters_for_normalized_data_dictionary.keys()) ])

list_of_T1_star_error_values = (
    list_of_W1_star_error_values * (list_of_T1_star_values)**2)

list_of_beta_values = np.array([
    fit_parameters_for_normalized_data_dictionary[j][3][0]
        for j in list(fit_parameters_for_normalized_data_dictionary.keys()) ])

fig, ax = plt.subplots()
ax.scatter(temperature_values, list_of_T1_star_values, s=20, color='red',
    marker=".")
plt.errorbar(temperature_values, list_of_T1_star_values,
    yerr=list_of_T1_star_error_values, fmt=".", markersize=5, capsize=3,
        color='red')
ax.set_title('Stretched exponential $T_1^*$ values Vs. '+
    'Temperature (with $\\beta$ values)', pad=15)
ax.set(xlabel='Temperature (K)',
    ylabel='$T_1^*$ values (s)')
ax.set_yscale('log')
ax.grid()

for i in range(len(temperature_values)):
    plt.annotate('{:.2f}'.format(list_of_beta_values[i]),
        (temperature_values[i], list_of_T1_star_values[i] *1.10))

plt.savefig(os.path.join(output_directory_full_path,
    'Stretched_exponential_T1_Vs_Temperature_'
        +'with_beta_values_normalized_data.png'))

plt.show()

###############################################################################