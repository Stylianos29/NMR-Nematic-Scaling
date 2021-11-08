# DESCRIPTION
# This program creates as output for each input file the following:
# 1. 

########################################################################
# DEPENDENCIES

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import os
import ast

########################################################################
# USER INPUT

# Experimental values of parameters
gyromagnetic_ratio = 7.2919e6 #Hz/Teslas
magnetic_field = 11.7 #Teslas
omega_Larmor = gyromagnetic_ratio * magnetic_field
tauC_arrhenius = 1e-15
tauC_resonance = 1/omega_Larmor
time_scale = 1e-9 #nano-seconds

number_of_bins_distribution = 1000000
number_of_scanning_values = 2

# Exploration ranges
critical_exponent_min = 0.01
critical_exponent_max = 10

# seconds
tauC_lower_cut_off_min = tauC_arrhenius
tauC_lower_cut_off_max = tauC_resonance

tauC_upper_cut_off_min = 2*tauC_resonance
tauC_upper_cut_off_max = 1e2

# Hz
W1maxTauC_min = 100
W1maxTauC_max = 250

########################################################################
# FUNCTIONS DEFINITIONS

# Constructing the scanning ranges
def scanning_parameter_range_values(lower_bound, upper_bound,
                                                            number_of_values):
    parameter_range_intervals = np.abs(upper_bound - lower_bound)/number_of_values
    list_of_values = [lower_bound + parameter_range_intervals*(k + 1/2)
                                            for k in range(number_of_values)]
    return list_of_values

# Power law distribution
def tauC_distribution_function(tauC, *distribution_arguments):
    lower_cut_off = distribution_arguments[0]
    exponent = distribution_arguments[1]
    return tauC**(-exponent)

# Histogram costruction
def histogram_bins(lower_bound, upper_bound, number_of_bins):
    bins_width_array = np.array([(number_of_bins/(number_of_bins-1)
        )**j for j in range(number_of_bins) ])
    bins_width_array = bins_width_array*(
        upper_bound - lower_bound)/np.sum(bins_width_array)
    bins_array = [lower_bound + bins_width_array[0]/2]
    for k in range(1,number_of_bins):
        bins_array.append(bins_array[k-1] + bins_width_array[k-1]/2
            + bins_width_array[k]/2)
    bins_array = np.array(bins_array)
    return bins_array, bins_width_array

def histogram_distribution_values(histogram_bins, distribution_function,
                                            *distribution_function_arguments):
    return distribution_function(histogram_bins,
                                            *distribution_function_arguments)

def distribution_integral(histogram_bins_array, histogram_values_array):
    histogram_bins_array = np.array(histogram_bins_array)
    histogram_values_array = np.array(histogram_values_array)
    return np.inner(histogram_bins_array, histogram_values_array)

# Symbolic relations
def W1_relation_tauC(tauC, W1maxTauC):
    return 2*W1maxTauC * (omega_Larmor*tauC)/(1+(omega_Larmor*tauC)**2)

def tauC_relation_W1_minus(W1, W1maxTauC):
    ratio_W1 = W1maxTauC/W1
    return ( ratio_W1 - np.sqrt(ratio_W1**2-1) )/omega_Larmor
    
def tauC_relation_W1_plus(W1, W1maxTauC):
    ratio_W1 = W1maxTauC/W1
    return ( ratio_W1 + np.sqrt(ratio_W1**2-1) )/omega_Larmor

def W1_distribution_from_tauC_minus(W1, upper_bound,
                              *tauC_distribution_function_arguments):
    ratio = upper_bound/W1
    minus_branch = ( tauC_distribution_function(tauC_relation_W1_minus(
        W1, upper_bound), *tauC_distribution_function_arguments)
             * (1/(omega_Larmor*upper_bound)) *
                                (- ratio**2 + ratio**3/np.sqrt(ratio**2-1)) )    
    return minus_branch

def W1_distribution_from_tauC_plus(W1, upper_bound,
                              *tauC_distribution_function_arguments):
    ratio = upper_bound/W1
    plus_branch = ( tauC_distribution_function(tauC_relation_W1_plus(
        W1, upper_bound), *tauC_distribution_function_arguments)
                * (1/(omega_Larmor*upper_bound)) *
                                    (ratio**2 + ratio**3/np.sqrt(ratio**2-1)) )
    return plus_branch

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

# Magnetization recovery calculation
# # Stretched exponential recovery function for I=1/2
def recovery_function(x):
    #this partial sum is used to avoid overflow issues
    partial_sum = 0.9 * np.exp(-6*x) + 0.1 * np.exp(-x)
    return 1 - partial_sum

# # Magnetization integral definition
def magnetization_integral(W1_histogram_bins, W1_histogram_bins_width,
                                        W1_histogram_values, recovery_time):
    return np.sum(W1_histogram_values*recovery_function(
        W1_histogram_bins*recovery_time)*W1_histogram_bins_width)

########################################################################
# INITIALIZATION

# Initializing the output dictionary
output_dictionary = {}

# Creating the scanning parameter ranges
critical_exponent_range = scanning_parameter_range_values(
    critical_exponent_min, critical_exponent_max, number_of_scanning_values)
tauC_lower_cut_off_range = scanning_parameter_range_values(
    tauC_lower_cut_off_min, tauC_lower_cut_off_max, number_of_scanning_values)
tauC_upper_cut_off_range = scanning_parameter_range_values(
    tauC_upper_cut_off_min, tauC_upper_cut_off_max, number_of_scanning_values)
W1maxTauC_range = scanning_parameter_range_values(
    W1maxTauC_min, W1maxTauC_max, number_of_scanning_values)

# #initial values
# critical_exponent = 0.9 #seconds
# tauC_lower_bound = 0.5*tauC_resonance
# tauC_upper_bound = 0.1 #seconds
# W1maxTauC = 120 #Hz

for critical_exponent in critical_exponent_range:
    for tauC_lower_bound in tauC_lower_cut_off_range:
        for tauC_upper_bound in tauC_upper_cut_off_range:
            for W1maxTauC in W1maxTauC_range:

########################################################################
# DISTRIBUTION OF TAU_C VALUES

                tauC_histogram_bins, tauC_histogram_bins_width = histogram_bins(
                    tauC_lower_bound, tauC_upper_bound, number_of_bins_distribution)

                tauC_distribution_histogram = histogram_distribution_values(
                    tauC_histogram_bins, tauC_distribution_function,
                        tauC_lower_bound, critical_exponent)

                tauC_distribution_integral_initial = distribution_integral(
                                        tauC_histogram_bins_width, tauC_distribution_histogram)

                tauC_distribution_histogram = (tauC_distribution_histogram
                                                        /tauC_distribution_integral_initial)

########################################################################
# DISTRIBUTION OF THE RELAXATION RATES

                tauC_intermediate = tauC_relation_W1_plus( W1_relation_tauC(
                    tauC_histogram_bins[0], W1maxTauC) , W1maxTauC)
                tauC_intermediate_index = (len(tauC_histogram_bins)
                -len(tauC_histogram_bins[tauC_histogram_bins > tauC_intermediate]))
                tauC_distribution_integral_lower = distribution_integral(
                    tauC_histogram_bins_width[tauC_intermediate_index:],
                    tauC_distribution_histogram[tauC_intermediate_index:])
                tauC_distribution_integral_upper = distribution_integral(
                    tauC_histogram_bins_width[:tauC_intermediate_index],
                    tauC_distribution_histogram[:tauC_intermediate_index])

                W1_lower_bound_minus = W1_relation_tauC(tauC_histogram_bins[0], W1maxTauC)
                W1_lower_bound_plus = W1_relation_tauC(tauC_histogram_bins[-1], W1maxTauC)

                W1_intermediate = max(W1_lower_bound_minus, W1_lower_bound_plus)
                W1_lower_bound = min(W1_lower_bound_minus, W1_lower_bound_plus)

                W1_histogram_bins, W1_histogram_bins_width = histogram_bins(W1_lower_bound,
                    W1maxTauC, number_of_bins_distribution)

                W1_distribution_histogram_lower = histogram_distribution_values(
                    W1_histogram_bins[W1_histogram_bins < W1_intermediate],
                    W1_distribution_from_tauC_plus, W1maxTauC,
                        tauC_lower_bound, critical_exponent)

                W1_distribution_histogram_upper_minus = histogram_distribution_values(
                    W1_histogram_bins[W1_histogram_bins > W1_intermediate],
                    W1_distribution_from_tauC_minus, W1maxTauC,
                        tauC_lower_bound, critical_exponent)

                W1_distribution_histogram_upper_plus = histogram_distribution_values(
                    W1_histogram_bins[W1_histogram_bins > W1_intermediate],
                    W1_distribution_from_tauC_plus, W1maxTauC,
                        tauC_lower_bound, critical_exponent)

                W1_distribution_histogram_upper = (W1_distribution_histogram_upper_plus
                                                + W1_distribution_histogram_upper_minus)

                W1_distribution_histogram_lower_integral = distribution_integral(
                    W1_histogram_bins_width[W1_histogram_bins < W1_intermediate],
                        W1_distribution_histogram_lower)
                W1_distribution_histogram_upper_integral = distribution_integral(
                    W1_histogram_bins_width[W1_histogram_bins > W1_intermediate],
                        W1_distribution_histogram_upper)

                W1_distribution_histogram_lower = (W1_distribution_histogram_lower
                    *tauC_distribution_integral_lower/W1_distribution_histogram_lower_integral)
                W1_distribution_histogram_upper = (W1_distribution_histogram_upper
                    *tauC_distribution_integral_upper/W1_distribution_histogram_upper_integral)

                W1_distribution_histogram = np.concatenate([W1_distribution_histogram_lower,
                                                            W1_distribution_histogram_upper])

                W1_distribution_integral_initial = distribution_integral(
                                        W1_histogram_bins_width, W1_distribution_histogram)
                W1_distribution_histogram = (W1_distribution_histogram
                                                        /W1_distribution_integral_initial)

########################################################################
# COMPARING NUMERICAL VALUES WITH ACTUAL ONES

                recovery_times = np.array( [ (0.25+0.25*np.mod(k,4))*pow(
                    10.0,np.floor_divide(k,4)) for k in range(-40,40) ])

                # Calculating numerically the recovery magnetization values
                magnetization_curve_numerical = [ magnetization_integral(W1_histogram_bins,
                    W1_histogram_bins_width, W1_distribution_histogram, recovery_time)
                        for recovery_time in recovery_times]

                # # plotting
                # fig, ax = plt.subplots()
                # ax.scatter(recovery_times, magnetization_curve_numerical, s=20, color='red',
                #                                                                     marker="x")
                # ax.grid()
                # ax.set_xscale('log')
                # ax.set_title('Magnetization Recovery Curve', pad=15)
                # ax.set(xlabel='Recovery time (s)', ylabel='Normalized recovery magnetization')
                # plt.show()

########################################################################
# COLLECTING DATA

                output_dictionary_key = (tauC_lower_bound, tauC_upper_bound,
                                                                critical_exponent, W1maxTauC)

                output_dictionary[output_dictionary_key] = magnetization_curve_numerical

########################################################################
# EXPORTING DATA

# output_directory = os.getcwd()
# output_filename = "Parameters_scan_magnetization_recovery_from_power_law.csv"
# recovery_data_dataframe = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in
#                                     output_dictionary.items() ]))
# recovery_data_dataframe.to_csv(os.path.join(output_directory,
#                                 output_filename), index = False, header=True)
