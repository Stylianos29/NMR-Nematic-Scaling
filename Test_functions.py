########################################################################
# Program description
# This program 

########################################################################

# Dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pprint
import os
import ast

########################################################################

# Experimental values of parameters
W1maxTauC = 120 #Hz
gyromagnetic_ratio = 7.2919e6 #Hz/Teslas
magnetic_field = 11.7 #Teslas
omega_Larmor = gyromagnetic_ratio * magnetic_field
tauC_resonance = 1/omega_Larmor
kB = 8.617333262145e-5 #units: eV/K

# User input
temperature_value = 25 # Critical temperature at about 25K
critical_exponent = 0.9
tauC_lower_bound = 0.1*tauC_resonance
W1max_safety_margin = 0.99*W1maxTauC
time_scale = 1e-9 #nano-seconds

# Number of bins for the W1 distribution histogram
number_of_bins_distribution = 10000 # FYI 100 bins are already many

print("Te passed lower bound for the τC values corresponds"+
    "to {:.2f}% the".format(100*tauC_lower_bound*omega_Larmor)
        +" resonance τC value.")

########################################################################
# FUNCTIONS DEFINITIONS

# Choice of a test function for the oscillation times

# def DeltaEC_distribution_function(DeltaEC, *args):
#     factor=args[0]
#     exponent=args[1]
#     return factor*DeltaEC**(-exponent)

# def DeltaEC_relation_tauC(tauC, tauC0):
#     return temperature_value*kB*np.log(tauC/tauC0)

# def tauC_distribution_function(tauC, *args):
#     tauC0 = args[]
#     DeltaEC = DeltaEC_relation_tauC(tauC, tauC0)
#     return (DeltaEC_distribution_function(DeltaEC, *args)
#                                 *(temperature_value*kB)/tauC)

def tauC_distribution_function(tauC, *args):
    factor=args[0]
    exponent=args[1]
    return factor*tauC**(-exponent)

# Symbolic relations
def W1_relation_tauC(tauC, W1maxTauC, omega_Larmor):
    return 2*W1maxTauC * (omega_Larmor*tauC)/(1+(omega_Larmor*tauC)**2)

def tauC_relation_W1_minus(W1, W1maxTauC, omega_Larmor):
    ratio_W1 = W1maxTauC/W1
    return ( ratio_W1 - np.sqrt(ratio_W1**2-1) )/omega_Larmor
    
def tauC_relation_W1_plus(W1, W1maxTauC, omega_Larmor):
    ratio_W1 = W1maxTauC/W1
    return ( ratio_W1 + np.sqrt(ratio_W1**2-1) )/omega_Larmor

def W1_distribution_from_tauC(W1, upper_bound,
                                        *tauC_distribution_function_arguments):
    ratio = upper_bound/W1
    minus_branch = ( tauC_distribution_function(tauC_relation_W1_minus(
        W1, upper_bound, omega_Larmor), *tauC_distribution_function_arguments)
             * (1/(omega_Larmor*upper_bound)) *
                                (- ratio**2 + ratio**3/np.sqrt(ratio**2-1)) )
    plus_branch = ( tauC_distribution_function(tauC_relation_W1_plus(
        W1, upper_bound, omega_Larmor), *tauC_distribution_function_arguments)
                * (1/(omega_Larmor*upper_bound)) *
                                    (ratio**2 + ratio**3/np.sqrt(ratio**2-1)) )
    return minus_branch + plus_branch

# Distribution construction
def histogram_bins(lower_bound, upper_bound, number_of_bins):
    bin_width = (upper_bound - lower_bound)/number_of_bins
    bins_array = np.array( [ ( lower_bound + bin_width * (j + 1/2) )
                                            for j in range(number_of_bins) ] )
    return bins_array

def histogram_distribution_values(histogram_bins, distribution_function,
                                            *distribution_function_arguments):
    bins_width = (max(histogram_bins) - min(histogram_bins))/(
                                                        len(histogram_bins)-1)
    histogram_bins = np.array(histogram_bins)
    non_normalized_distribution_values = distribution_function(histogram_bins,
                                            *distribution_function_arguments)
    integral = np.sum( non_normalized_distribution_values)*bins_width
    return non_normalized_distribution_values/integral

def normalization_check(histogram_bins_array, histogram_values_array):
    histogram_bins_array = np.array(histogram_bins_array)
    histogram_values_array = np.array(histogram_values_array)
    bin_width = (max(histogram_bins_array) - min(histogram_bins_array))/(
                                                len(histogram_bins_array)-1)
    return np.sum(histogram_values_array)*bin_width

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

# Magnetization recovery function
def recovery_function(x):
    return 1 - 0.9 * np.exp(-6*x) - 0.1 * np.exp(-x)

# Magnetization integral definition
def magnetization_integral(W1_histogram_bins, W1_histogram_values,
                                                                recovery_time):
    bins_width = (max(W1_histogram_bins) - min(W1_histogram_bins))/(
                                                    len(W1_histogram_bins)-1)
    return np.sum(W1_histogram_values*recovery_function(W1_histogram_bins
                                                    *recovery_time))*bins_width

def RSS_calculation(numerical_data, real_data):
    return np.sum( (numerical_data - real_data)**2 )

######################################################################
# RELATION BETWEEN tauC AND W1

W1_lower_bound = W1_relation_tauC(tauC_lower_bound, W1maxTauC, omega_Larmor)
tauC_upper_bound = tauC_relation_W1_plus(W1_lower_bound, W1maxTauC,
                                                                omega_Larmor)

tauC_histogram_bins = histogram_bins(tauC_lower_bound, tauC_upper_bound, 
                                                number_of_bins_distribution)

W1_values = np.array(W1_relation_tauC(tauC_histogram_bins,
                                                W1maxTauC, omega_Larmor))

plot_histogram(tauC_histogram_bins/time_scale, W1_values,
    histogram_title = 'Relation beetween $W_1$ and $\\tau_C$',
        x_axis_label = '${\\tau}_C$ (ns)', y_axis_label = '$W_1$ (Hz)')

########################################################################
# DISTRIBUTION OF THE FLIPPING TIMES

tauC_distribution_histogram = histogram_distribution_values(
        tauC_histogram_bins, tauC_distribution_function, 1, critical_exponent)

normalization_check(tauC_histogram_bins, tauC_distribution_histogram)

plot_histogram(tauC_histogram_bins, tauC_distribution_histogram,
histogram_title = 'Logarithmic plot of the Distribution of flipping times',
        x_axis_label = '${\\tau}_C$ (s)',
            y_axis_label = 'D(${\\tau}_C$) (Hz)', x_axis_scale = 'log',
                y_axis_scale = 'log')

########################################################################
# DISTRIBUTION OF RELAXATION RATES

W1_histogram_bins = histogram_bins(W1_lower_bound, W1max_safety_margin, 
                                                number_of_bins_distribution)

W1_distribution_histogram = histogram_distribution_values(W1_histogram_bins,
                W1_distribution_from_tauC, W1maxTauC, 1, critical_exponent)

normalization_check(W1_histogram_bins, W1_distribution_histogram)

plot_histogram(W1_histogram_bins, W1_distribution_histogram,
    histogram_title = 'Distribution of the relaxation rates',
        x_axis_label = '$W_1$ (Hz)', y_axis_label = 'D($W_1$) (s)')

########################################################################
# IMPORT ACTUAL DATA

# ".\NMR_Nematic_Scaling\NMR_measurements" directory, specified below
# in a platform-independent way
NMR_processed_data_relative_path = os.path.join("NMR_Nematic_Scaling",
"Output_Files", "Processed_raw_data", "Processed_raw_data_x=0.05898_H=11.7T")
# Full paths are specified user-independently as well
current_working_directory = os.getcwd()
NMR_processed_data_output_full_path = os.path.join(current_working_directory,
                                            NMR_processed_data_relative_path)

# Passing the content of the .csv file to a Pandas DataFrame
magnetization_recovery_normalized_data = pd.read_csv(
    os.path.join(NMR_processed_data_output_full_path,
                                "Normalized_raw_data_x=0.05898_H=11.7T.csv") )

# Importing the Magnetization recovery dictionary to a text file
magnetization_recovery_text_file_full_path = os.path.join(
    NMR_processed_data_output_full_path, "Recovery_magnetization-time_pairs_"+
        "per_temperature_dictionary_x=0.05898_H=11.7T.txt")
magnetization_recovery_text_file = open(
    magnetization_recovery_text_file_full_path, "r")
magnetization_recovery_dictionary = magnetization_recovery_text_file.read()
magnetization_curve_dictionary = ast.literal_eval(
                                        magnetization_recovery_dictionary)
magnetization_recovery_text_file.close()
#Print statements
pprint.pprint(magnetization_curve_dictionary)

temperature_value = float(temperature_value)
recovery_times = np.array(magnetization_recovery_normalized_data[
                    magnetization_curve_dictionary[temperature_value][0]])
recovery_times = recovery_times[np.logical_not(np.isnan( recovery_times ))]
magnetization_curve = np.array(magnetization_recovery_normalized_data[
                    magnetization_curve_dictionary[temperature_value][1]])
magnetization_curve = magnetization_curve[np.logical_not(np.isnan(
                                                        magnetization_curve))]

# plot_histogram(recovery_times, magnetization_curve,
#     histogram_title = 'Magnetization recovery curve',
#         x_axis_label = '$Recovery time (s)',
#             y_axis_label = 'Recvery magnetization', x_axis_scale='log')

########################################################################

# Calculating numerically the recovery magnetization values
magnetization_curve_numerical = [ magnetization_integral(W1_histogram_bins,
                                W1_distribution_histogram, recovery_time)
                                        for recovery_time in recovery_times]

RSS = RSS_calculation(magnetization_curve_numerical, magnetization_curve)

########################################################################
# Compare numerical values with actual ones

fig, ax = plt.subplots()

ax.scatter(recovery_times, magnetization_curve, s=20, color='black',
                                                                    marker="*")
ax.scatter(recovery_times, magnetization_curve_numerical, s=20, color='red',
                                                                    marker=".")

ax.grid()
ax.set_xscale('log')

ax.set_title('Magnetization Recovery Curve', pad=15)
ax.set(xlabel='Recovery time (s)', ylabel='Normalized recovery magnetization')

ax.legend(['NMR data', 'Numerical data (RSS={:.2f})'.format(RSS)],
                                                            loc='upper left')
plt.show()

########################################################################
# Invastigate optimal parameters

#input
N = 30
critical_exponent_range = 3
W1_lower_bound_range = 0.7*W1maxTauC
if W1_lower_bound_range>W1maxTauC:
    W1_lower_bound_range = 0.99*W1maxTauC

# Construct two arrays with the W1min and critical exponent values to be
# investigated make sure that the invastigated ranges starts above W1min = 0
RSS_values_array = np.zeros((N,N))
if (W1_lower_bound - 0.5*W1_lower_bound_range) <= 0:
    W1_lower_bound_min = 0.5*(W1_lower_bound_range/N)
else:
    W1_lower_bound_min = W1_lower_bound - 0.5*W1_lower_bound_range
W1_range_array = np.array( [ (W1_lower_bound_min + (W1_lower_bound_range/N)*j)
                                                        for j in range(N) ] )
#make sure that the invastigated ranges starts above critical_exponent = 0
if critical_exponent - critical_exponent_range/2 <= 0:
    critical_exponent_min = 0.5*(critical_exponent_range/N)
else:
    critical_exponent_min = critical_exponent - critical_exponent_range/2
critical_exponent_array = np.array( [ ( critical_exponent_min+(
                        critical_exponent_range/N)*j ) for j in range(N) ] )

# Calculating the RSS array values
for i in range(N):
    critical_exponent = critical_exponent_array[i]
    for j in range(N):
        W1_lower_bound = W1_range_array[j]
        W1_histogram_bins = histogram_bins(W1_lower_bound, W1max_safety_margin, 
                                                number_of_bins_distribution)
        W1_distribution_histogram = histogram_distribution_values(
            W1_histogram_bins, W1_distribution_from_tauC, W1maxTauC, 1,
                                                            critical_exponent)
        magnetization_curve_numerical = [ magnetization_integral(
                W1_histogram_bins, W1_distribution_histogram, recovery_time)
                                        for recovery_time in recovery_times]
        RSS_values_array[i][j] = RSS_calculation(magnetization_curve_numerical,
                                                        magnetization_curve)

pprint.pprint(RSS_values_array)

########################################################################
# PLOTTING THE RSS HEATMAP

b, a = np.meshgrid(W1_range_array, critical_exponent_array)

c = RSS_values_array
# c = c[:-1, :-1]
l_a = a.min()
r_a = a.max()
l_b = b.min()
r_b = b.max()
l_c, r_c  = np.amin(c), np.max(c)

figure, axes = plt.subplots()

c = axes.pcolormesh(a, b, RSS_values_array, cmap='coolwarm', vmin =
    np.amin(RSS_values_array), vmax = np.max(RSS_values_array) ) 
    #vmin=l_c, vmax=r_c)#, x_axis_label = 'Exponent', y_axis_label = 'Wmin')#,
    # shading='auto')
axes.set_title('RSS values heatmap')
axes.set_xlabel('Exponent')
axes.axis([l_a, r_a, l_b, r_b])
figure.colorbar(c)

plt.show()

########################################################################
# PLOTTING THE RSS HEATMAP

# Plot the numerical sets with the lowest RSS
RSS_minimum = np.array(np.where(RSS_values_array == np.amin(RSS_values_array)))
RSS = RSS_values_array[ RSS_minimum[0][0] ][ RSS_minimum[1][0] ]
W1_lower_bound = W1_range_array[ RSS_minimum[1][0] ]
critical_exponent = critical_exponent_array[ RSS_minimum[0][0] ]

print("Optimal lower bound W1 value: {:.1f}".format(W1_lower_bound)+
            " and critical exponent value: {:.2f}".format(critical_exponent))

W1_histogram_bins = histogram_bins(W1_lower_bound, W1max_safety_margin, 
                                                number_of_bins_distribution)
W1_distribution_histogram = histogram_distribution_values(
    W1_histogram_bins, W1_distribution_from_tauC, W1maxTauC, 1,
                                                critical_exponent)
magnetization_curve_numerical = [ magnetization_integral(W1_histogram_bins,
                        W1_distribution_histogram, recovery_time)
                                for recovery_time in recovery_times]

# W1_distribution_histogram = distribution_histogram(W1_distribution,
# number_of_bins_W1dist, critical_exponent, W1_lower_bound, W1max)
# Magnetization_curve_numerical = np.array( [ (
# magnetization_integral(W1_distribution_histogram[1], experimental_time,
# W1_lower_bound, W1max) ) for experimental_time in Recovery_times ] )

fig, ax = plt.subplots()

ax.scatter(recovery_times, magnetization_curve, s=20, color='black',
                                                                    marker="*")
ax.scatter(recovery_times, magnetization_curve_numerical, s=20, color='red',
                                                                    marker=".")

ax.grid()
ax.set_xscale('log')

ax.set_title('Magnetization Recovery Curve', pad=15)
ax.set(xlabel='Recovery time (s)', ylabel='Normalized recovery magnetization')

ax.legend(['NMR data', 'Numerical data (RSS={:.2f})'.format(RSS)],
                                                            loc='upper left')
plt.show()


plot_histogram(W1_histogram_bins, W1_distribution_histogram,
    histogram_title = 'Distribution of the relaxation rates',
        x_axis_label = '$W_1$ (Hz)', y_axis_label = 'D($W_1$) (s)')

########################################################################
# Set up the axes with gridspec
fig = plt.figure(figsize=(6, 6))
grid = plt.GridSpec(4, 4, hspace=0.2, wspace=0.2)
main_ax = fig.add_subplot(grid[:-1, 1:])
y_hist = fig.add_subplot(grid[:-1, 0], xticklabels=[], sharey=main_ax)
x_hist = fig.add_subplot(grid[-1, 1:], yticklabels=[], sharex=main_ax)

main_ax.set_title('Ela gie mou')

# scatter points on the main axes
main_ax.plot(W1_values, tauC_histogram_bins/time_scale, '*', markersize=3, alpha=0.2)

x_hist.plot(W1_histogram_bins, W1_distribution_histogram, color='blue')
# x_hist.invert_yaxis()

y_hist.plot(tauC_distribution_histogram, tauC_histogram_bins/time_scale, color='blue')
y_hist.invert_xaxis()

plt.show()
