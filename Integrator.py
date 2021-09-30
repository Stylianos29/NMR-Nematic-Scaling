import pandas as pd

import ast
import pprint

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma

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
#pprint.pprint(Magnetization_recovery_dict)

###################################################################################

#Data for testing

#for temperature_value in Magnetization_recovery_dict.keys():
temperature_value=50.0

x = np.array(df[Magnetization_recovery_dict[temperature_value][0]])
x = x[np.logical_not(np.isnan(x))]
y = np.array(df[Magnetization_recovery_dict[temperature_value][1]])
y = y[np.logical_not(np.isnan(y))]

#Making sure that all time recovery values are in ascenting order
arr1inds = x.argsort()
x = x[arr1inds[::1]]
y = y[arr1inds[::1]]

y_norm = ( y - min(y) ) / ( max(y) - min(y) )

###################################################################################

#preparing the grid
number_of_rows = 10000
DistMax = 0.5
deltaDist = DistMax / number_of_rows
number_of_columns = 100000
W1max=200
deltaW1 = W1max / number_of_columns

t_inflection = 4*10E-4
xpeak = 0.172722420529815795438
W1peak = xpeak / 10E-4

# grid initialization: rows and columns
grid_distribution = np.zeros(shape=(number_of_rows, number_of_columns))

#Calculation of the squared residue between y data values and prediction
def Squared_Residue(z, w):
    return np.linalg.norm(z-w)**2

#the center of each column is equidistant from its next
W1_fit_points = np.array([ deltaW1/2 + deltaW1 * n for n in range(0,number_of_columns)])

#find the column that the W1peak point belongs to
peak_column_number = int(np.floor_divide(W1peak, deltaW1)+1)

def recovery_function(x):
    return (1 - (9/10)*np.exp(-6*x) - (1/10)*np.exp(-x) )

#Squared_Residue(y_norm, recovery_function(x))

###################################################################################
#distribution
def normalization_check(z,dz):
    return np.sum(z)*dz

#triangular distribution
# initial_dist_shape_dict={"1":[2/(W1max*W1peak), 0],"-1":[-2/((W1max-W1peak)*W1max), 2/(W1max-W1peak)]}
# pprint.pprint(initial_dist_shape_dict)

# W1_distribution = np.array( [ initial_dist_shape_dict["1"][0]*W1_fit_points[k]
#                                 +initial_dist_shape_dict["1"][1]
#                                     for k in range(0, peak_column_number)]
#                     + [ initial_dist_shape_dict["-1"][0]*W1_fit_points[k]
#                                 +initial_dist_shape_dict["-1"][1]
#                                 for k in range(peak_column_number, number_of_columns)] )

beta = 5
alpha = (W1peak + beta)/beta - 30
W1_distribution = gamma.pdf(W1_fit_points, a = alpha, scale = beta)

deltaDist = max(W1_distribution)*1.05/number_of_rows

# W1_distribution[2000] = W1_distribution[2000]*1.00001

W1_distribution_discrete = []
for i in range(0,len(W1_distribution)):
    W1_distribution_discrete.append(int(np.rint(W1_distribution[i]/deltaDist)))

W1_distribution_discrete = np.array(W1_distribution_discrete)
np.sum(W1_distribution_discrete)

1/(deltaW1*deltaDist)

normalization_check(W1_distribution, deltaW1)

# W1_distribution_discrete[10000:11000]

###################################################################################

#W1peak
# def perturbation(distribution, peak_value):
#     for j in range(0,len(distribution)):
#         if j < peak_value:

#         else:


#perturb slightly the distribution somewhere
max_index = np.argmax(W1_distribution_discrete)
# W1_distribution_discrete[np.floor_divide(max_index*9,10)] += 4

###################################################################################

Difference_first = [ (W1_distribution_discrete[j+1] - W1_distribution_discrete[j]) for j in range(0,number_of_columns -1) ]
Difference_first.insert(0,0)

Difference_second = [ (Difference_first[j+1] - Difference_first[j-1]) for j in range(1,number_of_columns -1) ] #2*W1_distribution_discrete[j] + 
Difference_second.insert(0,0)
Difference_second.insert(number_of_columns,0)

###################################################################################

# #Quick plot test 
# fig, ax = plt.subplots()
# # ax.scatter(W1_fit_points, W1_distribution_discrete, color='red') #s=20,
# # ax.scatter(W1_fit_points, Difference_first, s=20, color='green')
# ax.scatter(W1_fit_points, Difference_second, s=20, color='red')
# ax.grid()

# ax.set(xlabel='Relaxation rates (1/s)', ylabel='', title='Probability distribution of the relaxation rates')

# #fig.savefig('NMR-Nematic-Scaling\Output_Files\Test_of_construction_triangular_distribution_T=30K.png')
# # plt.xlim(73, 75)
# plt.show()

# ###################################################################################

# import collections

# counter=collections.Counter(W1_distribution_discrete)
# del(counter[0])

# plt.bar(range(len(counter)), list(counter.values()), align='center')
# plt.xticks(range(len(counter)), list(counter.keys()))

# plt.show()

###################################################################################

def integrator(recovery_time, W1_values, W1_distribution_values, delta_W1, hist_columns_numbers):
    return np.sum(np.array( [ ( recovery_function( recovery_time * W1_values[j] ) * W1_distribution_values[j] *delta_W1  ) for j in range(0, hist_columns_numbers)] ))

# for temperature_value in Magnetization_recovery_dict.keys():
#     x = np.array(df[Magnetization_recovery_dict[temperature_value][0]])
#     x = x[np.logical_not(np.isnan(x))]
#     y = np.array(df[Magnetization_recovery_dict[temperature_value][1]])
#     y = y[np.logical_not(np.isnan(y))]

#     #Making sure that all time recovery values are in ascenting order
#     arr1inds = x.argsort()
#     x = x[arr1inds[::1]]
#     y = y[arr1inds[::1]]

#     y_norm = ( y - min(y) ) / ( max(y) - min(y) )

y_fit = np.array( [ integrator(x[k], W1_fit_points, W1_distribution, deltaW1, number_of_columns) for k in range(0,len(x)) ] )
y_fit
#y_fit = recovery_function(x*(W1peak-130))

Squared_Residue(y_norm, y_fit)

###################################################################################

#Quick plot test 
y_actual = y_norm
y_integration = y_fit

fig, ax = plt.subplots()
ax.scatter(x, y_actual, s=20, color='black')
ax.scatter(x, y_integration, s=10, color='red')
ax.set_xscale('log')
ax.grid()
ax.set(xlabel='Recovery time (s)', ylabel='Magnetization', title='Magnetization Curves reconstruction')

fig.savefig('NMR-Nematic-Scaling\Output_Files\Test_of_construction_T=30K_alpha='+"{:.2f}".format(alpha)+'_beta='+str(beta)+'.png')
# fig.savefig('NMR-Nematic-Scaling\Output_Files\test2.png')
plt.show()

###################################################################################

def log_dev_recovery_function(x):
    return ( (27/5)*x*np.exp(-6*x) + (1/10)*x*np.exp(-x) )

def second_log_dev_recovery_function(x):
    return ( (27/5)*x*np.exp(-6*x) + (1/10)*x*np.exp(-x) - (162/5)*(x**2)*np.exp(-6*x) - (1/10)*(x**2)*np.exp(-x))


np.dot(log_dev_recovery_function(t_inflection*W1_fit_points),W1_distribution)*deltaW1
