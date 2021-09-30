from numpy.lib.function_base import delete, diff
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
temperature_value=25.0

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

t_inflection = 1E-3
Inflection_magnetization_exp = 0.55
xpeak = 0.172722420529815795438
W1peak = xpeak / t_inflection

slope_at_inflection = .22

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

###################################################################################

def log_dev_recovery_function(x):
    return ( (27/5)*x*np.exp(-6*x) + (1/10)*x*np.exp(-x) )

def second_log_dev_recovery_function(x):
    return ( (27/5)*x*np.exp(-6*x) + (1/10)*x*np.exp(-x) - (162/5)*(x**2)*np.exp(-6*x) - (1/10)*(x**2)*np.exp(-x))

#Inflection Magnetization (0.20736543348999795)
# np.dot(log_dev_recovery_function(t_inflection*W1_fit_points),W1_distribution)*deltaW1

def inflection_slope(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1):
    return np.dot(log_dev_recovery_function(t_inflection*W1_fit_points),W1_distribution_discrete*deltaDist)*deltaW1
    
print(inflection_slope(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1)/0.20736543348999795)

#Inflection Magnetization (0.13902848935038245)
# np.dot(second_log_dev_recovery_function(t_inflection*W1_fit_points),W1_distribution)*deltaW1
def inflection_slope_difference(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1):
    return np.dot(second_log_dev_recovery_function(t_inflection*W1_fit_points),W1_distribution_discrete*deltaDist)*deltaW1

print(inflection_slope_difference(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1)/0.13902848935038245)

###################################################################################

#185996278

# np.rint(1/(deltaW1*deltaDist)) - 185996278

# normalization_check(W1_distribution_discrete, deltaW1)


# np.sum(W1_distribution_discrete) - 185996278

def renormalization(discrete_distribution):
    discrete_distribution = discrete_distribution / (np.sum(discrete_distribution) * (deltaW1*deltaDist))
    for i in range(0,len(discrete_distribution)):
        discrete_distribution[i] = int(np.rint(discrete_distribution[i]))
    return discrete_distribution

###################################################################################

#W1peak
# def perturbation(distribution, peak_value):
#     for j in range(0,len(distribution)):
#         if j < peak_value:

#         else:

#perturb slightly the distribution somewhere

###################################################################################
# counter=0
# deviations_second=[0]
# while( (counter < 25) and (len(deviations_second)!=0) ):

def numerical_difference(discrete_distribution):
    difference = [ (discrete_distribution[j+1] - discrete_distribution[j]) for j in range(0,len(discrete_distribution) -1) ]
    difference.insert(0,0)
    while (len(difference) < len(discrete_distribution)):
        difference.append(0)
    difference = np.array(difference)
    return difference

Difference_first = numerical_difference(W1_distribution_discrete)

# Difference_second = [ (Difference_first[j+1] - Difference_first[j-1]) for j in range(1,number_of_columns -1) ] #2*W1_distribution_discrete[j] + 
# Difference_second.insert(0,0)
# Difference_second.insert(number_of_columns,0)
# Difference_second = np.array(Difference_second)

Difference_second = numerical_difference(Difference_first)

###################################################################################

def monotonicity_check(discrete_distribution, peak_point):
    monotonicity_list = []
    for j in range(0, peak_point):
        if (discrete_distribution[j] <= -1):
            monotonicity_list.append(j)
    for j in range(peak_point+1, len(discrete_distribution)):
        if (discrete_distribution[j] >= 1):
            monotonicity_list.append(j)
    return monotonicity_list

###################################################################################
###################################################################################

x_transition1 = 0.065213865307513286845

x_transition2 = 0.43783397651520976759

if (inflection_slope_difference(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1) < 0):
    if (inflection_slope(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1) > slope_at_inflection):
        print('Area 2')
    else:
        print('Area 1')
else:
    if (inflection_slope(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1) > slope_at_inflection):
        print('Area 3')
    else:
        print('Area 4')

max_index = np.argmax(W1_distribution_discrete)
W1_distribution_discrete[np.floor_divide(max_index*9,10)] += 2

###################################################################################

def difference_first_check(discrete_distribution, peak_point):
    result=[]
    for j in range(0, peak_point):
        if ( Difference_first[j] > 1 ):
            result.append(j)
    return result

counter = 1
while True:
    Difference_first = numerical_difference(W1_distribution_discrete)
    monotonicity = monotonicity_check(Difference_first, max_index)
    deviations_first_indices = difference_first_check(Difference_first, max_index)

    if ( (len(monotonicity) == 0) and (len(deviations_first_indices) == 0) ):
        break
    else:
        if ((len(monotonicity) > 0)):
            for k in monotonicity:
                if k < max_index:
                    W1_distribution_discrete[k] = W1_distribution_discrete[k-1]
                # else:
                #     W1_distribution_discrete[k] = W1_distribution_discrete[k-1]
        if ((len(deviations_first_indices) > 0)):
            for k in deviations_first_indices:
                if k < max_index:
                    W1_distribution_discrete[k-1] += 1
                # else:
                #     W1_distribution_discrete[k] = W1_distribution_discrete[k-1]

    print(counter)
    counter+=1

print(monotonicity)

W1_distribution_discrete = renormalization(W1_distribution_discrete)

inflection_slope(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1)
inflection_slope_difference(t_inflection, W1_fit_points, W1_distribution_discrete, deltaDist, deltaW1)

#185996278
# np.sum(W1_distribution_discrete) - 185996278

# Difference_first[monotonicity[0]]
# W1_distribution_discrete[monotonicity[0]]

###################################################################################

deviations_first, deviations_first_count = np.unique(Difference_first, return_counts=True)
deviations_second, deviations_second_count = np.unique(Difference_second, return_counts=True)

for k in [-1, 0, 1]:
    deviations_first = np.delete(deviations_first, np.where(deviations_first == k))
    deviations_second = np.delete(deviations_second, np.where(deviations_second == k))

if len(deviations_second)>0:
    deviations_first_indices={}
    deviations_second_indices={}
    for k in deviations_first:
        result = np.nonzero(Difference_first == k)
        deviations_first_indices[k] = result
    for j in deviations_second:
        result = np.nonzero(Difference_second == j)
        deviations_second_indices[j] = result

list(deviations_first_indices[2])[0]

###################################################################################

# # # #so we need to focus on 21419 because of the way Difference_first has been set up

# for k in deviations_first:
#     print(list(deviations_first_indices[k][0])[0])

# W1_distribution_discrete[2380]

# if index_first < max_index:
#     if W1_distribution_discrete[index_first-1] < W1_distribution_discrete[index_first]-1:
#         W1_distribution_discrete[index_first-1] = W1_distribution_discrete[index_first]-1
#     if W1_distribution_discrete[index_first+1] < W1_distribution_discrete[index_first]:
#         W1_distribution_discrete[index_first+1] = W1_distribution_discrete[index_first]

###################################################################################

#Quick plot test
fig, ax = plt.subplots()

# ax.scatter(W1_fit_points, Difference_second, s=20, color='green')
# ax.scatter(W1_fit_points, Difference_second, s=20, color='red')

ax.scatter(W1_fit_points, W1_distribution, s=20, color='red')
ax.scatter(W1_fit_points, W1_distribution_discrete*deltaDist, s=20, color='green')

ax.grid()

ax.set(xlabel='Relaxation rates (1/s)', ylabel='', title='Probability distribution of the relaxation rates')

#fig.savefig('NMR-Nematic-Scaling\Output_Files\Test_of_construction_triangular_distribution_T=30K.png')
# plt.xlim(73, 75)
plt.show()

###################################################################################