import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gamma

number_of_rows = 10000
number_of_columns = 100000
W1max=200
deltaW1 = W1max / number_of_columns

t_inflection = 4*10E-4
xpeak = 0.172722420529815795438
W1peak = xpeak / 10E-4

#the center of each column is equidistant from its next
W1_fit_points = np.array([ deltaW1/2 + deltaW1 * n for n in range(0,number_of_columns)])

beta = 5
alpha = (W1peak + beta)/beta - 25
W1_distribution = gamma.pdf(W1_fit_points, a = alpha, scale = beta)

def W1_tauC(W1max, omegaL, tauC):
    return ( (2 * W1max * omegaL * tauC) / ( 1 + (tauC * omegaL)**2 ) )

def derivative_W1_tauC(W1max, omegaL, tauC):
    return ( (2 * W1max * omegaL * (1 - (tauC * omegaL)**2) ) / ( (1 + (tauC * omegaL)**2)**2 ) )

#TauC distribution
tauCmin=1e-5
tauCmax=10
DeltaTauC=(tauCmax-tauCmin)/number_of_columns

TauC_fit_points=[]
for j in range(0,number_of_columns):
    TauC_fit_points.append( DeltaTauC*(j+0.5) )
TauC_fit_points = np.array(TauC_fit_points)

omegaL = 1
TauC_distribution = derivative_W1_tauC(W1max, omegaL, TauC_fit_points)
#W1_tauC(W1max, omegaL, TauC_fit_points) *

###################################################################################

#Quick plot test 
fig, ax = plt.subplots()
ax.scatter(TauC_fit_points, TauC_distribution, s=20, color='red')
# ax.set_xscale('log')
ax.grid()
ax.set(xlabel='Oscillation times (s)', ylabel='Probability distribution', title='OScillation times distribution')

plt.show()

###################################################################################

#DeltaEC distribution
DeltaECmin=1e-5
DeltaECmax=1000
DeltaDeltaEC=(DeltaECmax-DeltaECmin)/number_of_columns

DeltaEc_fit_points=[]
for j in range(0,number_of_columns):
    DeltaEc_fit_points.append( DeltaDeltaEC*(j+0.5) )
DeltaEc_fit_points = np.array(DeltaEc_fit_points)

def derivative_W1_DeltaE(W1max, omegaL, tauC0, DeltaE, Temperature):
    kB = 8.617333262145e-5 #eV/K
    return 2 * np.exp(DeltaE/(kB * Temperature)) * (tauC0 * omegaL) * (1 - np.exp( (2 * DeltaE) / (kB * Temperature) ) * (tauC0 * omegaL)**2 ) * W1max / ( kB * Temperature * (1 + np.exp((2 * DeltaE)/(kB * Temperature)) * (tauC0 * omegaL)**2 )**2 ) 

omegaL = 1
T=1000
DeltaEc_distribution = derivative_W1_DeltaE(W1max, omegaL, tauCmin, DeltaEc_fit_points, T)

W1_DeltaEc(W1max, omegaL, DeltaEc_fit_points) *

nan_array = np.isnan(DeltaEc_distribution)
not_nan_array = ~ nan_array
temporary = DeltaEc_distribution[not_nan_array]

len(temporary)
len(DeltaEc_fit_points)

###################################################################################

#Quick plot test
ax.clear()
fig, ax = plt.subplots()
ax.scatter(DeltaEc_fit_points, temporary, s=20, color='red')
# ax.set_xscale('log')
ax.grid()
ax.set(xlabel='Energy Barrier (eV)', ylabel='Probability distribution', title='Energy Barrier Distribution')
# plt.xlim(40, DeltaECmax)
plt.show()

###################################################################################
