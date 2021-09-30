import numpy as np

def testmod():
       print("test module success")

#Simple exponential form
def exponential(x, M0, phi, alpha):
    return M0*(1-phi*np.exp(-alpha*x))

def recovery_function(x):
    return (1 - (9/10)*np.exp(-6*x) - (1/10)*np.exp(-x) )

#Simplified Stretched exponential form
def reduced_stretched_exponential(x, alpha, beta):
    return 1-np.exp(-alpha*x)**beta

#Stretched exponential form
def stretched_exponential(x, M0, phi, alpha, beta):
    return M0*(1-phi*np.exp(-(alpha*x)**beta) )

#Stretched exponential recovery function for I=1/2
def stretched_recovery_function(x, M0, phi, alpha, beta):
    return ( M0*(1 - 2*phi*( (9/10)*np.exp(-1*pow((6*alpha*x), beta) ) - (1/10)*np.exp(-1*pow((alpha*x), beta) ) ) ) )
