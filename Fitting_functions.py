import numpy as np

def testmod():
       print("test module success")

#Simple exponential form
def exponential(x, M0, phi, alpha):
    return M0*(1-phi*np.exp(-alpha*x))

#Simplified Stretched exponential form
def reduced_stretched_exponential(x, alpha, beta):
    return 1-np.exp(-alpha*x)**beta

#Stretched exponential form
def stretched_exponential(x, M0, phi, alpha, beta):
    return M0*(1-phi*np.exp(-(alpha*x))**beta)
