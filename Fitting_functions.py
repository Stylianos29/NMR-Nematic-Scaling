import numpy as np

def testmod():
       print("test module success")

#Simple exponential form
def exponential(x, M0, phi, alpha):
    result = M0*(1-phi*np.exp(-alpha*x))
    return result

#Stretched exponential form
def stretched_exponential(x, M0, phi, alpha, beta):
    return M0*(1-phi*np.exp(-(alpha*x))**beta)
