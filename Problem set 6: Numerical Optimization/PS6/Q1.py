import numpy as np

def Q(theta):
    Q = 1/theta + np.exp(theta)
    return Q

def Q2(theta):
    Q = np.cos(theta)
    return Q