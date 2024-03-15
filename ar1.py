"""
Author: Grayson Olin
File: ar1.py
Language: Python 3
Date Created: July 26, 2021
Last Modified: July 31, 2021

Implements a correlation matrix for an AR(1) model
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

def cov(N, rho = 0.5, sigma = 13.5):
    mat = cov_mat(N, rho)
    cov = (sigma**2/(1 - rho**2))*mat
    return cov

def cov_mat(N, rho):
    mat = np.zeros(shape=(N,N))
    for n in range(N):
        lst = [rho**n]*(N-n)
        mat += np.diag(lst, n)
    mat = mat + mat.T
    mat[range(N), range(N)] = 1
    return mat

def noise(rho, sigma, years):
    Rt = [np.random.normal(0, sigma)]
    for t in years[1:]:
        epsilon = np.random.normal(0, sigma)
        R = rho*Rt[-1] + epsilon
        Rt.append(R)
    return Rt
