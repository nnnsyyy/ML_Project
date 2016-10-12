# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis function."""
    # Creates the matrix with the degrees we want to apply to our data x
    # replecating the vector with [1 2 3 4 ... degree] to a matrix
    #degree_mat = np.tile(list(range(0,degree+1)), (len(x), 1))
    #return np.transpose(np.power(x,np.transpose(degree_mat)))
    X=np.zeros((x.shape[0],(degree+1)*x.shape[1]))
    for i in range(degree+1):
	for j in range(x.shape[1]):
            X[:,i+j]=x[j]**i
    return X
