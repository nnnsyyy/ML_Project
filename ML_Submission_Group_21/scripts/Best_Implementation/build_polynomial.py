# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """
	Polynomial basis function. Takes the basis x and maps it onto the polynomial basis
	of degree d in the form [1, x_1, ... x_n, x_1², ..., x_n²,...,x_1^d, ..., x_n^d]
	It means the basis goes from R^n to R^{n*d+1}
 
    @param x : the basis in the form [x_1 ... x_n], x_i in R^n
    @param d : the degree of the polynomial basis we want to obtain.
    @return X : the polynomial basis from x, in the form [1, x_1, ... x_n, x_1², ..., x_n²,...,x_1^d, ..., x_n^d]
    """
    X=np.zeros((x.shape[0], (degree)*x.shape[1]+1))
    for i in range(1, degree+1):
        for j in range(x.shape[1]):
            #print((i-1)*(x.shape[1])+j+1)
            X[:,(i-1)*x.shape[1]+j+1]=x[:,j]**i
    X[:,0]=1
    return X
