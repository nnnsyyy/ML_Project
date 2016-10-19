# -*- coding: utf-8 -*-
"""a function used to compute the solution of the normal equation."""

import numpy as np

def least_squares(y, tx):
    """calculate the least squares solution."""
    # Least squares, returns mse, and optimal weights
    # Computes (tx^{T}*tx)^{-1}*tx^{T}*y
    return np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
