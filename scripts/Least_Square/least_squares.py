# -*- coding: utf-8 -*-
"""a function used to compute the solution of the normal equation."""

import numpy as np
import costs as co
from proj1_helpers import *

def least_squares(y, tx):
    """calculate the least squares solution."""
    # Least squares, returns mse, and optimal weights
    # Computes (tx^{T}*tx)^{-1}*tx^{T}*y
    
    #x_inv=np.linalg.inv(np.dot(tx.T,tx))
    xtx=np.dot(tx.T,tx)
    w = np.linalg.solve(xtx, np.dot(tx.T,y))
    return w
