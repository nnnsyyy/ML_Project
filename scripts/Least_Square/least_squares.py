# -*- coding: utf-8 -*-
"""a function used to compute the solution of the normal equation."""

import numpy as np
import costs as co

def least_squares(y, tx):
    """calculate the least squares solution."""
    # Least squares, returns mse, and optimal weights
    # Computes (tx^{T}*tx)^{-1}*tx^{T}*y
    
    x_inv=np.linalg.inv(np.dot(tx.T,tx))
    #return np.linalg.solve(np.dot(tx.T,tx),np.dot(tx.T,y))
    w = np.linalg.solve(x_inv, np.dot(tx.T,y))
    loss = co.compute_loss(y, tx, w)
    return loss, w
