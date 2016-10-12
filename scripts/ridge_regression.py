# -*- coding: utf-8 -*-
"""a function used to compute the solution of the normal equation for the ridge regression."""

import numpy as np

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    return np.linalg.solve(np.dot(tx.T,tx)+lamb*np.identity(tx.shape[1]),np.dot(tx.T,y))#/(2*len(tx))

