# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx, mean_x, std_x


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold cross-validation."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def normalize(tX):
    """
	Custom function that computes the standardization of the data.
	@param : all the parameters of the model.
    """
    return (tX-np.mean(tX,axis=0))/np.std(tX,axis=0)


def sanitize_NaN(tX):
    """
	Removes the NaNs from the data and replace it with the median of the valid data.
	The columns are hard coded, represent the columns from the dataset for the project 1
    """
    x = tX.copy()
    negative_NaN_table = np.array([0,4,5,6,12,23,24,25,26,27,28])
    NEGATIVE_NAN = -999.0
    zero_NaN_table = [29]
    ZERO_NAN = 0
    for row in negative_NaN_table:
        x_without_nan = x[:,row][np.where(x[:,row] != NEGATIVE_NAN)]
        x[:,row][np.where(x[:,row] == NEGATIVE_NAN)] = np.median(x_without_nan)
    for row in zero_NaN_table:
        x_without_nan = x[:,row][np.where(x[:,row] != ZERO_NAN)]
        x[:,row][np.where(x[:,row] == NEGATIVE_NAN)] = np.median(x_without_nan)
    return x


def exclude_NaN(tX):
    """
	Removes the columns containing NaNs from the data set.
	(i.e. the ones having -999 or zeros that should not be there)
	The columns are hard coded, represent the columns from the dataset for the project 1 
    """
    sort_no_NaN= [1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,28,29]
    tX_reduced = tX[:,sort_no_NaN]
    return tX_reduced


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
