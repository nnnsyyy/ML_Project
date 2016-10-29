# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np

# =============================================================================
### FIRST FUNCTIONS THAT ACT ON OUR RAW DATA ##################################
# =============================================================================
def count_NaN(tX):
    """
        Counts the number of NaNs in a column and adds it in a column at the end in order for the information to be conserved.
        @param tX : the input of our data.
        
    """
    negative_NaN_table = np.array([0,4,5,6,12,23,24,25,26,27,28])
    NEGATIVE_NAN = -999.0
    zero_NaN_table = [29]
    ZERO_NAN = 0
    nan_count = np.zeros((tX.shape[0]))
    for i in range (tX.shape[0]):
            nan_count[i]=nan_count[i]+(tX[i,negative_NaN_table]== NEGATIVE_NAN).sum()+(tX[i,zero_NaN_table] == ZERO_NAN)
    return np.c_[tX, nan_count]

def standardize(x, mean_x=None, std_x=None):
    """
        Standardize the original data set. 
        If the standard deviation of a column is 0, we remove it.
        @param x : the input 2d array that we want to standardize
        @param mean_x : the mean we want to apply to the columns
        @param std_x : the standard deviation we want to apply to the columns
        @return tx : the standardized input x with the columns with std == 0 removed
        @return mean_x : the means of the columns (including those which are removed later on)
        @return std_x : the standard deviations of the columns (including those which are removed later on)
    """
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
        
    x = x - mean_x
    
    if std_x is None:
        std_x = np.std(x, axis=0)
    #Iterate over all columns of the table. If its std is 0, we remove it from the dataset, it means that
    #all its values are the same
    excluded_col = np.empty((1,0))    
    for i in range(x.shape[1]):
        if std_x[i] == 0:
            excluded_col = np.append(excluded_col, i)
        else:
            x[:, i] = x[:, i] / std_x[i]
    tx = np.array(x)
    tx = np.delete(tx, excluded_col, axis=1)
    return tx, mean_x, std_x

def sanitize_NaN(tX,median_vec=None):
    """
    Removes the NaNs from the data and replace it with the median of the valid data. 
    The columns are hard coded, represent the columns from the dataset for the project 1
    Returns the computed median, and can apply a median taken as input. 
    The input median has to be the median of the NaN columns below.
    @param tX : the raw data from train.csv
    @param median_vec : the vector of the medians that were previously returned from this function 
                        (contains the median of the valid input of the columns of the array below.)
    """
    
    x = tX.copy()
    #Hard coding of the columns of the data from train.csv that contains some NaNs in their columns.
    #There are two types of NaNs, either -999 or 0, and we distinguish both cases 
    #(our vector median_vec does not, it simply contains all the medians of the valid data)
    negative_NaN_table = np.array([0,4,5,6,12,23,24,25,26,27,28])
    NEGATIVE_NAN = -999.0
    zero_NaN_table = [29]
    ZERO_NAN = 0
    # Compute the median of the valid data is no median is provided
    if median_vec is None:
        n_iter=0
        median_vec = np.zeros(len(negative_NaN_table) + len(zero_NaN_table))
        for row in negative_NaN_table:
            x_without_nan = x[:,row][np.where(x[:,row] != NEGATIVE_NAN)]
            #We need to distinguish the case where we have only NaNs in the column, which happens when we
            #split the data with our split_dataset method.
            if len(x_without_nan > 0):
                median_vec[n_iter] = np.median(x_without_nan)
            else:
                median_vec[n_iter] = 0
            n_iter=n_iter+1
        for row in zero_NaN_table:
            x_without_nan = x[:,row][np.where(x[:,row] != ZERO_NAN)]
            #We also distinguish the columns here.
            if len(x_without_nan > 0):
                median_vec[n_iter] = np.median(x_without_nan)
            else: 
                median_vec[n_iter] = 0
            n_iter=n_iter+1
    else:
        assert len(median_vec) == len(negative_NaN_table) + len(zero_NaN_table)
        
    #Replace the NaN values with the median of the table        
    for i,row in enumerate(negative_NaN_table):
        x[:,row][np.where(x[:,row] == NEGATIVE_NAN)] = median_vec[i]
    for j,row in enumerate(zero_NaN_table):
        x[:,row][np.where(x[:,row] == ZERO_NAN)] = median_vec[i+j+1]
    return x, median_vec
        
def split_dataset(y,tX,ids):
    """
        Splits the initial dataset into four smaller datasets, according to the the PRI_jet_num
        We do the splits manually because we ended having problems with the datatypes when we tried to do
        something more automatised.
        @param y : the raw y of our data
        @param tX : the raw features matrix from our data
        @param ids : the ids for the features of our data (splitting them will allow us merge them correctly later on)
        @return : 3 tuples with 4 arrays each containing the splitted version of our data
    """
    split_0 = np.where(tX[:,22]==0)
    split_1 = np.where(tX[:,22]==1)
    split_2 = np.where(tX[:,22]==2)
    split_3 = np.where(tX[:,22]==3)
    
    y_0 = [y[i] for i in split_0]
    y_1 = [y[i] for i in split_1]
    y_2 = [y[i] for i in split_2]
    y_3 = [y[i] for i in split_3]
    
    ids0 =  [ids[i] for i in split_0]
    ids1 =  [ids[i] for i in split_1]
    ids2 =  [ids[i] for i in split_2]
    ids3 =  [ids[i] for i in split_3]
    
    return [y_0,y_1,y_2,y_3],[tX[split_0,:],tX[split_1,:],tX[split_2,:],tX[split_3,:]],[ids0,ids1,ids2,ids3]

def merge_dataset(y_split,id_split):
    """
        Given the y_split tuple and id_split tuple, merges them and sorts them according to the id
        @param y_split : tuple with 4 entries, containing our predictions from the model.
        @param id_split : the id of each of our predictions
        @return : the merged list of predictions y and the merged ids (sorted w.r.t to ids)
    """
    y_tot = np.squeeze(np.hstack(y_split))
    ids_tot = np.squeeze(np.hstack(id_split))
    
    y_id_merged = np.array([(y_out,ids_out) for (ids_out,y_out) in sorted(zip(ids_tot,y_tot))])
    return y_id_merged[:,0],y_id_merged[:,1]


# =============================================================================
### FUNCTION FOR GRADIENT DESCENT #############################################
# =============================================================================

def generate_initial_w(tX):#, sanatization_vec=None):
    """
    Generates an initial guess, which is a median of each column in the data. 
    @param tX : the sanatized raw data
    @return: a vector wich contains a median of each row
    """
    #san_idx = np.array([0,4,5,6,12,23,24,25,26,27,28,29])
    initial_w = np.zeros(tX.shape[1])
    # generate a median for each row
    #if sanatization_vec is None:
    for col in range(tX.shape[1]):
        initial_w[col] = np.median(tX[:,col])
    #else:
    #    for col in range(tX.shape[1]):
    #        if col in san_idx:
    #            initial_w[col] = 
    #        else:
    #            initial_w[col] = np.median(tX[:,col])
    return initial_w

    
# =============================================================================
### FUNCTION FOR CROSS VALIDATION #############################################
# =============================================================================

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold cross-validation."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    
    
# =============================================================================
### FUNCTION FOR SGD ##########################################################
# =============================================================================
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
