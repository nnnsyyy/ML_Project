# Useful starting lines

# gradient descent: gd_run(intial_w, max_iters, gamma)
# stochastic gradient descent: sgd_run(intial_w, max_iters, gamma)
# least squares: ls_run()
# ridge regression: rr_run(lambda_)
# logistic regression: lr_run(initial_w, max_iters, gamma)
# regularized logistic regression: rlr_run(lambda_, initial_w, max_iters, gamma)

# all input files are stored in filefold '../data'
# all output should be saved in filefold 'results', with the name 'output_(method_name).csv'

import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from implementations import *
from helpers import *

#helper methods
def load_data():
    """
    Loads the training data.
    """
    print('LOADING THE DATA: ',end=" ")
    DATA_TRAIN_PATH = '../data/train.csv'
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print('DONE')
    return y, tX, ids

def clean_input(tX):
    """
    Cleans the raw input via various methods.
    """
    tX = count_NaN(tX)
    tX, median_tr = sanitize_NaN(tX)
    tX, mean_tr, std_tr = standardize(tX)
    return tX, median_tr, mean_tr, std_tr

def test_model(weights, median_tr, mean_tr, std_tr, method_name):
    """
    Tests the supplied weights by making a prediction on the testing data. Places the results 
    at results/output_<method_name>.csv.
    """
    DATA_TEST_PATH = '../data/test.csv'
    print('IMPORTING TESTING DATA :',end=" ")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print('DONE')
    
    tX_test_sorted = count_NaN(tX_test)
    tX_test_sorted,median_vec = sanitize_NaN(tX_test_sorted,median_tr)
    tX_test_sorted,mean_tr,std_tr = standardize(tX_test_sorted,mean_tr,std_tr)
    OUTPUT_PATH = 'results/output_' + method_name + '.csv' 
    # Fill in desired name of output file for submission
    print('EXPORTING TESTING DATA WITH PREDICTIONS :',end=" ")
    y_pred = predict_labels(np.array(weights), np.array(tX_test_sorted))
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('DONE')
   
#gradient descent
def gd_run(initial_w, max_iters, gamma):
    """ Gradient descent running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first 
        part, then the whole simulation takes part in 3 steps :
            1. Load the training data.
            2. Train the model.
            3. Make prediction on the testing data.
    """
    
    #1. LOAD THE DATA
    y, tX, ids = load_data()


    #2. TRAIN THE MODEL
    #Let us now clean the input
    tX,median_tr = sanitize_NaN(tX)
    tX,mean_tr,std_tr = standardize(tX)

    weights, loss = least_squares_GD(y, tX, initial_w, max_iters, gamma)
    print('Weights on whole set\n',weights,'\nLoss', loss)
    
    #3. TEST THE MODEL AND EXPORT THE RESULTS
    #test_model(weights, median_tr, mean_tr, std_tr, 'least_squares_GD')
    DATA_TEST_PATH = '../data/test.csv'
    print('IMPORTING TESTING DATA :',end=" ")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print('DONE')
    
    tX_test_sorted,median_vec = sanitize_NaN(tX_test,median_tr)
    tX_test_sorted,mean_tr,std_tr = standardize(tX_test_sorted,mean_tr,std_tr)
    OUTPUT_PATH = 'results/output_least_squares_GD.csv' 
    # Fill in desired name of output file for submission
    print('EXPORTING TESTING DATA WITH PREDICTIONS :',end=" ")
    y_pred = predict_labels(np.array(weights), np.array(tX_test_sorted))
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('DONE')
 
#gd_run(np.zeros(30), 2000, 2**-6)
    
#stochastic gradient descent
def sgd_run(initial_w, max_iters, gamma):
    """ Stochastic gradient descent running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first
        part, then the whole simulation takes part in 3 steps :
            1. Load the training data.
            2. Train the model.
            3. Make prediction on the testing data.
    """
    
    #1. LOAD THE DATA
    y, tX, ids = load_data()


    #2. TRAIN THE MODEL
    #Let us now clean the input
    tX,median_tr = sanitize_NaN(tX)
    tX,mean_tr,std_tr = standardize(tX)
    
    weights, loss = least_squares_SGD(y, tX, initial_w, max_iters, gamma)
    print('Weights on whole set\n',weights,'\nLoss', loss)
    
    #3. TEST THE MODEL AND EXPORT THE RESULTS
    #test_model(weights, median_tr, mean_tr, std_tr, 'least_squares_SGD')
    DATA_TEST_PATH = '../data/test.csv'
    print('IMPORTING TESTING DATA :',end=" ")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print('DONE')
    
    tX_test_sorted,median_vec = sanitize_NaN(tX_test,median_tr)
    tX_test_sorted,mean_tr,std_tr = standardize(tX_test_sorted,mean_tr,std_tr)
    OUTPUT_PATH = 'results/output_least_squares_SGD.csv' 
    # Fill in desired name of output file for submission
    print('EXPORTING TESTING DATA WITH PREDICTIONS :',end=" ")
    y_pred = predict_labels(np.array(weights), np.array(tX_test_sorted))
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('DONE')

#sgd_run(np.zeros(30), 605, 2**-13)

#least_square
def ls_run():
    """ Least square running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first 
        part, then the whole simulation takes part in 3 steps :
            1. Load the training data.
            2. Train the model. 
            3. Make prediction on the testing data.
    """
    
    #1. LOAD THE DATA
    y, tX, ids = load_data()

    
    #2. TRAIN THE MODEL
    #Let us now clean the input
    tX, median_tr, mean_tr, std_tr = clean_input(tX)

    weights, loss = least_squares(y, tX)
    print('Weights on whole set\n',weights,'\nLoss',loss)
    
    #3. TEST THE MODEL AND EXPORT THE RESULTS
    test_model(weights, median_tr, mean_tr, std_tr, 'least_squares')

# ridge regression
def rr_run(lambda_):
    """ Ridge regression running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first 
        part, then the whole simulation takes part in 3 steps :
            1. Load the training data.
            2. Train the model.
            3. Make prediction on the testing data.
    """
    
    #1. LOAD THE DATA
    y, tX, ids = load_data()

    #2. TRAIN THE MODEL
    #Let us now clean the input
    tX, median_tr, mean_tr, std_tr = clean_input(tX)

    weights, loss = ridge_regression(y, tX, lambda_)
    print('Weights on whole set\n',weights,'\nLoss',loss)

    #3. TEST THE MODEL AND EXPORT THE RESULTS
    test_model(weights, median_tr, mean_tr, std_tr, 'ridge_regression')

# logistic regression	
def lr_run(initial_w, max_iters, gamma):
    """ Logistic regression running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first 
        part, then the whole simulation takes part in 3 steps :
            1. Load the training data.
            2. Train the model.
            3. Make prediction on the testing data.
    """

    #1. LOAD THE DATA
    y, tX, ids = load_data()
    
    #2. TRAIN THE MODEL
    #Let us now clean the input
    tX, median_tr, mean_tr, std_tr = clean_input(tX)

    weights, loss = logistic_regression(y, tX, initial_w, max_iters, gamma)
    print('Weights on whole set\n',weights,'\nLoss',loss)
    
    #4. TEST THE MODEL AND EXPORT THE RESULTS
    test_model(weights, median_tr, mean_tr, std_tr, 'logistic_regression')


# regularized logistic regression
def rlr_run(lambda_, initial_w, max_iters, gamma):
    """ Regularized logistic regression running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first 
        part, then the whole simulation takes part in 3 steps :
            1. Load the training data.
            2. Train the model.
            3. Make prediction on the testing data.
    """
    
    
    #1. LOAD THE DATA
    y, tX, ids = load_data()
    
    #2. TRAIN THE MODEL
    #Let us now clean the input
    tX, median_tr, mean_tr, std_tr = clean_input(tX)

    weights, loss = regularized_logistic_regression(y, tX, lambda_, initial_w, max_iters, gamma)
    print('Weights on whole set\n',weights,'\nLoss',loss)
    
    #3. TEST THE MODEL AND EXPORT THE RESULTS
    test_model(weights, median_tr, mean_tr, std_tr, 'reg_logistic_regression')