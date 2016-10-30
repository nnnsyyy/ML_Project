# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from least_squares import *

def run():
    """ Least square running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first part, then the whole simulation takes part in 3 steps :
            1. Load the training data.
            2. Train the model. 
            3. Make prediction on the testing data.
    """
    
    #1. LOAD THE DATA
    print('LOADING THE DATA: ',end=" ")
    DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print('DONE')

    
    #2. TRAIN THE MODEL
    #Let us now clean the input
    tX = count_NaN(tX)
    tX,median_tr = sanitize_NaN(tX)
    tX,mean_tr,std_tr = standardize(tX)

    weights, loss = least_squares(y, tX)

    print('Weights on whole set\n',weights)
    
    #3. TEST THE MODEL AND EXPORT THE RESULTS
    DATA_TEST_PATH = '../data/test.csv'  # Download train data and supply path here 
    print('IMPORTING TESTING DATA :',end=" ")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print('DONE')
    
    tX_test_sorted = count_NaN(tX_test)
    tX_test_sorted,median_vec = sanitize_NaN(tX_test_sorted,median_tr)
    tX_test_sorted,mean_tr,std_tr = standardize(tX_test_sorted,mean_tr,std_tr)
    OUTPUT_PATH = '../data/results_ls.csv' 
	# Fill in desired name of output file for submission
    print('EXPORTING TESTING DATA WITH PREDICTIONS :',end=" ")
    y_pred = predict_labels(np.array(weights), np.array(tX_test_sorted))
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('DONE')

run()
