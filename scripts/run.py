# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from ridge_regression import *

def run():
    """ridge regression running script. works on the RAW data"""
    
    #0. DEFINE PARAMETERS FOR OUR RUN
    seed = 12
    
    #not possible yet to run polynomial  degrees at the same time.
    degrees = np.array([1])
    k_fold = 4
    lambdas = np.logspace(-1,2,5)
    
    #1. LOAD THE DATA
    print('LOADING THE DATA: ',end=" ")
    DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here 
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print('DONE')
    
    #2. RUN CROSS VALIDATION TO GET BEST LAMBDA
    print('CROSS VALIDATION')
    rmse,lambda_ = cross_validation(y,tX,degrees,lambdas,k_fold,seed)
    #Let us now clean the input
    tX,median_tr = sanitize_NaN(tX)
    tX,mean_tr,std_tr = standardize(tX)
    tX = build_poly(tX,degrees[0])
    
    #3. TRAIN THE MODEL
    weights = ridge_regression(y, tX, lambda_[0])

    print('Weights on whole set\n',weights)
    
    #4. TEST THE MODEL AND EXPORT THE RESULTS
    DATA_TEST_PATH = 'data/test.csv'  # Download train data and supply path here 
    print('IMPORTING TESTING DATA :',end=" ")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print('DONE')
    
    tX_test_sorted,median_vec = sanitize_NaN(tX_test,median_tr)
    tX_test_sorted,mean_tr,std_tr = standardize(tX_test_sorted,mean_tr,std_tr)
    tX_test_sorted = build_poly(tX_test_sorted, degrees[0])
    OUTPUT_PATH = 'data/output_sanitized_normalization_degree1_lambda_finer_test.csv' # Fill in desired name of output file for submission
    print('EXPORTING TESTING DATA WITH PREDICTIONS :',end=" ")
    y_pred = predict_labels(np.array(weights), np.array(tX_test_sorted))
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('DONE')

run()
