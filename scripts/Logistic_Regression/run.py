    # Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from logistic_regression import *

def run():
    """ L regression running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first part, then the whole simulation takes part in 4 steps :
            0. Set the parameters :
                - seed :    seed for the random number generation.
                - k_fold :  the number of samples we have for the cross_validation
                - degrees : the degrees of the polynomial we want to test on.
                - lambdas : the range of lambdas we want to do grid search on.
            1. Load the training data.
            2. Use cross_validation to estimate the error in order to pick the lambda and polynomial degree with the least error.
            3. Train the model on the best polynomial degree and lambda.
            4. Make prediction on the testing data.
    """
    
    #0. DEFINE PARAMETERS FOR OUR RUN
    seed = 1
    
    #not possible yet to run polynomial  degrees at the same time.
    degrees = np.array([3])
    k_fold = 4
    gammas = [0.00000001]#np.logspace(-3,-2,2)
    max_iters = 2000
    #1. LOAD THE DATA
    print('LOADING THE DATA: ',end=" ")
    DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    y = (y+1)/2
    print('DONE')
    
    #2. RUN CROSS VALIDATION TO GET BEST gamma
    print('CROSS VALIDATION')
    #degree, gamma, error = cross_validation(y,tX,degrees,gammas,max_iters,k_fold,seed)
    degree = degrees[0]
    gamma = gammas[0]
    
    #3. TRAIN THE MODEL
    #Let us now clean the input
    tX,median_tr = sanitize_NaN(tX)
    tX,mean_tr,std_tr = standardize(tX)
    tX = build_poly(tX,degree)

    weights = logistic_regression(y, tX, gamma,max_iters)

    print('Weights on whole set\n',weights)
    
    #4. TEST THE MODEL AND EXPORT THE RESULTS
    DATA_TEST_PATH = '../data/test.csv'  # Download train data and supply path here 
    print('IMPORTING TESTING DATA :',end=" ")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print('DONE')
    
    tX_test_sorted,median_vec = sanitize_NaN(tX_test,median_tr)
    tX_test_sorted,mean_tr,std_tr = standardize(tX_test_sorted,mean_tr,std_tr)
    tX_test_sorted = build_poly(tX_test_sorted, degree)
    OUTPUT_PATH = 'results/output_sanitized_normalization_test_deg_2_gamma_8.csv' # Fill in desired name of output file for submission
    print('EXPORTING TESTING DATA WITH PREDICTIONS :',end=" ")
    y_pred = predict_labels(np.array(weights), np.array(tX_test_sorted))
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print('DONE')
    
run()
