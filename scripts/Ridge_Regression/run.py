# Useful starting lines
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *
from ridge_regression import *
from helpers import *

def run():
    """ Ridge regression running script. It is self-contained.
        Complete the whole pipeline of the simulation. The parameters are given in the first part, then the whole simulation takes part in 4 steps :
            0. Set the parameters :
                - seed :    seed for the random number generation.
                - k_fold :  the number of samples we have for the cross_validation
                - degrees : the degrees of the polynomial we want to test on.
                - lambdas : the range of lambdas we want to do grid search on.
                N.B. All those are multidimensional arrays, as we want to be able to indepentently choose parameters for each of the splits
            1. Load the training data.
            2. Splits the dataset
            3.Use cross_validation to estimate the error in order to pick the lambda and polynomial degree with the least error for
                each split of our data and store the best w and lambda for each split. The parameters change for each split.
            4. Train each model on the best polynomial degree and lambda.
            5. Make prediction on the testing data.
    """
    
    #0. DEFINE PARAMETERS FOR OUR RUN
    seed = 1
    
    #not possible yet to run polynomial  degrees at the same time.
    degrees = np.array([[8,9,10,11],[8,9,10,11,12,13],[13,14,15,16],[12,13,14,15]])
    k_fold = 4
    lambdas = [np.logspace(-2,1,30),np.logspace(-1,1,25),np.logspace(-1,3,25),np.logspace(0,2,25)]

    export_file="test_split_data_param_5_mse"
    #1. LOAD THE DATA
    print('LOADING THE DATA: ',end=" ")
    DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print('DONE')
    
    #2. SPLITTING THE DATA
    
    print('SPLITTING THE DATA: ',end=" ")  
    degree_split = list()
    weight_split = list()
    error_split = list()
    lambda_split = list()
    median_split = list()
    mean_split = list()
    std_split = list()
    
    y_split,tx_split,id_split = split_dataset(y,tX,ids)
    print('DONE')   
    #3. RUN CROSS VALIDATION TO GET BEST LAMBDA
    
    for split,(y_s,tx_s,id_s) in enumerate(zip(y_split,tx_split,id_split)):
        #To make sure they are arrays of the correct dimension
        y_s = np.squeeze(y_s)
        tx_s = np.squeeze(tx_s)
        print('\n\tCROSS VALIDATION FOR SPLIT NUMBER',split)
        #Perform cross validation and save best output
        best_degree, best_lambda_, best_error = cross_validation(y_s,tx_s,degrees[split],lambdas[split],k_fold,seed,split)
        degree_split.append(best_degree)
        lambda_split.append(best_lambda_)  
        error_split.append(best_error)
        
        #4. TRAIN THE MODELS
        #Let us now clean the input
        tx_s = count_NaN(tx_s)
        tx_s,median_tr = sanitize_NaN(tx_s)
        tx_s,mean_tr,std_tr = standardize(tx_s)
        tx_s = build_poly(tx_s,best_degree)
        print('Size of the vectors',y_s.shape,tx_s.shape)
        weights = ridge_regression(y_s, tx_s, best_lambda_)
        
        #Save the calculation of the weights,median,mean,std for each model
        weight_split.append(weights)
        median_split.append(median_tr)
        mean_split.append(mean_tr)
        std_split.append(std_tr)
        
    print('Degrees',degree_split)
    print('Lambdas',lambda_split)
    print('Errors',error_split)
    #5. TEST THE MODEL AND EXPORT THE RESULTS
    prediction_data(median_split,mean_split,std_split,degree_split,weight_split,export_file)

    
    
def prediction_data(median_split,mean_split,std_split,degrees_split,weight_split,export_file):
    """
        Computes the prediction part of the machine learning algorithm, the fifth part of the pipeline described above
            5. Make prediction on the testing data.
                a. Split them
                b. Do the prediction on each small model
                c. Merge the prediction and export it to a .csv file  
        @param median_split : tuple containing the vectors of medians computed on each of the splits in training data
        @param mean_split : tuple containing the vectors of means computed on each of the splits in training data
        @param std_split : tuple containing the vectors of standard deviations computed on each of the splits in training data
        @param degrees_split : vector containing the best degree for the polynomial basis of each of the splits
        @param weight_split : tuple containing the vectors of weights computed on each of the splits in training data
        @param export_file : the string we add to the export data to make sure it is uniquely identifiable
        """
    DATA_TEST_PATH = '../data/test.csv'  # Download train data and supply path here 
    print('\nIMPORTING TESTING DATA :',end=" ")
    y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
    print('DONE')
    
    #5.a. Splitting the testing data
    print('SPLITTING TESTING DATA :',end=" ")
    y_test_split,tx_test_split,id_test_split = split_dataset(y_test,tX_test,ids_test)    
    print('DONE')    
    #5.b. prediction on each model
    y_pred = list()
    
    for split,(y_test_s,tx_test_s,id_test_s) in enumerate(zip(y_test_split,tx_test_split,id_test_split)):  
        print('PREDICTION FOR TESTING DATA SPLIT NUMBER',split)
        
        #Formatting to the correct datatype
        y_test_s = np.squeeze(y_test_s)
        tx_test_s = np.squeeze(tx_test_s)
        id_test_s = np.squeeze(id_test_s)
        print('Size of the vectors',y_test_s.shape,tx_test_s.shape) 
        #Formatting the data themselves
        print('Counting NaN',end='. ')
        tx_test_s = count_NaN(tx_test_s)
        print('Sanitizing',end = ' . ')
        tx_test_s,median_vec = sanitize_NaN(tx_test_s,median_split[split])
        print('Standardizing',end = ' .')
        tx_test_s,mean_te,std_te = standardize(tx_test_s,mean_split[split],std_split[split])
        print('Building polynomial basis')        
        tx_test_s = build_poly(tx_test_s, degrees_split[split])
        
        #Prediction
        y_pred.append(predict_labels(np.array(weight_split[split]), np.array(tx_test_s)))   
    
    print('MERGING TESTING DATA',end="")
    y_pred_merged, ids_merged = merge_dataset(y_pred,id_test_split)
    print('DONE')
    
    OUTPUT_PATH = 'results/output_sanitized_normalization_'+export_file+'.csv' 
    print('EXPORTING TESTING DATA WITH PREDICTIONS :',end=" ")
 
    create_csv_submission(ids_merged, y_pred_merged, OUTPUT_PATH)
    print('DONE')    
    
    
run()
