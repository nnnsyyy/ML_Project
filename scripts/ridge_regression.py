# -*- coding: utf-8 -*-
"""a function used to compute the solution of the normal equation for the ridge regression."""

import numpy as np
from proj1_helpers import predict_labels
from build_polynomial import build_poly
from plots import cross_validation_visualization
from helpers import *

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    return np.linalg.solve(np.dot(tx.T,tx)+lamb*np.identity(tx.shape[1]),np.dot(tx.T,y))#/(2*len(tx))

def cross_validation(y,tX,degrees,lambdas,k_fold,seed):
    """
        Computes the cross_validation for the given parameters and returns the best result for each polynomial degree.
	The best result will be the one associated with the lambda minimizing the classification error, i.e. the percentage of failures in the retrieval process.
        Note that we give the RAW data to the cross_validation, without any transformation on them.
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # cross validation:    
    class_best = np.zeros(len(degrees))
    class_best_lambda = np.zeros(len(degrees))
    for j,degree in enumerate(degrees):
        
        print('\n Testing for a polynomial of degree ', degree)
        #Training and testing errors for each lambda, so we are able to visualize them afterwards.
        class_tr = np.zeros(len(lambdas))
        class_te = np.zeros(len(lambdas))
        
        for i,lambda_ in enumerate(lambdas):
            print('lambda=',round(lambda_,6),end=", ")
            
            #This is actually where the k-fold cross-validation is computed. We sum all the errors and then average them. 
            loss_tr_tot=0
            loss_te_tot=0
            for k in range(k_fold+1):
                loss_tr_tmp,loss_te_tmp =cross_validation_rr(y,tX,k_indices,k,lambda_,degree)
                loss_tr_tot += loss_tr_tmp
                loss_te_tot += loss_te_tmp
                
            class_tr[i] = loss_tr_tot/k_fold
            class_te[i] = loss_te_tot/k_fold
            print('Percentage of classification error : ',class_te[i])
        class_best[j] = min(class_te)
        class_best_lambda[j] = lambdas[int(np.argmin(class_te))]
        cross_validation_visualization(lambdas, class_tr, class_te)
        
    print('\nBest error :',class_best)
    print('Best lambda :',class_best_lambda)
    return class_best,class_best_lambda


def cross_validation_rr(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for each step of the k-fold cross validation."""
    
    #1. WE DIVIDE THE DATA IN THE SUBGROUPS
    # get k'th subgroup in test, others in train: 
    x_test = np.array(x[k_indices[k-1]])
    y_test = np.array(y[k_indices[k-1]])
    x_train = np.empty((0,x.shape[1]))
    y_train =  np.empty((0,1))
    #This for loops gets the other groups
    for k_iter,validation_points in enumerate(k_indices):
        if(k_iter!=k-1):
            x_train=np.append(x_train,x[validation_points],axis=0)
            y_train=np.append(y_train,y[validation_points])

    #2. WE FORMAT THE DATA            
    #we sanitize and standardize our training data here, and apply the same median, mean and variance to the testing data  
    x_train,median_train = sanitize_NaN(x_train)
    x_test,median_test = sanitize_NaN(x_test,median_train)
    
    x_train,mean_tr,std_tr = standardize(x_train)
    x_test, mean_te,ste_te = standardize(x_test,mean_tr,std_tr)
    
    # form data with polynomial degree:
    x_train_poly = build_poly(x_train,degree)
    x_test_poly = build_poly(x_test,degree)
    #print('Shape of polynomial training date :', x_train_poly.shape)
    
    #3. WE RUN THE MODEL AND COMPUTE THE ERROR
    # ridge regression: 
    w_rr = ridge_regression(y_train,x_train_poly,lambda_)
    
    # calculate the classification error for train and test data:
    loss_tr= sum(abs(y_train-predict_labels(w_rr,x_train_poly)))/(2*len(y_train))
    loss_te = sum(abs(y_test-predict_labels(w_rr,x_test_poly)))/(2*len(y_test))
    
    #MSE error computed here, as the RMSE error is not summable.
    #loss_tr = 2*compute_mse(y_train,x_train_poly,w_rr)
    #loss_te = 2*compute_mse(y_test,x_test_poly,w_rr)
    return loss_tr, loss_te#, loss_tr_class,loss_te_class
