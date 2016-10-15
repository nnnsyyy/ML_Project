# -*- coding: utf-8 -*-
"""a function used to compute the solution of the normal equation for the ridge regression."""

import numpy as np
from costs import compute_mse
from build_polynomial import build_poly
from plots import cross_validation_visualization
from helpers import build_k_indices

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    return np.linalg.solve(np.dot(tx.T,tx)+lamb*np.identity(tx.shape[1]),np.dot(tx.T,y))#/(2*len(tx))


def cross_validation(y,tX,degrees,lambdas,k_fold,seed):
    """
        Computes the cross_validation for the given parameters and returns the best result for each polynomial degree.
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # cross validation:    
    rmse_best = np.zeros(len(degrees))
    rmse_best_lambda = np.zeros(len(degrees))
    for j,degree in enumerate(degrees):
        
        print('\n Testing for a polynomial of degree ', degree)
        #Training and testing errors for each lambda, so we are able to visualize them afterwards.
        rmse_tr = np.zeros(len(lambdas))
        rmse_te = np.zeros(len(lambdas))
        
        for i,lambda_ in enumerate(lambdas):
            print('lambda=',round(lambda_,6),end=", ")
            
            #This is actually where the k-fold cross-validation is computed. We sum all the errors and then average them. 
            loss_tr_tot=0
            loss_te_tot=0
            for k in range(k_fold+1):
                loss_tr_tmp,loss_te_tmp =cross_validation_rr(y,tX,k_indices,k,lambda_,degree)
                loss_tr_tot += loss_tr_tmp
                loss_te_tot += loss_te_tmp
                
            rmse_tr[i] = loss_tr_tot/k_fold
            rmse_te[i] = loss_te_tot/k_fold
            print('RMSE_BEST_VALUE : ',rmse_te[i])
        rmse_best[j] = min(rmse_te)
        rmse_best_lambda[j] = lambdas[int(np.argmin(rmse_te))]
        cross_validation_visualization(lambdas, rmse_tr, rmse_te)
        
    print('\nBest error :',rmse_best)
    print('Best lambda :',rmse_best_lambda)
    return rmse_best,rmse_best_lambda


def cross_validation_rr(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for each step of the k-fold cross validation."""
    
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
            
    # form data with polynomial degree:
    x_train_poly = build_poly(x_train,degree)
    x_test_poly = build_poly(x_test,degree)

    # ridge regression: 
    w_rr = ridge_regression(y_train,x_train_poly,lambda_)
    
    # calculate the loss for train and test data:
    loss_tr = np.sqrt(2*compute_mse(y_train,x_train_poly,w_rr))
    loss_te = np.sqrt(2*compute_mse(y_test,x_test_poly,w_rr))
    return loss_tr, loss_te
