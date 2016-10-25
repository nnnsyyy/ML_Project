# -*- coding: utf-8 -*-
"""a function used to compute the solution for the logistic regression."""

import numpy as np
from proj1_helpers import predict_labels
from build_polynomial import build_poly
from plots import cross_validation_visualization
from helpers import *

def sigmoid(t):
    """apply sigmoid function on t."""
    sig = 1/(1+np.exp(-t))
    return sig
    
def calculate_loss(y, tx, w,lambda_):
    """computes the cost by negative log likelihood and adds a penalization term
       Normalizes the cost function as well
    
    """
    xw = np.squeeze(np.dot(tx,w))
    res = np.where(xw <= 20, np.log(1+np.exp(xw)),xw)
    return 1/len(y)*(np.sum(res) - np.dot(y,xw)) + lambda_*np.dot(w,w)
    
def calculate_gradient(y, tx, w,lambda_):
    """compute the gradient of loss."""
    sigma = np.squeeze(sigmoid(np.dot(tx,w)))
    return 1/len(y)*np.dot(tx.T,sigma-y) + 2*lambda_*w

def learning_by_gradient_descent(y, tx, w, gamma,lambda_):
    """
    Do one step of gradient descen using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y,tx,w,lambda_) 
    grad_w = calculate_gradient(y,tx,w,lambda_)
    w = w -gamma*grad_w
    return loss, w

def regularized_logistic_regression(y, tx, gamma,lambda_,max_iters):
    """ Implements regularized logistic regression.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @param gamma : parameter for the gradient descent
        @param lambda : parameter to penalize the large weights
        @return : function that computes the weights that best fit the data given as input
    
    """
    threshold = 1e-8
    losses = []

    # build tx
    w = np.zeros((tx.shape[1]))

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma,lambda_)
        # log info
        #if iter % 50 == 0:
        print("\t\tCurrent iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if (len(losses) > 1): 
            if(np.abs(losses[-1] - losses[-2]) < threshold):
                break
    # visualization
    print("\t\tThe loss={l}".format(l=calculate_loss(y, tx, w,lambda_)))
    return w
    
def cross_validation(y,tX,degrees,gamma,lambdas,max_iters,k_fold,seed):
    """
        Uses the cross_validation to find the best of the the given parameters and returns the best result (degree, error and lambda)
        The best result will be the one associated with the gamma minimizing the classification error, i.e. the percentage of failures in the retrieval process.
        Note that we give the RAW data to the cross_validation, without any transformation on them.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @param degrees : a vector containing the different polynomial degrees for the polynomial basis (i.e. we want to return the degree that best fits the data)
        @param gamma : step size for the gradient descent
        @param lambdas : vector of penalization parameters that we want to test on : we return the best one
        @param max_iters : the maximum number of iterations
        @param k_fold : the number of groups in which we partition the data for the cross validation
        @param seed : the seed for the random number generation
        @return best_degree_final : the degree of the polynomial basis that best fits the data
        @return best_gamma_final : the gamma that minimizes the error for the best_degree_final polynomial basis
        @return best_error_final : the classification error done by our data, i.e. the percentage of mismatches
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # cross validation:      
    best_lambda = np.zeros(len(degrees))
    best_error = np.zeros(len(degrees))
    for j,degree in enumerate(degrees):
        
        print('\n Testing for a polynomial of degree ', degree)
        #Training and testing errors for each gamma, so we are able to visualize them afterwards.
        class_error_tr = np.zeros(len(lambdas))
        class_error_te = np.zeros(len(lambdas))
        
        for i,lambda_ in enumerate(lambdas):
            print('lambda=',round(lambda_,6),end=", ")
            
            #This is actually where the k-fold cross-validation is computed. We sum all the errors and then average them. 
            loss_tr_sum=0
            loss_te_sum=0
            for k in range(k_fold+1):
                loss_tr_tmp,loss_te_tmp =cross_validation_lr(y,tX,k_indices,k,gamma,lambda_,max_iters,degree)
                loss_tr_sum += loss_tr_tmp
                loss_te_sum += loss_te_tmp
                
            class_error_tr[i] = loss_tr_sum/k_fold
            class_error_te[i] = loss_te_sum/k_fold
            print('Percentage of classification error : ',class_error_te[i])
        best_error[j] = min(class_error_te)
        best_lambda[j] = lambdas[int(np.argmin(class_error_te))]
        cross_validation_visualization(lambdas, class_error_tr, class_error_te,degree)
        
    best_error_final = min(best_error)
    print(best_error_final.shape)
    print(np.argmin(best_error))
    best_lambda_final = best_lambda[int(np.argmin(best_error))]
    best_degree_final = degrees[int(np.argmin(best_error))]
        
    print('\nBest degree :',best_degree_final)
    print('Best error :',best_error_final)
    print('Best lambda :',best_lambda_final)
    return best_degree_final,best_lambda_final,best_error_final


def cross_validation_lr(y, x, k_indices, k, gamma,lambda_,max_iters, degree):
    """ Return the classification error of the logistic regression for each step of the k-fold cross validation.
    
    @param y : raw output variable 
    @param x :raw input variable, might be a polynomial basis obtained from the input x
    @param k_indices : the indices of the data that belong to each of the K groups of the cross_validation.
    @param k : the index of the group that we are using for the testing.
    @param gamma : the gamma with which we're doing the cross_validation
    @param lambda : the penalization parameter we're working on.
    @param max_iters : the max number of iterations of the logistic regression
    @param degree : the degree of the polynomial basis with which we're doing the cross validation
    @return loss_tr : the classification error made on the training data.
    @return loss_te : the classification error made on the testing data.
    """
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
    x_train = count_NaN(x_train)
    x_test = count_NaN(x_test)    
    
    x_train,median_train = sanitize_NaN(x_train)
    x_test,median_test = sanitize_NaN(x_test,median_train)
    
    x_train,mean_tr,std_tr = standardize(x_train)
    x_test, mean_te,ste_te = standardize(x_test,mean_tr,std_tr)
    
    # form data with polynomial degree:
    x_train_poly = build_poly(x_train,degree)
    x_test_poly = build_poly(x_test,degree)
    #print('Shape of polynomial training date :', x_train_poly.shape)
    
    #3. WE RUN THE MODEL AND COMPUTE THE ERROR
    # Relgularized logistic regression: 
    w_rlr = regularized_logistic_regression(y_train,x_train_poly,gamma,lambda_,max_iters)
    
    # calculate the classification error for train and test data:
    loss_tr= sum(abs((2*(y_train)-1)-predict_labels(w_rlr,x_train_poly)))/(2*len(y_train))
    loss_te = sum(abs((2*y_test-1)-predict_labels(w_rlr,x_test_poly)))/(2*len(y_test))
    
    return loss_tr, loss_te
