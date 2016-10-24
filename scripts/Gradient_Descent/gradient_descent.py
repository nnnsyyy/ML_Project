# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs as co
from helpers import *
from proj1_helpers import *
from plots import *
from costs import *


def compute_gradient_MSE(y, tx, w):
    """Compute the gradient.""" 
    e = y-np.dot(tx,w)
    
    # Gradient for MSE
    return -1/len(y)*np.dot(tx.T,e)

def compute_gradient_MAE(y, tx, w):
    """Compute the gradient.""" 
    e = y-np.dot(tx,w)    
    # Gradient for MAE
    return -1/len(y)*np.dot(tx.T,np.sign(e))


def gradient_descent_MSE(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        w = w-gamma*compute_gradient_MSE(y,tx,w)
        loss = co.compute_loss(y,tx,w)
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              #bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def gradient_descent_MAE(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        w = w-gamma*compute_gradient_MAE(y,tx,w)
        loss = co.compute_loss(y,tx,w)
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
            #  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws

def least_squares_GD(y, tX, gamma, max_iters):
    """
    TODO
    """
    #Generating initial guess using the mean of each column
    initial_w = generate_initial_w(tX)
    #ws = [[initial_w]]
    w = initial_w

    for n_iter in range(max_iters):
        w = w-gamma*compute_gradient_MAE(y,tX,w)
        #print(w)
        #ws.append(np.copy(w))
        
    #print(ws)    
    return w
    #return gradient_descent_MSE(y, tX, initial_w, max_iters, gamma)

def cross_validation(y,tX,gammas,max_iters, k_fold,seed):
    """
    TODO
    """

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # cross validation:      
    best_gamma = np.zeros(len(max_iters))
    best_error = np.zeros(len(max_iters))
    for j,max_iter in enumerate(max_iters):
        
        print('\n Testing for max iterations of : ', max_iter)
        #Training and testing errors for each max iterations value, so we are able to visualize them afterwards.
        class_error_tr = np.zeros(len(gammas))
        class_error_te = np.zeros(len(gammas))
        
        for i,gamma in enumerate(gammas):
            print('gamma=',round(gamma,6),end=", ")
            
            #This is actually where the k-fold cross-validation is computed. We sum all the errors and then average them. 
            loss_tr_sum=0
            loss_te_sum=0
            for k in range(k_fold+1):
                loss_tr_tmp,loss_te_tmp =cross_validation_gd(y,tX,k_indices,k,gamma,max_iter)
                loss_tr_sum += loss_tr_tmp
                loss_te_sum += loss_te_tmp
                
            class_error_tr[i] = loss_tr_sum/k_fold
            class_error_te[i] = loss_te_sum/k_fold
            print('Percentage of classification error : ',class_error_te[i])
        best_error[j] = min(class_error_te)
        best_gamma[j] = gammas[int(np.argmin(class_error_te))]
        cross_validation_visualization(gammas, class_error_tr, class_error_te,max_iter)
        
    best_error_final = min(best_error)
    print(best_error_final.shape)
    print(np.argmin(best_error))
    best_gamma_final = best_gamma[int(np.argmin(best_error))]
    best_max_iter_final = max_iters[int(np.argmin(best_error))]
        
    print('\nBest max_iters :',best_max_iter_final)
    print('Best error :',best_error_final)
    print('Best gamma :',best_gamma_final)
    return best_max_iter_final,best_gamma_final,best_error_final


def cross_validation_gd(y, x, k_indices, k, gamma, max_iter):
    """
    TODO
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
    x_train,median_train = sanitize_NaN(x_train)
    x_test,median_test = sanitize_NaN(x_test,median_train)
    
    x_train,mean_tr,std_tr = standardize(x_train)
    x_test, mean_te,ste_te = standardize(x_test,mean_tr,std_tr)
    
    #3. WE RUN THE MODEL AND COMPUTE THE ERROR
    # gradient descent: 
    w_gd = least_squares_GD(y_train,x_train,gamma,max_iter)
    
    # calculate the classification error for train and test data:
    loss_tr= sum(abs(y_train-predict_labels(w_gd,x_train)))/(2*len(y_train))
    loss_te = sum(abs(y_test-predict_labels(w_gd,x_test)))/(2*len(y_test))
    
    #MSE error computed here, as the RMSE error is not summable.
    loss_tr = 2*compute_mse(y_train,x_train,w_gd)
    loss_te = 2*compute_mse(y_test,x_test,w_gd)
    return loss_tr, loss_te#, loss_tr_class,loss_te_class
