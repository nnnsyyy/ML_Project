import numpy as np
import costs as co
import gd_helpers as gd_h
import sgd_helpers as sgd_h
import lr_helpers as lr_h
from helpers import *
from proj1_helpers import *
from build_polynomial import *

### GRADIENT DESCENT

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """
    Implements gradient descent.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @param initial_w : the intial guess
        @param max_iters : the maximum number of iterations that the algorithm will run for
        @param gamma: the size of the step in each of the iterations
        @return : weights that describe the generated model and the loss associated with them
    """
    w = initial_w
    for n_iter in range(max_iters):
        w = w - gamma * gd_h.compute_gradient_MAE(y, tx, w)
        loss = co.compute_loss(y, tx, w)
        print("Gradient Descent({bi}/{ti}): loss={l}".format(bi = n_iter, ti = max_iters - 1, l = loss))
           
    return (w, loss)

### STOCHASTIC GRADIENT DESCENT

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """
    Implements stochastic gradient descent with the batch size of 1.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @param initial_w : the intial guess
        @param max_iters : the maximum number of iterations that the algorithm will run for
        @param gamma: the size of the step in each of the iterations
        @return : weights that describe the generated model and the loss associated with them
    """
    shuffle = True

    w = initial_w
    n_iter = 0
    batch_size = 1
    np.random.seed(1)    
    
    while(n_iter < max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, np.random.random_integers(max_iters), batch_size, shuffle):
            w = w - gamma * sgd_h.compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = co.compute_loss(minibatch_y, minibatch_tx, w)
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(bi = n_iter, ti = max_iters - 1, l = loss))
            n_iter += 1
            if(n_iter >= max_iters):
                return (w, loss)
            
    return (w, loss)

### LEAST SQUARES

def least_squares(y, tx):
    """
    Implements lleast squares.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @return : weights that describe the generated model and the loss associated with them
    """
    xtx = np.dot(tx.T, tx)
    w = np.linalg.solve(xtx, np.dot(tx.T, y))
    loss = co.compute_loss(y, tx, w)
    return (w, loss)

### RIDGE REGRESSION

def ridge_regression(y, tx, lambda_):
    """ 
    Implements ridge regression.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @param lambda_ : parameter to penalize the large weights
        @return :  weights that describe the generated model and the loss associated with them
    """
    w = np.linalg.solve(np.dot(tx.T, tx) + lambda_ * np.identity(tx.shape[1]), 
                        np.dot(tx.T, y))
    
    loss = co.compute_loss(y, tx, w)
    
    return (w, loss)

### LOGISTIC REGRESSION

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ 
    Implements logistic regression.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @param initial_w : the intial guess
        @param max_iters : the maximum number of iterations that the algorithm will run for
        @param gamma: the size of the step in each of the iterations
        @return : weights that describe the generated model and the loss associated with them
    """
    threshold = 1e-8
    losses = []

    w = initial_w

    for iter in range(max_iters):
        loss, w = lr_h.learning_by_gradient_descent(y, tx, w, gamma)
        
        if iter % 50 == 0:
            print("\t\tCurrent iteration={i}, the loss={l}".format(i = iter, l = loss))

        losses.append(loss)
        if (len(losses) > 1): 
            if(np.abs(losses[-1] - losses[-2]) < threshold):
                break

    loss = lr_h.calculate_loss(y, tx, w)
    print("\t\tThe loss={l}".format(l = loss))
    return (w, loss)

### REGULARIZED LOGISTIC REGRESSION

def regularized_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ 
    Implements regularized logistic regression.
        @param y : raw output variable 
        @param tx :raw input variable, might be a polynomial basis obtained from the input x
        @param initial_w : the intial guess
        @param max_iters : the maximum number of iterations that the algorithm will run for
        @param gamma: the size of the step in each of the iterations
        @return : weights that describe the generated model and the loss associated with them
    """
    threshold = 1e-8
    losses = []

    w = initial_w
    
    for iter in range(max_iters):

        loss, w = rlr_h.learning_by_gradient_descent(y, tx, w, gamma, lambda_)
        
        if iter % 500 == 0:
            print("\t\tCurrent iteration={i}, the loss={l}".format(i = iter, l = loss))

        losses.append(loss)
        if (len(losses) > 1): 
            if(np.abs(losses[-1] - losses[-2]) < threshold):
                break
    
    loss = rlr_h.calculate_loss(y, tx, w)
    print("\t\tThe loss={l}".format(l = loss))
    return (w, loss)