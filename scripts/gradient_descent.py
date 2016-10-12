# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs as co


def compute_gradient_MSE(y, tx, w):
    """Compute the gradient.""" 
    e = y-np.dot(tx,w)
    
    # Gradient for MSE
    return -1/len(y)*np.dot(np.transpose(tx),e)

def compute_gradient_MAE(y, tx, w):
    """Compute the gradient.""" 
    e = y-np.dot(tx,w)    
    # Gradient for MAE
    return -1/len(y)*np.dot(np.transpose(tx),np.sign(e))


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
