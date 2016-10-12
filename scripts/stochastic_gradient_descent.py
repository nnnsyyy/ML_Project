# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
import numpy as np
import costs as co

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    e = y-np.dot(tx,w)
    
    #Gradient for MSE
    #return -1/len(y)*np.dot(np.transpose(tx),e)

    #Gradient for MAE
    return -1/len(y)*np.dot(np.transpose(tx),np.sign(e))


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""

    # Parameters for the random part
    shuffle = True
    batch_size = 32

    # Define parameters to store w and loss
    
    ws = [initial_w]
    losses = []
    w = initial_w
    n_iter = 0
    while(n_iter < max_epochs):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size,shuffle):
            w = w-gamma*compute_stoch_gradient(minibatch_y,minibatch_tx,w)
            loss = co.compute_loss(minibatch_y,minibatch_tx,w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_epochs - 1, l=loss, w0=w[0], w1=w[1]))
            n_iter+=1
            if(n_iter >= max_epochs):
                return losses,ws
    return losses, ws
