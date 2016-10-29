import numpy as np

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