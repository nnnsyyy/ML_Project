import numpy as np

# Helper methods for logistic regression

def sigmoid(t):
    """
    Applies the sigmoid function on t.
    """
    sig = 1 / (1 + np.exp(-t))
    return sig
    
def calculate_loss(y, tx, w):
    """
    Computes the cost by negative log likelihood.
    """
    xw = np.squeeze(np.dot(tx, w))
    res = np.where(xw <= 20, np.log(1 + np.exp(xw)), xw)
    return 1 / len(y) * (np.sum(res) - np.dot(y, xw))
    
def calculate_gradient(y, tx, w):
    """
    Computes the gradient of loss.
    """
    sigma = np.squeeze(sigmoid(np.dot(tx, w)))
    return 1 / len(y) * np.dot(tx.T, sigma - y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Does one step of gradient descen using logistic regression.
    Returns the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad_w = calculate_gradient(y, tx, w)
    w = w - gamma * grad_w
    return loss, w