import numpy as np

# Helper methods for regularized logistic regression

def sigmoid(t):
    """
    Applies the sigmoid function on t.
    """
    sig = 1 / (1 + np.exp(-t))
    return sig
    
def calculate_loss(y, tx, w, lambda_):
    """
    Computes the cost by negative log likelihood, adds a penalization term
    and normalizes the cost function.
    """
    xw = np.squeeze(np.dot(tx, w))
    res = np.where(xw <= 20, np.log(1 + np.exp(xw)), xw)
    return 1 / len(y) * (np.sum(res) - np.dot(y, xw)) + lambda_ * np.dot(w,w)
    
def test (xw):
    print(xw)
    xw = np.log(1 + np.exp(xw))
    return xw    
    
def calculate_gradient(y, tx, w, lambda_):
    """
    Computes the gradient of loss.
    """
    sigma = np.squeeze(sigmoid(np.dot(tx, w)))
    return 1 / len(y) * np.dot(tx.T, sigma - y) + 2 * lambda_ * w

def learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Does one step of gradient descent using regularized logistic regression.
    Returns the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w, lambda_) 
    grad_w = calculate_gradient(y, tx, w, lambda_)
    w = w - gamma * grad_w
    return loss, w