import numpy as np

def compute_stoch_gradient(y, tx, w):
    """
    Computes a gradient.
    """
    e = y - np.dot(tx, w)
    
    #Gradient for MSE
    #return -1/len(y)*np.dot(np.transpose(tx),e)

    #Gradient for MAE
    return -1 / len(y) * np.dot(np.transpose(tx), np.sign(e))