import numpy as np

# GRADIENT DESCENT HELPER METHODS

def compute_gradient_MAE(y, tx, w):
    """Compute the gradient.""" 
    e = y-np.dot(tx,w)    
    # Gradient for MAE
    return -1/len(y)*np.dot(np.transpose(tx),np.sign(e))