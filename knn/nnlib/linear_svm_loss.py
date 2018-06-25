import numpy as np
from random import shuffle

def msvm_loss_native(W, X, y, reg=0.01, deltaD=1):
    """
    W: weights, 3073x10
    X: image, num_trainx3073
    y: 10x1
    reg: regularization pena
    """
    # step1 , calc all the scores.
    
    #print("sm",sm)
    num_labels = W.shape[1]
    num_images = X.shape[0]
    dw = np.zeros(W.shape) # 3073x10 dw matix
    for i in range(num_images):
        score_m = X[i,:].dot(W) # score_m 1x10 for all the labels
        for j in range(num_labels):
            if j == y[i]:
                continue
            margin =score_m[j] - score_m[y[i]] + deltaD
            if margin > 0:
                loss += margin
                # calc gradian fo W. the wrong labels add corrensponding Image.
                # the right lables, substract the corresponding image.
                dw[:,j] += X[i].T   
                dw[:,y[i]] += -X[i].T
    
    loss /= num_images # average for all the images.
    loss += 0.5*reg*np.sum(W*W) # add regularization penaty
    
    dw /= num_images
    dw += reg*W
    
    return loss, dw
                
            
            
        
        