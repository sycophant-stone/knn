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
    loss = 0.0
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

def msvm_loss_vector(W, X, y, reg=0.01, deltaD=1):
    """
    以前是分开算第i个图像的所有的WX值.这次利用矩阵一下算出来.
    """
    num_labels = W.shape[1]
    num_images = X.shape[0]
    loss = 0.0
    dw = np.zeros(W.shape) # 3073x10 dw matix
        
    s_m = X.dot(W) #num_imagex10, 每行是一幅图在对应算子中的权重值.
    correct_score_m = s_m[range(num_images), list(y)].reshape(-1,1) # num_trainx1, 描述对应正确label的得分
    margin_m = np.maximum(0, s_m - correct_score_m + deltaD)
    margin_m[range(num_images),list(y)] = 0 # 这部分别忘了
    loss = np.sum(margin_m)/num_images + 0.5* reg * np.sum(W * W)
    
    # 先计算一个系数, 不正确的label给1, 正确的lable在0的基础上减去 偏差和的均值.
    coef = np.zeros((num_images, num_labels))
    coef[margin_m > 0] = 1
    coef[range(num_images), list(y)] = 0
    coef[range(num_images), list(y)] = -np.sum(coef, axis = 1)
    
    dw = (X.T).dot(coef) # 为什么要这么做? 所谓梯度,其实描述的是image像素的变化.
    dw = dw/num_images + reg * W
    
    return loss, dw
    
    
    
    
                
            
            
        
        