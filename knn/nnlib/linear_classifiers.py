import numpy as np
from nnlib.linear_svm_loss import *

class LinearClassifier(object):
    
    def __init__(self):
        self.W = None
    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iterators=100, 
             batch_size=200, verbose=False):
        """
        X: num_images*3073
        y: num_images*10
        
        outputs: list of loss value for every iteration
        """
        # 初始化W
        # 随机选取 子采样的 batch 个样本及对应label
        # 计算出loss以及梯度下降的变化
        # 将W按照梯度下降的方向更新.
        if self.W is None:
            self.W = 0.001 * np.random.randn(X.shape[1], y.shape[0])
        
        num_images = X.shape[0]
        num_labels = y.shape[0]
        loss_history = []
        for itr in range(num_iterators):
            # step1 subsamples
            batch_index = np.random.choice(num_images, batch_size, replace=True)
            batch_x = X[batch_index]
            batch_y = y[batch_index]
            loss, grad = self.loss(batch_x, batch_y, reg)
            loss_history.append(loss)
            
            self.W += -grad * learning_rate
            
            if verbose and itr % 100 == 0:
                print("iteration/allIterations, loss", itr, num_iterators, loss)
        
        return history_loss
    
    def predict(self,X):
        """
        score = X*W, 最大的那个返回label.
        """
        y_pred = np.zeros(X.shape[0])
        score_m = X.dot(self.W)
        y_pred = np.argmax(score_m, axis=1)
        
        return y_pred
    
    def loss(self,X_batch, y_batch, reg):
        pass
    
    
class LinearSVM(LinearClassifier):
    """重写LinearClassifier的函数"""
    
    def loss(self, X_batch, y_batch, reg):
        return msvm_loss_vector(self.W, X_batch, y_batch, reg=0.01, deltaD=1)
    
        
        
