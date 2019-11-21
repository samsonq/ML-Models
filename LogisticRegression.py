"""
Author: Samson Qian
"""
import numpy as np 
import keras


class LogisticRegression:
    """
    """
    
    def __init__(self):
        self.c_0 = 0
        self.c_1 = 0
    
    def fit(self, features, labels):
        """
        Takes dataset with labels of 1s and 0s for prediction.
        """
        labels += np.sign(0.5 - labels)*0.001
        labels = y_transform(labels)
        n = len(features)
        self.c_1 = ((x * y).sum() - 1/n * y.sum() * x.sum())/((x**2).sum() - 1/n * x.sum()**2)
        self.c_0 = 1/n * (y.sum() - c_1 * x.sum())
        
    def predict(self, feature):
        """
        """
        probability = fitted_function(feature, self.c_0, self.c_1)
        if probability >= 0.5:
            return 1
        else:
            return 0
        
    def y_transform(self, y):
        """
        """
        return np.log((1-y)/y)
    
    def fitted_function(self, x, c0, c1):
        """
        """
        return 1/(1 + np.exp(c0 + c1 * x))
