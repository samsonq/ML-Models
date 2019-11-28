"""
Author: Samson Qian
"""
import numpy as np
from matplotlib import pyplot as plt


class PCA:
    """
    Principal Component Analysis that maps a dataset with 2 features and combines them into one feature.
    The new feature is defined as the dot product between the top eigenvector of X(X^T) and the feature
    vector of the data (x1, x2).
    """
    def __init__(self):
        """
        Constructor method that initializes the top eigenvector of the data.
        """
        self.top_eigenvector = np.array([])

    def fit(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        x = x - np.mean(x)  # center the x data by subtracting mean
        y = y - np.mean(y)  # center the y data by subtracting mean
        X = np.array([x, y])
        X_transpose = X.transpose()
        eigenvectors = np.linalg.eigh(X.dot(X_transpose))
        self.top_eigenvector = abs(eigenvectors[-1])  # positive values of the top eigenvector of X(X^T)

    def map(self, x):
        """

        :param x:
        :return:
        """
        return x.dot(self.top_eigenvector)
