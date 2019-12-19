"""
Author: Samson Qian
"""
import numpy as np


class LinearRegression:
    """

    """

    def __init__(self):
        """
        Constructor method to initialize weights of Linear Regression model.
        """
        self.weights = np.array([])

    def fit(self, x, y):
        """

        :param x:
        :param y:
        :return:
        """
        x = np.hstack((np.ones(x.shape[0], 1), x))
        self.weights = ((x.transpose().dot(x))**(-1)).dot(x.transpose()).dot(y)

    def predict(self, x):
        """

        :param x:
        :return:
        """
        x = np.hstack((np.ones(x.shape[0], 1), x))
        return x.dot(self.weights)

    def get_weights(self):
        """

        :return:
        """
        return self.weights
