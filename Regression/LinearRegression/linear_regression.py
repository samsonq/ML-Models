"""
Author: Samson Qian
"""
import numpy as np


class LinearRegression:
    """
    Model to perform linear regression on a dataset with any model in linear from that uses
    products of weights and multiples of x. Can contain multiple features for multiple regression.
    Contains methods for training and predicting.

    Linear algebra-based implementation to calculate optimal weights and minimize MSE loss function.
    """
    def __init__(self):
        """
        Constructor method to initialize weights of Linear Regression model.
        """
        self.weights = np.array([])

    def fit(self, x, y):
        """
        Linear algebra method to calculate optimal weights for minimizing MSE based on
        the normal equation.
        b = (X^T*X)^(-1)*X^T*y
        Creates input matrix X with inputted training features and calculates dot products to
        return an array of weights that satisfies the equation above.
        :param x: tensor of training data containing any number of features
        :param y: array of labels for each input in x
        :return: array of calculated weights for model
        """
        assert len(x) == len(y), "Length of features and labels must be the same"
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        x = np.hstack((np.ones((x.shape[0], )).reshape(x.shape[0], 1), x))
        #self.weights = np.linalg.inv((x.transpose().dot(x))).dot(x.transpose()).dot(y)  if matrix is invertible
        self.weights = np.linalg.solve(x.transpose().dot(x), x.transpose().dot(y))
        return self.weights

    def predict(self, x):
        """
        Given matrix of data, x, make a prediction on the label, y, based on
        the weights calculated through training. Calculates the dot product between
        inputted data and weights to give result of linear model.
        :param x: tensor of data to make predictions on
        :return: prediction of x
        """
        x = np.array(x)
        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        x = np.hstack((np.ones((x.shape[0], )).reshape(x.shape[0], 1), x))
        return x.dot(self.weights)

    def get_weights(self):
        """
        Getter method that returns the weights of the model.
        :return: weights of model
        """
        return self.weights
