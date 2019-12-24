"""
Author: Samson Qian
"""
import numpy as np


class LinearRegression:
    """
    Model to perform linear regression on a dataset with any model in linear from that uses
    products of weights and multiples of x. Can contain multiple features for multiple regression.
    Contains methods for training and predicting. Linear algebra based approach to calculate
    optimal weights to minimize MSE loss function.
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
        x = np.hstack((np.ones((x.shape[0], )).reshape(x.shape[0], 1), x.reshape(x.shape[0], 1)))
        self.weights = np.linalg.inv((x.transpose().dot(x))).dot(x.transpose()).dot(y)
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
        x = np.hstack((np.ones(x.shape[0]), x))
        return x.dot(self.weights)

    def get_weights(self):
        """
        Getter method that returns the weights of the model.
        :return: weights of model
        """
        return self.weights


def test():
    model = LinearRegression()
    xx = np.array([1, 2, 3, 4, 5])
    yy = np.array([2, 4, 6, 8, 10])

    model.fit(xx, yy)
    print(model.predict([3]))


test()
