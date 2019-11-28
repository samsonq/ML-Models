"""
Author: Samson Qian
"""
import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:
    """
    A Logistic Regression model that uses single-featured, labeled data. Contains methods to
    train a model based on data and make predictions to new data. This is a classification
    model where a probability is calculated of a data being (1) or (0).
    """
    
    def __init__(self):
        """
        Class constructor that initializes both weights of the model to 0.
        """
        self.c_0 = 0
        self.c_1 = 0
    
    def fit(self, x, y):
        """
        Takes single-featured dataset with labels of 1s and 0s and fits a model based on the data.
        Update weights of the model through calculations with Linear Algebra and plots out
        the fitted Logistic Regression model accordingly.
        :param x: array of training data
        :param y: array of binary labels
        :returns: weights of the fitted model, c0 and c1
        """
        y += np.sign(0.5 - y) * 0.001  # nudge the data
        y = LogisticRegression.y_transform(y)  # linearize labels through transformation
        n = len(y)  # number of data points
        self.c_1 = ((x * y).sum() - 1/n * y.sum() * x.sum())/((x**2).sum() - 1/n * x.sum()**2)
        self.c_0 = 1/n * (y.sum() - self.c_1 * x.sum())
        xx = np.linspace(min(x) - 1, max(x) + 1, 1000)
        plt.plot(xx, LogisticRegression.fitted_logistic(xx, self.c_0, self.c_1), color="black")
        return self.c_0, self.c_1

    def predict_prob(self, x):
        """
        Given a feature value, x, calculates and returns the probability that it is class 1.
        :param x: feature data
        :return: probability it is 1
        """
        return LogisticRegression.fitted_logistic(x, self.c_0, self.c_1)
        
    def predict(self, x):
        """
        Given a feature value, x, predict and return whether it is class 1.
        :param x: feature data
        :return: 1 or 0
        """
        probability_threshold = 0.5
        probability = LogisticRegression.fitted_logistic(x, self.c_0, self.c_1)
        return 1 if probability >= probability_threshold else 0

    @staticmethod
    def y_transform(y):
        """
        Helper function to linearize the logistic function in order to calculate weights.
        :param y: y value
        :return: linearized logistic value
        """
        return np.log((1-y)/y)

    @staticmethod
    def fitted_logistic(x, c0, c1):
        """
        The linearized logistic function that takes in a data, x, and weights, c0 and c1, and
        calculates the outputted value.
        :param x: x value
        :param c0: weight 0
        :param c1: weight 1
        :return: output of logistic function
        """
        return 1/(1 + np.exp(c0 + c1 * x))
