"""
Author: Samson Qian
"""
import numpy as np


class KNN:
    """
    A K Nearest Neighbors model to make classifications on inputted data by computing the 'k'
    nearest data points to a new data point and taking the majority class as the vote. Calculates
    the Euclidean distance between points to find closest neighbors.
    """
    def __init__(self, neighbors=1, standardize=False):
        """
        Initialize data of model and number of neighbors to find.
        :param neighbors: number of nearest neighbors
        :param standardize: standardize data to scale
        """
        self.neighbors = neighbors  # number of neighbors (k)
        self.standardize = standardize
        if self.standardize:
            self.mean = 0
            self.std = 0
        self.data = None
        self.labels = None

    def fit(self, X, y):
        """
        Fits training data to model for making predictions.
        :param X: array of training features
        :param y: array of training labels
        """
        if self.standardize:
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
            X = (X - self.mean)/self.std  # standardize each column
        self.data = X
        self.labels = y

    def predict(self, X):
        """
        Making predictions on the training data.
        :param X: array of test features
        :return: predicted labels of test set
        """
        if self.standardize:
            X = (X - self.mean)/self.std
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        """
        Finds nearest neighbors of one data point and makes prediction.
        :param x: one data point
        :return: predicted label of data point
        """
        distances = [self._euclidean_distance(x, i) for i in self.data]
        nearest = np.argsort(distances)[:self.neighbors]
        neighbors = [self.labels[i] for i in nearest]
        return max(neighbors, key=neighbors.count)

    @staticmethod
    def _euclidean_distance(x1, x2):
        """
        Calculates the Euclidean distance between 2 data points.
        :param x1: first data point
        :param x2: second data point
        :return: euclidean distance between 2 points
        """
        return np.sqrt(np.sum((x1-x2)**2))
