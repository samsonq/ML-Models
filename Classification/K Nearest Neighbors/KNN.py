"""
Author: Samson Qian
"""
import numpy as np


class KNN:
    """
    A K Nearest Neighbors model to make classifications on inputted data by computing the 'k'
    nearest data points to a new data point and taking the majority class as the vote. Calculates the
    Euclidean distance between 2 points. Option to standardize data before fitting to improve performance.
    """
    def __init__(self, neighbors=1):
        """
        Initialize data of model and number of neighbors to find.
        :param neighbors: number of nearest neighbors
        """
        self.neighbors = neighbors  # number of neighbors (k)
        self.data = None
        self.labels = None

    def fit(self, X, y, standardize=False):
        """
        Fits training data to model for making predictions.
        :param X: array of training features
        :param y: array of training labels
        :param standardize: standardize data
        """
        if standardize:
            X = (X - np.mean(X, axis=0))/np.std(X, axis=0)  # standardize each column
        self.data = X
        self.labels = y

    def predict(self, X):
        """
        Making predictions on the training data.
        :param X: array of test features
        :return: predicted labels of test set
        """
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        """
        Finds nearest neighbors of one data point and makes prediction.
        :param x: one data point
        :return: predicted label of data point
        """
        distances = [self.euclidean_distance(x, i) for i in self.data]
        nearest = np.argsort(distances)[:self.neighbors]
        neighbors = [self.labels[i] for i in nearest]
        return max(neighbors, key=neighbors.count)

    @staticmethod
    def euclidean_distance(x1, x2):
        """
        Calculates the Euclidean distance between 2 data points.
        :param x1: first data point
        :param x2: second data point
        :return: euclidean distance between 2 points
        """
        return np.sqrt(np.sum((x1-x2)**2))
