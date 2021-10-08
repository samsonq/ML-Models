"""
Author: Samson Qian
"""
import numpy as np


class PCA:
    """
    pca that maps a dataset with 2 features and combines them into one feature.
    The new feature is defined as the dot product between the top eigenvector of X(X^T) and the feature
    vector of the data (x1, x2).
    """
    def __init__(self):
        """
        Constructor method that initializes the top eigenvector of the data.
        """
        self.top_eigenvector = np.array([])

    def fit(self, x1, x2):
        """
        Fits the inputted features into the PCA by standardizing each feature according to mean, and
        determining the top eigenvector of the matrix X(X^T).
        :param x1: array of data for first feature
        :param x2: array of data for second feature
        """
        if type(x1) == "list":
            x1 = np.array(x1)
        if type(x2) == "list":
            x2 = np.array(x2)
        x1 = x1 - np.mean(x1)  # center the x data by subtracting mean
        x2 = x2 - np.mean(x2)  # center the y data by subtracting mean
        X = np.array([x1, x2])
        X_transpose = X.transpose()
        eigenvectors = np.linalg.eigh(X.dot(X_transpose))
        self.top_eigenvector = eigenvectors[-1][-1]  # positive values of the top eigenvector of X(X^T)

    def map(self, x):
        """
        Takes a feature vector of size 2 and maps it to a single value through the dot product with
        the top eigenvector calculated by fitting the data.
        :param x: feature of vector of size 2
        :return: single mapped value
        """
        x = np.array(x)
        return x.dot(self.top_eigenvector)

    def get_eigen(self):
        """
        Getter method to retrieve the top eigenvector calculated by fitting data.
        :return: top eigenvector of X(X^T)
        """
        return self.top_eigenvector
