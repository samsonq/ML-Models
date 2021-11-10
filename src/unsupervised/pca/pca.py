"""
Implementation of Principal Component Analysis
Author: Samson Qian
"""
import numpy as np


class PCA:
    """
    Principal Component Analysis
    """

    def __init__(self, n_components):
        """
        Initialize PCA variables.
        :param n_components: number of principal components
        """
        self.n_components = n_components
        self.components = None
        self.mean = 0

    def fit(self, X):
        """
        Fit PCA to find top n eigenvectors and create components.
        :param X: input data
        :return: top n eigenvectors (principal components)
        """
        # Mean Centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean  # subtract mean from every column in X

        # covariance (needs samples as columns)
        cov = np.cov(X.T)  # create covariance matrix of X

        # compute eigenvalues, eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov)  # eigenvectors of covariance matrix

        # -> eigenvector v = [:, i]
        # transpose eigenvectors for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors by eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]  # sorted eigenvectors by eigenvalues

        # store top n eigenvectors
        self.components = eigenvectors[:self.n_components]  # pick top n eigenvectors as principal components
        return self.components

    def transform(self, X):
        """
        Transform X by taking dot product with fitted principal components.
        :param X: input data
        :return: transformed data
        """
        # project input data to new dimension
        X = X - self.mean()
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        """
        Fit, then transform input data based on principal components.
        :param X: input data
        :return: transformed data
        """
        self.fit(X)
        return self.transform(X)
