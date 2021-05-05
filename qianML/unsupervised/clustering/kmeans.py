import numpy as np


class KMeans:
    """

    """
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.clusters = np.array([])

    def fit(self, data, max_iter=100):
        """
        Performs Lloyd's algorithm to fit KMeans.
        :param data: array of data points
        :param max_iter: iterations for fitting
        :return: clusters
        """
        dimensionality = data.shape[0]
        self.clusters = ...  # initialize random clusters
        for _ in range(max_iter):
            for point in data:
                for cluster in self.clusters:
                    distance = np.linalg.norm(cluster-point)

        return self.clusters
