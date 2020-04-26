import numpy as np

class NaiveBayes:
    """

    """
    def __init__(self):
        """

        """
        self.data = None
        self.labels = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        """
        self.data = X
        self.labels = y

    def predict(self, X):
        """
        
        :param X:
        :return:
        """
