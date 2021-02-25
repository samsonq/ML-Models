import numpy as np


class LogReg:
    """
    A Logistic Regression model that uses multi-featured, labeled data. Contains methods to
    train a model based on data and make predictions to new data. This is a classification
    model where a probability is calculated of a data being (1) or (0).
    """
    def __init__(self, learning_rate=0.1, iterations=100, probability_threshold=0.5):
        """
        Define logistic hyperparameters.
        :param learning_rate: training learning rate
        :param iterations: training iterations
        :param probability_threshold: classification threshold
        """
        self.lr = learning_rate
        self.iter = iterations
        self.threshold = probability_threshold
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        """
        Fits logistic regression weights based on inputted training data.
        :param X: array of training data
        :param y: array of training labels
        """
        self.weights = np.zeros(X.shape[1])
        for _ in range(self.iter):
            ## TODO
            p = self.sigmoid((np.dot(X, self.weights)) + self.bias)
            weight_change = (1) / X.shape[0]
            bias_change = (1) / X.shape[0]
            self.weights -= self.lr * weight_change
            self.bias -= self.lr * bias_change
        return

    @staticmethod
    def sigmoid(x):
        """
        The sigmoid function.
        :param x: input
        :return: sigmoid output
        """
        return 1/(1+np.exp(-1*x))

    def predict(self, X):
        """
        Predict labels of inputted data using weights and bias.
        :param X: data
        :return: classifications (1 or 0)
        """
        probabilities = self.sigmoid((np.dot(X, self.weights)) + self.bias)
        return probabilities > self.threshold
