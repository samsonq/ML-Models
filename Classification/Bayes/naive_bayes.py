import numpy as np


class NaiveBayes:
    """
    Naive Bayes is a classifier that assumes mutual independence among the features of the data. This
    assumption allows feasible estimation of probabilities using Bayes' Theorem by separating multiple
    conditions into single conditions. This addresses the curse of dimensionality in which more features
    require significantly more data to make probability calculations.

    This implementation takes only categorical features to make a prediction. For numerical features, use
    Gaussian or Multinomial Naive Bayes.
    """
    def __init__(self):
        """

        """
        self.data = None
        self.labels = None
        self.classes = None

    def fit(self, X, y):
        """

        :param X: feature data array
        :param y: label data array
        """
        self.data = X
        self.labels, self.classes = y, np.unique(y)

    def predict(self, X):
        """
        Predict class using Bayes' Theorem and independence assumption.
        :param X: feature data array
        :return: class prediction
        """
        max_probability = 0
        prediction = np.zeros(shape=(X.shape[0], 1))
        for category in self.classes:
            category_probability = np.sum(self.labels==category)/self.labels.shape[0]
            feature_probabilities = np.array([])
            for d in X:
                for i, feature in enumerate(d):
                    feature_probabilities = np.append(feature_probabilities,
                                                      np.sum(self.data.T[i]==feature)/self.data.shape[0])
            if category_probability * np.prod(feature_probabilities) > max_probability:
                max_probability = category_probability * np.prod(feature_probabilities)
                prediction = category
        return prediction


class GaussianNaiveBayes(NaiveBayes):
    """
    Gaussian Naive Bayes is an extension of Naive Bayes in which a Gaussian distribution is fit on
    numerical features in the data to estimate probabilities, rather than direct computation, which is
    often infeasible for continuous variables.
    """
    def __init__(self):
        super().__init__()
        self.gaussian = None

    def fit(self, X, y):
        """

        :param X:
        :param y:
        :return:
        """