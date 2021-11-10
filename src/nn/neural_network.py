"""
Author: Samson Qian
"""
import numpy as np


class SingleNeuron:
    """

    """
    def __init__(self):
        """

        """
        self.weights = np.array([])
        self.bias = 0

    def forward_propagation(self, data, activation="sigmoid"):
        """

        :param data: inputted data
        :param activation: activation function of neuron
        :return: output of neuron
        """
        sum = np.dot(self.weights, data) + self.bias
        if activation == "relu":
            return SingleNeuron.relu(sum)
        elif activation == "sigmoid":
            return SingleNeuron.sigmoid(sum)

    def update_weights(self, weights):
        """

        :param weights:
        :return:
        """
        self.weights = weights

    def update_bias(self, bias):
        """

        :param bias:
        :return:
        """
        self.bias = bias

    @staticmethod
    def sigmoid(x):
        """

        :param x:
        :return:
        """
        return 1/(1+np.exp(-x))

    @staticmethod
    def relu(x):
        """

        :param x:
        :return:
        """
        return max(0, x)

    @staticmethod
    def mse(x, pred_x):
        """

        :param x:
        :param pred_x:
        :return:
        """
        return np.mean((x - pred_x) ** 2)


class NeuralNetwork(SingleNeuron):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = 0
        self.weights = {}
        self.biases = {}
