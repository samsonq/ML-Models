"""
Author: Samson Qian
"""
import numpy as np


class Convolution3x3:
    """

    """
    def __init__(self, num_filters):
        """

        :param num_filters:
        """
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3)

    @staticmethod
    def conv_spots(image):
        """

        :param image:
        :return:
        """
        height, width = image.shape
        for i in range(height-2):
            for j in range(width-2):
                spot = image[i:(i+3), j:(j+3)]
                yield spot, i, j

    def convolve(self, image):
        """

        :param image:
        :return:
        """
        height, width = image.shape
        convolved = np.zeros((height-2, width-2, self.num_filters))

        for spot, i, j, in self.conv_spots(image):
            convolved[i, j] = np.sum(spot*self.filters, axis=(1, 2))

        return convolved


class MaxPooling2:
    """

    """
    def __init__(self):
        """

        """

    @staticmethod
    def pool_spots(image):
        """

        :param image:
        :return:
        """
        height, width, depth = image.shape
        height = height // 2
        width = width // 2
        for i in range(height):
            for j in range(width):
                spot = image[(i*2):(i*2+2), (j*2):(j*2+2)]
                yield spot, i, j

    def pool(self, image):
        """

        :param image:
        :return:
        """
        height, width, depth = image.shape
        pooled = np.zeros((height//2, width//2, depth))

        for spot, i, j, in self.pool_spots(image):
            pooled[i, j] = np.amax(spot, axis=(0, 1))

        return pooled


class Softmax:
    """

    """
    def __init__(self, size, num_outputs):
        self.weights = np.random.randn(size, num_outputs)
        self.biases = np.zeros(num_outputs)

    def softmax(self, x):
        x = x.flatten()

        numerator = np.exp(np.dot(x, self.weights) + self.biases)
        return numerator/np.sum(numerator, axis=0)
