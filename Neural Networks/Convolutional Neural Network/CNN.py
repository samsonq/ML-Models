"""
Author: Samson Qian
"""
import numpy as np


class Convolution:
    """
    A convolution layer used to convolve an image with pixels. Represented by a 3-d array with
    a specified number of filters to use and the size of the filters.
    """
    def __init__(self, num_filters, size):
        """
        Defines the number and size of filters for the convolution.
        :param num_filters: number of filters
        :param size: size of filter
        """
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, size[0], size[1])

    @staticmethod
    def conv_spots(image):
        """
        Helper method to find spots on image to perform convolution.
        :param image: image pixels
        :return: spots of image for convolution
        """
        height, width = image.shape

        for i in range(height-2):
            for j in range(width-2):
                spot = image[i:(i+3), j:(j+3)]
                yield spot, i, j

    def convolve(self, image):
        """
        Applies the filter and performs a convolution on the image.
        :param image: image to convolve
        :return: output after convolution
        """
        height, width = image.shape
        convolved = np.zeros((height-2, width-2, self.num_filters))

        for spot, i, j, in self.conv_spots(image):
            convolved[i, j] = np.sum(spot*self.filters, axis=(1, 2))

        return convolved


class MaxPooling:
    """
    Pooling layer to reduce the size of an image after convolution. Uses max pooling with a pooling
    size of 2 on the image.
    """
    def __init__(self, pool_size=2):
        """
        Define the pooling size of the layer.
        :param pool_size: size of pooling
        """
        self.pool_size = pool_size

    def pool_spots(self, image):
        """
        Finds spots to perform max pooling on image.
        :param image: image to pool after convolution
        :return: spots to perform max pooling on image
        """
        height, width, depth = image.shape
        height = height // self.pool_size
        width = width // self.pool_size
        for i in range(height):
            for j in range(width):
                spot = image[(i*self.pool_size):(i*self.pool_size+2),
                             (j*self.pool_size):(j*self.pool_size+2)]
                yield spot, i, j

    def pool(self, image):
        """

        :param image: image to pool
        :return: image after max pooling
        """
        height, width, depth = image.shape
        pooled = np.zeros((height//2, width//2, depth))

        for spot, i, j, in self.pool_spots(image):
            pooled[i, j] = np.amax(spot, axis=(0, 1))

        return pooled


class Softmax:
    """
    Softmax function used for activation function to determine probabilities of output
    class predictions. Will be implemented for classification in image detection.
    """
    def __init__(self, size, num_outputs):
        """
        Initialize softmax features.
        :param size: size of softmax
        :param num_outputs: number of classes
        """
        self.weights = np.random.randn(size, num_outputs)
        self.biases = np.zeros(num_outputs)

    def softmax(self, x):
        """
        Computes the softmax equation and outputs a probability of a class.
        :param x: predictions
        :return: softmax value
        """
        x = x.flatten()  # flatten the shape of the predictions

        numerator = np.exp(np.dot(x, self.weights) + self.biases)
        return numerator/np.sum(numerator, axis=0)


class CNN:
    """
    Implements a convolutional neural network with convolution and pooling layers
    to make image classifications.
    """
    def __init__(self, conv):
        """
        Define NN layers
        :param conv: convolution layer
        """
        self.conv = conv

    def feedforward(self, image, label, pool="max"):
        """
        Passes the data forward once through the network and computes the output.
        :param image: image data
        :param label: true label of image
        :param pool: pooling schema, default max
        :returns: softmax probability, loss, accuracy
        """
        out = self.conv.convolve((image / 255) - 0.5)
        if pool == "max":
            out = MaxPooling().pool(out)
        out = Softmax(image.size, 10).softmax(out)

        loss = -np.log(out[label])  # cross-entropy loss
        acc = 1 if np.argmax(out) == label else 0

        return out, loss, acc

    def train(self, X, y):
        """
        Trains the neural network with the inputted training data.
        :param X: training data features
        :param y: training data labels
        :returns: loss, number of correct predictions
        """
        loss = 0
        num_correct = 0

        for i, (im, label) in enumerate(zip(X, y)):
            _, l, acc = self.feedforward(im, label)
            loss += l
            num_correct += acc

            if i % 100 == 99:
                print(
                    '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
                    (i + 1, loss / 100, num_correct)
                )
                loss = 0
                num_correct = 0

        return loss, num_correct
