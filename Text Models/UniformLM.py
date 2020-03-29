import numpy as np
import pandas as pd


class UniformLM(object):
    """
    Uniform Language Model.
    """
    def __init__(self):
        """
        Initializes a Uniform language model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = None

    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occurring
        in the language.
        :param tokens: tuple of tokens
        :returns: probability model of uniform occurrences of words
        """
        total = len(set(tokens))
        uniform = pd.Series([1 / total for _ in range(total)], index=set(tokens))
        self.mdl = uniform

    def probability(self, words):
        """
        Gives the probability a sequence of words
        appears under the language model.
        :param words: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.
        """
        if not set(words).issubset(self.mdl.index.values):
            return 0
        probabilities = self.mdl.loc[list(words)]
        return np.prod(probabilities)

    def sample(self, M):
        """
        Selects tokens from the language model of length M, returning
        a string of tokens.
        :param M: number of words to sample
        :returns: sequence of M words sampled by model
        """
        return " ".join(self.mdl.sample(M, replace=True).index.values)
