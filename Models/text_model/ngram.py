"""
Author: Samson Qian
"""
import pandas as pd
import numpy as np


class UnigramLM(object):
    """
    Uni-gram Language Model.
    """
    def __init__(self):
        """
        Initializes a Uni-gram language model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = None

    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occurring
        in the language.
        :param tokens: tuple of tokens
        """
        counts = {x: tokens.count(x) for x in tokens}
        total = len(tokens)
        unigram = pd.Series([counts[x] / total for x in set(tokens)], index=set(tokens))
        self.mdl = unigram

    def probability(self, words):
        """
        Gives the probability a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
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
        return " ".join(np.random.choice(self.mdl.index.values, M, p=self.mdl.values))


class NGramLM(object):
    """
    N-gram Language Model.
    """
    def __init__(self, N, tokens):
        """
        Initializes a N-gram language model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        :param N: number of grams
        :param tokens: tuple of tokens
        """
        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception("N must be greater than 1")
        elif N == 2:
            unigram = UnigramLM()
            unigram.train(tokens)
            self.prev_mdl = unigram
        else:
            mdl = NGramLM(N - 1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        Takes in a list of tokens and returns a list of N-grams.
        The START/STOP tokens in the N-grams should be handled as
        explained in the notebook.
        :param tokens: tuple of tokens
        :returns: n-grams created from tokens
        """
        ngrams = []
        for i in range(len(tokens) - self.N + 1):
            ngrams.append(tuple(tokens[i:i + self.N]))
        return ngrams

    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe indexed on distinct tokens, with three
        columns (ngram, n1gram, prob).
        :param ngrams: list of n-grams
        """
        probabilities = pd.DataFrame({"ngram": ngrams, "n1gram": ngrams})
        probabilities["n1gram"] = probabilities["n1gram"].apply(lambda x: x[0:self.N - 1])
        # ngram counts C(w_1, ..., w_n)
        counts_n = probabilities["ngram"].map(probabilities["ngram"].value_counts())
        # n-1 gram counts C(w_1, ..., w_(n-1))
        counts_n1 = probabilities["n1gram"].map(probabilities["n1gram"].value_counts())
        # Create the conditional probabilities
        probabilities["prob"] = counts_n / counts_n1
        probabilities = probabilities.drop_duplicates(subset="ngram").reset_index(drop=True)

        return probabilities

    def probability(self, words):
        """
        probability gives the probability a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.
        """
        words_ngrams = self.create_ngrams(words)
        if not set(words_ngrams).issubset(self.ngrams):
            return 0
        prob = 1
        for i in range(0, len(words)):
            if i < self.N - 1:
                temp = self
                while temp.N != i + 2:
                    temp = self.prev_mdl
                prob *= temp.prev_mdl.probability(tuple([words[j] for j in range(0, i + 1)]))
                continue
            else:
                seq = tuple([words[k] for k in range(i - self.N + 1, i + 1)])
                prob *= self.mdl[self.mdl["ngram"] == seq]["prob"].values[0]
        return prob

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.
        :param M: number of words to sample
        :returns: sequence of M words sampled by model
        """
        def generate(length):
            tokens = ["\x02"]
            for i in range(length):
                if len(tokens) < self.N - 1:
                    tokens.append(self.prev_mdl.sample(1))
                    continue
                elif len(self.mdl[self.mdl["n1gram"] == tuple([tokens[j] for j in range(-1, -self.N, -1)])]) == 0:
                    tokens.append("\x03")
                    continue
                else:
                    probabilities = self.mdl[self.mdl["n1gram"] == tuple([tokens[j] for j in range(-1, -self.N, -1)])]
                    sampled = np.random.choice(probabilities["ngram"], 1, p=probabilities["prob"])[0]
                    tokens.append(sampled[-1])
            return tokens

        # Transform the tokens to strings
        sample_str = " ".join(generate(M))

        return sample_str
