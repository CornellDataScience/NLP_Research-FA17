import numpy as np


def vocablist(arr, filename):
    with open(filename) as f:
        for i, l in enumerate(f):
            arr.append((i, l))
    return i + 1


# Returns a pandas dataframe of words and vectors
def vocabvecs(filename):
    arr = []
    vocablist(arr, filename)
    return {word: x for (x, word) in arr}


def wordtovec(s, vocab):
    ret = np.zeros(len(vocab))
    ret[vocab[s]] = 1
    return ret


def doctovec(filename, string):
    vocab = vocabvecs(filename)
    return map(string.split(" +"), lambda s: wordtovec(s, vocab))
