import ast
import numpy as np

lib = {}

with open("vocab.txt") as file:
    for (i, l) in enumerate(file):
        lib[l] = i

def load_vecs():
    acc = []
    with open("vec.txt") as file:
        for l in file:
            line = l.split("=>")
            acc.append({'text': line[0], 'vec': ast.literal_eval(line[1])})
    return acc

def word_to_one_hot(s):
    ret = np.zeros(len(lib))
    ret[lib[s]] = 1
    return ret
