import numpy as np
import dl_style_transfer.workspace.data_helpers
import os

here = os.path.dirname(os.path.abspath(__file__))

yelp = list(open(os.path.join(here, 'yelp_sentences.txt')))
shake = list(open(os.path.join(here, 'shake_sentences.txt')))


col = dict()
word_to_ind = dict()
ind_to_word = dict()

def __line_into_col__(line):
    tokens = dl_style_transfer.workspace.data_helpers.clean_str(line)
    for wor in tokens:
        if col.get(wor) is None:
            col[wor] = 1
        else:
            col[wor] = col[wor] + 1

for l1, l2 in zip(yelp, shake):
    __line_into_col__(l1)
    __line_into_col__(l2)

lis = list(col.items())
lis.sort(key=lambda count: count[1], reverse=True)
for i, word in enumerate(col):
    word_to_ind[word] = i
    ind_to_word[i] = word

voc_len = len(word_to_ind)


def get_bag():
    bag = np.zeros((len(shake) + len(yelp), voc_len))
    for j,sent in enumerate(shake):
        for wor in dl_style_transfer.workspace.data_helpers.clean_str(sent):
            bag[j, word_to_ind[wor]] = bag[j, word_to_ind[wor]] + 1
    for j,sent in enumerate(yelp):
        for wor in dl_style_transfer.workspace.data_helpers.clean_str(sent):
            bag[j, word_to_ind[wor]] = bag[j, word_to_ind[wor]] + 1
    return bag


def string_to_vec(string):
    tokens = dl_style_transfer.workspace.data_helpers.clean_str(string)
    vec = np.zeros(voc_len)
    for wor in tokens:
        vec[word_to_ind[wor]] = vec[word_to_ind[wor]] + 1
    return vec


def get_ryans_strange_input():
    vec = []
    for l1, l2 in zip(yelp, shake):
        vec.append(dl_style_transfer.workspace.data_helpers.clean_str(l1))
        vec.append(dl_style_transfer.workspace.data_helpers.clean_str(l2))
    return np.array([i for l in vec for i in l])


def vocab_length():
    return voc_len
