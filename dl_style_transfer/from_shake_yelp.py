import numpy as np
import dl_style_transfer.workspace.data_helpers

shake = list(open('shake_sentences.txt'))
yelp = list(open('yelp_sentences.txt'))

col = dict()
word_to_ind = dict()
ind_to_word = dict()

def __line_into_col__(line):
    tokens = dl_style_transfer.workspace.data_helpers.clean_str(line)
    for wor in tokens:
        if col.get(wor) == None:
            col[wor] = 1
        else:
            col[wor] = col[wor] + 1

col = col.items()
col.sort(key=lambda count: count[1], reverse=True)
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
        for wor in sent:
            bag[j, word_to_ind[wor]] = bag[j, word_to_ind[wor]] + 1
    return bag


def string_to_vec(string):
    tokens = dl_style_transfer.workspace.data_helpers.clean_str(string)
    vec = np.zeros(voc_len)
    for wor in tokens:
        vec[word_to_ind[wor]] = vec[word_to_ind[wor]] + 1

