import numpy as np
import dl_style_transfer.workspace.data_helpers
import os


here = os.path.dirname(os.path.abspath(__file__))

sents = list(open(os.path.join(here, 'yelp_sentences.txt'))) + list(open(os.path.join(here, 'shake_sentences.txt')))

thresh = 5

col = dict()
word_to_ind = dict()
ind_to_word = dict()


def __line_into_col__(line):
    tokens = dl_style_transfer.workspace.data_helpers.clean_str(line).split(" ")
    for wor in tokens:
        if col.get(wor) is None:
            col[wor] = 1
        else:
            col[wor] = col[wor] + 1


for l in sents:
    __line_into_col__(l)

lis = list(col.items())
lis.sort(key=lambda count: count[1], reverse=True)
for i, word in enumerate(lis):
    word_to_ind[word[0]] = i
    ind_to_word[i] = word[0]

voc_len = len(word_to_ind)

shape = (len(sents), voc_len)

def get_small_bag():
	bag = []
	for sent in sents:
		sbag =[]
		for wor in dl_style_transfer.workspace.data_helpers.clean_str(sent).split(" "):
			sbag.append(word_to_ind[wor])
		bag.append(sbag)
	return bag
			
			

def get_bag():
    bag = np.zeros(shape)
    for j,sent in enumerate(sents):
        for wor in dl_style_transfer.workspace.data_helpers.clean_str(sent).split(" "):
            bag[j, word_to_ind[wor]] = bag[j, word_to_ind[wor]] + 1
    return np.log(1 + bag) / np.max(np.log(1 + bag), axis=1)


def string_to_vec(string):
    tokens = dl_style_transfer.workspace.data_helpers.clean_str(string).split(" ")
    vec = np.zeros(voc_len)
    for wor in tokens:
        vec[word_to_ind[wor]] = vec[word_to_ind[wor]] + 1
    return vec


def get_ryans_strange_input():
    vec = []
    for l in sents:
        vec.append(dl_style_transfer.workspace.data_helpers.clean_str(l))
    return np.array([word_to_ind[i] for l in vec for i in l.split(" ")])


def vocab_length():
    return voc_len
