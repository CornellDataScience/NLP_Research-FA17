import dl_style_transfer.make_shakespeare_corpus as msc
import dl_style_transfer.workspace.kate_new as k
import numpy as np


inds = msc.sentence_mat()
corpus = msc.get_corpus()
vocab = msc.get_vocab()
print(corpus)
kate = k.Kate(128, len(vocab), 32, 6.26)
kate.train()


kate.save_model("./Saved_Kate.mod")