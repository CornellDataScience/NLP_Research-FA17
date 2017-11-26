import dl_style_transfer.make_gutenberg_corpus as mgc
import dl_style_transfer.workspace.kate_new as k
import numpy as np


inds = mgc.sentence_mat()
corpus = mgc.get_corpus()
vocab = mgc.get_vocab()
print(inds.shape)
kate = k.Kate(128, len(vocab), inds.shape[1], 32, 6.26)
kate.train()


kate.save_model("./Saved_Kate.mod")