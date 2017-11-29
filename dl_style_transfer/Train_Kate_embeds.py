import dl_style_transfer.from_shake_yelp
import dl_style_transfer.workspace.kate_new as k
import numpy as np


x_train = dl_style_transfer.from_shake_yelp.get_ryans_strange_input()
kate = k.Kate(128, dl_style_transfer.from_shake_yelp.vocab_length(), 32, 6.26)
kate.train()


kate.save_model("./Saved_Kate.mod")