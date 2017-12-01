import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../'))

import dl_style_transfer.from_shake_yelp as yelp
from dl_style_transfer.workspace.kate_new import Kate


x_train = yelp.get_ryans_strange_input()
kate = Kate(128, yelp.vocab_length(), 32, 6.26)
kate.train(x_train, 100, 32)

kate.save_model("saved/Saved_Kate.ckpt")

