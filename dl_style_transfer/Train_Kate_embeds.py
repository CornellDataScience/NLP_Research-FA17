import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../'))

import dl_style_transfer.from_shake_yelp as yelp

from dl_style_transfer.workspace.kate_new import Kate
from sklearn.model_selection import train_test_split

seed = 1337
np.random.seed(seed)

dataset = yelp.get_ryans_strange_input()
train, test = train_test_split(dataset)
kate = Kate(128, yelp.vocab_length(), 32, 6.26)
kate.train(train, 100, 128)
kate.save_model("saved/kate_hot.ckpt")


def random_sample(data, num_samples):
    """Samples along first dimension of `data`"""
    idxs = np.random.choice(data.shape[0], size=num_samples, replace=False)
    return data[idxs]

# TODO: Write code to batch when reconstructing, right now its not batching

train_result = kate.reconstruct(train)
test_result = kate.reconstruct(test)
print("Training Accuracy:", np.mean(train_result == train))
print("Testing Accuracy:", np.mean(test_result == test))
