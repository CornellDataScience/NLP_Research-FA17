import numpy as np
import os
import sys

sys.path.append(os.path.abspath('../'))

import dl_style_transfer.from_shake_yelp as yelp

from dl_style_transfer.workspace.kate_new import Kate
from sklearn.model_selection import train_test_split
from time import time

seed = 1337
np.random.seed(seed)

data = yelp.get_small_bag()
train, test = train_test_split(data)
kate = Kate(yelp.vocab_length(), 128, False, 32, 6.26)
# kate.train(train, 100, 128)


def train_batch(data, batch_size, epoch=100, start_stop_info=True, progress_interval=5):
    n_batches = np.ceil(len(data) / batch_size)
    for _ in range(epoch):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            npbatch = np.zeros((len(batch), yelp.vocab_length()))
            for i in range(len(batch)):
                for j in range(len(batch[i])):
                    npbatch[i, j] += npbatch[i, j] + 1
            npbatch = np.log(1 + npbatch) / np.expand_dims(np.max(np.log(1 + npbatch), axis=1), -1)
            kate.train(npbatch, 1, 128)


train_batch(data, 128, 100)
kate.save_model("saved/100-epoch/kate-bag-words-100.ckpt")


def random_sample(data, num_samples):
    """Samples along first dimension of `data`"""
    idxs = np.random.choice(data.shape[0], size=num_samples, replace=False)
    return data[idxs]


def compute_accuracy(data, batch_size, start_stop_info=True, progress_interval=5):
    if start_stop_info:
        print("Computing accuracy for dataset of size", data.shape[0])

    n_batches = np.ceil(data.shape[0] / batch_size)

    accuracy = 0
    last_time = time()
    count = 0
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        npbatch = np.zeros((len(batch), yelp.vocab_length()), dtype=np.float32)
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                npbatch[i, j] += npbatch[i, j] + 1
        npbatch = np.log(1 + npbatch) / np.expand_dims(np.max(np.log(1 + npbatch), axis=1), -1)
        reconstructed = kate.reconstruct(npbatch)
        accuracy += np.sum(reconstructed == npbatch)
        current_time = time()
        if progress_interval is not None and (current_time - last_time) >= progress_interval:
            last_time = current_time
            print("Computing accuracy. Percent complete:", 100 * count / n_batches)
        count += 1

    if start_stop_info:
        print("Finished computing accuracy!")

    accuracy /= data.shape[0]
    return accuracy


batch_size = 128
num_samples = 50000
train_accuracy = compute_accuracy(random_sample(train, num_samples), batch_size)
test_accuracy = compute_accuracy(random_sample(test, num_samples), batch_size)
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
