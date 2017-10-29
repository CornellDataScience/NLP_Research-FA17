import h5py
import numpy as np

# A file to convert the cifar10vgg.h5 preset weights
# to a numpy representation

f = h5py.File("cifar10vgg.h5", "r+")

layer_labels = f.attrs['layer_names']

col = {}

for layer in layer_labels:
    g = f.pop(layer)
    for w in g.attrs['weight_names']:
        col[str(w)] = np.array(g.get(w))

 
np.savez("cifar10vgg_numpy.npz", **col)
