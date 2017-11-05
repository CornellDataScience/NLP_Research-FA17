
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)




x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = y_train.astype('float32')


print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

def loss(y_true, y_pred):
    y_true_rev = K.reverse(y_true,0)
    y_pred_rev = K.reverse(y_pred,0)
    b = K.equal(y_true, y_true_rev)


    equal = K.mean(K.square(y_pred - y_pred_rev), axis=1)
    equal = K.expand_dims(equal, 1)
    not_equal = 1 / (equal + 0.0001)

    return tf.where(b, equal, not_equal)


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1)))

model.compile(loss=loss,
              optimizer=keras.optimizers.Adam())

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs, verbose=1)


x_preds = model.predict(x_test)
i = np.random.randint(10000, size=500)
vis_data = TSNE(n_components=2).fit_transform(x_preds[i, :])

vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_test[i], cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
