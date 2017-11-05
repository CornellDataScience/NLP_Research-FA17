
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
import numpy as np
import tensorflow as tf

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


batch_size = 256
epochs = 20

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
    not_equal = .1 / ((equal + 0.0001)**2)

    l2_regularizer = K.mean(K.square(y_pred), axis=1)
    l2_regularizer = K.expand_dims(equal, 1)

    return tf.where(b, equal, not_equal) + (0.01 * (1 / l2_regularizer))


# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256))
model.add(keras.layers.LeakyReLU())
keras.layers.GaussianNoise(0.4)
model.add(Dense(128))
model.add(keras.layers.LeakyReLU())
keras.layers.GaussianNoise(0.1)
model.add(Lambda(lambda  x: K.l2_normalize(x,axis=1)))

model.compile(loss=loss,
              optimizer=keras.optimizers.Adadelta())

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    channel_shift_range=1.0)

datagen.fit(x_train)

model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size), steps_per_epoch=len(x_train) / 256, epochs=epochs, verbose=1)


x_preds = model.predict(x_test)
i = np.random.randint(10000,size=2000)
vis_data = TSNE(n_components=2).fit_transform(x_preds[i, :])

vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_test[i], cmap=plt.cm.get_cmap("jet", 10), s=20)
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
