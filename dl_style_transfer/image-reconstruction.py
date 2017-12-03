import os
import sys
import numpy as np
import scipy.io
import scipy.misc
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3

WEIGHTS = 'cifar10vgg_numpy.npz'
weights = np.load(WEIGHTS)

def load_weights(layer_name):
    layer_type = layer_name.split('_')[0]
    layer_num = int(layer_name.split('_')[1])
    W = weights["b'" + str(layer_name) + "/kernel:0'"]
    b = weights["b'" + str(layer_name) + "/bias:0'"]
    if layer_type == 'conv2d':
        beta = weights["b'batch_normalization_" + str(layer_num) + "/beta:0'"]
        gamma = weights["b'batch_normalization_" + str(layer_num) + "/gamma:0'"]
        mov_mean = weights["b'batch_normalization_" + str(layer_num) + "/moving_mean:0'"]
        mov_var = weights["b'batch_normalization_" + str(layer_num) + "/moving_variance:0'"]
        return W, b, beta, gamma, mov_mean, mov_var
    elif layer_type == 'dense':
        if layer_num == 1:
            beta = weights["b'batch_normalization_14/beta:0'"]
            gamma = weights["b'batch_normalization_14/gamma:0'"]
            mov_mean = weights["b'batch_normalization_14/moving_mean:0'"]
            mov_var = weights["b'batch_normalization_14/moving_variance:0'"]
            return W, b, beta, gamma, mov_mean, mov_var
        else:
            return W, b
        
def conv(inputs, n_filters, name):
    W, b, beta, gamma, mov_mean, mov_var = load_weights(name)
    activations = tf.layers.conv2d(inputs, n_filters, [3,3], 
                                   padding='SAME',
                                   activation=tf.nn.relu,
                                   kernel_initializer=tf.constant_initializer(W),
                                   bias_initializer=tf.constant_initializer(b),
                                   trainable=False,
                                   name=name)
    """normed = tf.layers.batch_normalization(activations,
                                          beta_initializer=tf.constant_initializer(beta),
                                          gamma_initializer=tf.constant_initializer(gamma),
                                          moving_mean_initializer=tf.constant_initializer(mov_mean),
                                          moving_variance_initializer=tf.constant_initializer(mov_var),
                                          trainable=False,
                                          name=name)"""
    normed = bn(activations, beta, gamma, mov_mean, mov_var, scope=name)
    return normed

def pool(inputs, name):
    return tf.layers.average_pooling2d(inputs, [2,2], [2,2], padding='SAME', name='name')

def bn(inputs, beta, gamma, avg_mean, avg_var, scope='BN'):
    with tf.variable_scope(scope):
        return tf.nn.batch_normalization(inputs, avg_mean, avg_var, beta, gamma, 1e-7, name='Normalized')


def fc(inputs, name):
    W, b, beta, gamma, mov_mean, mov_var = load_weights(name)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    activations = tf.nn.relu(tf.matmul(inputs, W) + b)
    """normed = tf.layers.batch_normalization(activations,
                                          beta_initializer=tf.constant_initializer(beta),
                                          gamma_initializer=tf.constant_initializer(gamma),
                                          moving_mean_initializer=tf.constant_initializer(mov_mean),
                                          moving_variance_initializer=tf.constant_initializer(mov_var),
                                          trainable=False,
                                          name=name)"""
    normed = bn(activations, beta, gamma, mov_mean, mov_var, scope=name)
    return normed

def softmax(inputs, name):
    W, b = load_weights(name)
    W = tf.constant(W)
    b = tf.constant(np.reshape(b, (b.size)))
    return tf.nn.softmax(tf.matmul(inputs, W) + b)

n_images = 1
img = tf.get_variable('img', initializer=np.zeros([n_images, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS], 
                                                  dtype=np.float32))

conv1_1 = conv(img, 64, 'conv2d_1')
conv1_2 = conv(conv1_1, 64, 'conv2d_2')
pool1 = pool(conv1_2, 'pool1')

conv2_1 = conv(pool1, 128, 'conv2d_3')
conv2_2 = conv(conv2_1, 128, 'conv2d_4')
pool2 = pool(conv2_2, 'pool2')

conv3_1 = conv(pool2, 256, 'conv2d_5')
conv3_2 = conv(conv3_1, 256, 'conv2d_6')
conv3_3 = conv(conv3_2, 256, 'conv2d_7')
pool3 = pool(conv3_3, 'pool3')

conv4_1 = conv(pool3, 512, 'conv2d_8')
conv4_2 = conv(conv4_1, 512, 'conv2d_9')
conv4_3 = conv(conv4_2, 512, 'conv2d_10')
pool4 = pool(conv4_3, 'pool4')

conv5_1 = conv(pool4, 512, 'conv2d_11')
conv5_2 = conv(conv5_1, 512, 'conv2d_12')
conv5_3 = conv(conv5_2, 512, 'conv2d_13')
pool5 = pool(conv5_3, 'pool5')

flattened = tf.reshape(pool5, [-1, 512])
fc6 = fc(flattened, 'dense_1')
pred = softmax(fc6, 'dense_2')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        if sys.version[0] == '3':
            dict = pickle.load(fo, encoding='bytes')
        elif sys.version[0] == '2':
            dict = pickle.load(fo)
    return dict

MEAN = 120.707
STD = 64.15

def normalize(X_test):
    return (X_test - MEAN)/(STD + 1e-7)

def denormalize(X_norm):
    return (X_norm)*(STD + 1e-7) + MEAN

def generate_noise_image():
    return np.random.uniform(-20, 20, (1, IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS)).astype('float32')

CONTENT_IMAGE = 'images/' + sys.argv[1]
OUTPUT_DIR = 'reconstructions/'
CONTENT_LAYER = conv4_2

def load_image(path):
    image = scipy.misc.imread(path)
    image = np.reshape(image, ((1,) + image.shape)) 
    image = normalize(image)                     
    return image

def save_image(path, image):
    image = image[0]
    image = denormalize(image)                   
    scipy.misc.imsave(path, image)
    
content_image = load_image(CONTENT_IMAGE)
input_image = generate_noise_image()

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

sess.run(img.assign(content_image))
embedding = sess.run(CONTENT_LAYER)

sess.run(img.assign(input_image))
content_loss = tf.nn.l2_loss(CONTENT_LAYER - tf.constant(embedding))

optimizer = tf.train.AdamOptimizer(0.1)
train_step = optimizer.minimize(content_loss)

sess.run(tf.global_variables_initializer())

ITERATIONS = 10000

for it in range(ITERATIONS):
    sess.run(train_step)
    if it%100 == 0:
        reconstruction = sess.run(img)
        print('Iteration %d' % (it))
        print('loss: ', sess.run(content_loss))

        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)

        filename = 'reconstructions/' +  str(sys.argv[1])[:5] + '%d.png' % (it)
        save_image(filename, reconstruction)
