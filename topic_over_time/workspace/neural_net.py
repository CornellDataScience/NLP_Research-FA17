from gensim_lda_model import Gensimembedder
from gensim import corpora, models
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import random

def gen_data(ids):
    '''
    generate embedding from the set of ids
    '''
    data = reviews[reviews['review_id'].isin(ids)]
    out = []
    for d in data['text']:
        out.append(model.embed_sent(d))
    return np.array(out)

def one_hot(stars):
    res = []
    for s in stars:
        out = np.array([0.0]*5)
        out[s-1] = 1.0
        res.append(out)
    return np.array(res)

def one_hot_three(stars):
    res = []
    for s in stars:
        out = np.array([0.0]*3)
        if s == 5:
            out[2] = 1.0
        elif s==1:
            out[0] = 1.0
        else:
            out[1] = 1.0
        res.append(out)
    return np.array(res)


def layer(input_data, size_in, size_out, name):
    with tf.name_scope(name):
        # weight as random normal variables
        w = tf.Variable(tf.random_normal([size_in, size_out]), name = 'W')
        # bias as random normal variables
        b = tf.Variable(tf.random_normal([size_out]), name = 'B')

        activation = tf.atan(tf.matmul(input_data, w) + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activation", activation)

        return activation


def output_layer(input_data, size_in, size_out, name):
    with tf.name_scope(name):
        # weight as random normal variables
        w = tf.Variable(tf.random_normal([size_in, size_out]), name = 'W')
        # bias as random normal variables
        b = tf.Variable(tf.random_normal([size_out]), name = 'B')

        activation = tf.atan(tf.matmul(input_data, w) + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activation", activation)

        return activation


def build_model(x, input_size, hidden, out_size):
    prev = input_size
    activation = x
    # build a series of hidden layers
    for name,i in enumerate(hidden):
        activation = layer(activation, prev, i, 'hiddenlayer-'+str(name))
        prev = i

    # build an output layer
    out = output_layer(activation, hidden[-1], out_size, 'output')

    return out


if __name__ == '__main__':
    business = pd.read_csv('chinese_business_clean.csv')
    reviews = pd.read_csv('chinese_reviews_clean.csv')

    lda =  models.LdaModel.load('gensim/lda.model')
    dictionary = corpora.Dictionary.load('gensim/chinsese_dict.dict')
    model = Gensimembedder(model = lda, dictionary = dictionary)

    case_review = reviews[reviews['business_id'] == 'yfxDa8RFOvJPQh0rNtakHA']
    id_train, id_test, star_train, star_test = train_test_split(case_review['review_id'], case_review['stars'], test_size=0.33)

    embed_train = gen_data(id_train)
    embed_test = gen_data(id_test)

    one_hot_star = one_hot_three(star_train)
    one_hot_star_test = one_hot_three(star_test)

    x = tf.placeholder(tf.float32, shape = [None, 128], name = 'input_topic') # number of topics
    y = tf.placeholder(tf.float32, shape = [None, 3], name = 'softmax') # 5 stars
    learning_rate = 0.05
    out = build_model(x, 128, [100, 100], 3) # shape of (?, 5)

    # loss
    with tf.name_scope("loss"):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= out, labels = y))
        tf.summary.scalar("loss", cross_entropy)
    # optimization
    with tf.name_scope("train"):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
    # reports
    with tf.name_scope("accuracy"):
        correct = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y, 1)), dtype = tf.float32))
        tf.summary.scalar("accuracy", correct)



    training_epoch = 50000


    with tf.Session() as sess:
        summ = tf.summary.merge_all()
        writer = tf.summary.FileWriter("tmp/log/1")
        writer.add_graph(sess.graph)
        sess.run(tf.global_variables_initializer())

        for epoch in range(training_epoch):

            idx = random.sample(set(np.arange(1638)), 10)
            x_in = embed_train[idx]
            y_out = one_hot_star[idx]

            r = random.randint(0,1638-11)
            if epoch % 5 == 0:
                s = sess.run(summ, feed_dict = {x:embed_train[r:r+10], y:one_hot_star[r:r+10]})
                writer.add_summary(s,epoch)
            if epoch % 5000 == 0:
                [accuracy] = sess.run([correct], feed_dict = {x:embed_train[r:r+10], y:one_hot_star[r:r+10]})
                print ('%.2f' % accuracy)
            sess.run(opt, feed_dict = {x:x_in, y:y_out})

        pred = tf.nn.softmax(out)  # Apply softmax to logits
        # Calculate accuracy

        print("Accuracy:", sess.run(correct, feed_dict = {x: embed_test, y: one_hot_star_test}))
