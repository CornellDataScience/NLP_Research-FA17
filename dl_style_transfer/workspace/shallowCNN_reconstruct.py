
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from shallowCNN_transfer import TextCNN
from tensorflow.contrib import learn
from sklearn.preprocessing import OneHotEncoder

# Parameters
# ==================================================

# Model Initialization
tf.flags.DEFINE_string("model_path", "", "Path to the trained model.")


# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_integer("num_reconstructions", 1, "Number of reconstructions to built (default: 1)")


# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
# TODO: This is very crude, should use cross-validation
dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default() as sess:
        if FLAGS.model_path in (None, ""):
            raise ValueError("`model_path` must be specified, but was:", FLAGS.model_path)

        print("Loading saved model")
        restore_saver = tf.train.import_meta_graph(FLAGS.model_path + ".meta")  # Construct the graph
        restore_saver.restore(sess, FLAGS.model_path)  # Initialize the variables
        print("Constructing Reconstruction Graph...")
        cnn = TextCNN(
            num_reconstructions=FLAGS.num_reconstructions,
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(vocab_processor.vocabulary_),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        sess.run(cnn.init_op)  # Initialize the cnn

        # Expand data into the embeddings
        data_batch = x_train[:1]
        embeddings_batch = np.zeros(data_batch.shape + (FLAGS.embedding_dim,))
        for i, sentence in enumerate(x_train):
            embeddings_batch[i] = np.array(OneHotEncoder(n_values=len(vocab_processor.vocabulary_).fit_transform(np.array(sentence).reshape(1, -1))).todense()).reshape(len(vocab_processor.vocabulary_), -1)

        # Assign reconstruction tensor to data_batch to generatae target_content
        target_content = sess.run(cnn.activations, feed_dict={cnn.reconstructions: embeddings_batch})
        target_content = [tf.constant(target) for target in target_content]
        losses = [tf.nn.l2_loss(target - activation) for target, activation in zip(target_content, cnn.activations)]
        loss = tf.reduce_sum(losses)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        n_iters = 1000
        for i in range(n_iters):
            _, loss = sess.run([train_op, loss])
            if i%50 == 0:
                print("Loss:", loss)