import tensorflow as tf
import numpy as np
import os
import sys
import data_helpers
from tensorflow.contrib import learn

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "../yelp_sentences.txt", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "../runner_up.txt", "Data source for the negative data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "runs/1512363664/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
vocab_dict = vocab_processor.vocabulary_._mapping
x_test = np.array(list(vocab_processor.transform(x_text)))

test_idx = int(sys.argv[1])

sequence_length = 144
embedding_size = 128

checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()

with graph.as_default():
    session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_conf)

    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        activations = []
        targets = []

        for i, n in enumerate([3, 4, 5]):
            activations.append(graph.get_operation_by_name("conv-maxpool-" + str(n) + "/conv").outputs[0])
            targets.append(tf.constant(sess.run(activations[i], {input_x: x_test[test_idx].reshape(1, -1)})))

        # print("\n", x_text[test_idx], "\n")

        with tf.variable_scope('reconstructions'):

            reconstruction = tf.get_variable('reconstuction',
                                             initializer=tf.random_uniform([1, sequence_length, embedding_size]))
            embedded_chars_expanded = tf.expand_dims(reconstruction, -1)

            convolutions = []
            losses = []

            for i, filter_size in enumerate([3, 4, 5]):
                W_trained = graph.get_operation_by_name("conv-maxpool-" + str(filter_size) + "/W").outputs[0]
                W = tf.constant(sess.run(W_trained))

                conv = tf.nn.conv2d(embedded_chars_expanded, W, strides=[1, 1, 1, 1],
                                    padding='VALID', name='conv' + str(i))
                convolutions.append(conv)
                losses.append(tf.nn.l2_loss(conv - targets[i]))

            loss = tf.reduce_sum(losses)
            train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

            sess.run(tf.global_variables_initializer())

            n_iters = 10000
            for it in range(n_iters):
                sess.run(train_step)

        reconstructed = sess.run(reconstruction)
        words = []
        for idx in np.argmax(reconstructed, axis=-1).flatten():
            words.append(list(vocab_dict.keys())[list(vocab_dict.values()).index(idx)])
        sentence_length = np.where(x_test[test_idx] == 0)[0][0]
        print(' '.join(word for word in words[:sentence_length]))
