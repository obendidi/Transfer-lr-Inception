import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import *

import time
import sys

SEED = None
EVAL_BATCH_SIZE = 1


def error_measure(predictions, labels):
    """ calculate sum squared error of predictions """
    return np.sum(np.power(predictions - labels, 2)) / (predictions.shape[0])


def eval_in_batches(data, sess, eval_prediction, eval_data_node):
    """
    Evaluate the data on the eval_prediction NN model in batches

    data: (num_images, img_size, img_size, num_channels) image data tensor
    sess: tensorflow session
    eval_prediction: neural network model to predict from
    eval_data_node:  model input placeholder for feed_dict
    returns (num_images, num_classes) shaped tensor of prediction vectors for each data image
    """
    size = data.shape[0]  # num images in data
    if size < EVAL_BATCH_SIZE:
        raise ValueError(
            "batch size for evals larger than dataset size: %d" % size)

    predictions = np.ndarray(shape=(size, num_classes), dtype=np.float32)
    for begin in range(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        # get next batch from begin index to end index
        if end <= size:
            predictions[begin:end, :] = sess.run(
                    eval_prediction,
                    feed_dict={eval_data_node: data[begin:end, ...]})
        else:  # if end index is past the end of the data, fit input to batch size required for feed_dict
            batch_predictions = sess.run(
                    eval_prediction,
                    feed_dict={eval_data_node: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions


data_dir = "."
filename = "labelsxdata.txt"

image_height = 600
image_width = 600
num_classes = 12
batch_size = 1
dataset = Dataset(data_dir, filename, image_height, image_width)

# Generate a validation set.
validation_dataset = dataset.X_test
validation_labels = dataset.y_test
train_dataset = dataset.X_train
train_labels = dataset.y_train

train_size = train_labels.shape[0]
print("train size is %d" % train_size)

train_data_node = tf.placeholder(tf.float32,shape=(None, image_height, image_width, 3))

train_labels_node = tf.placeholder(tf.float32, shape=(None, num_classes))

eval_data_node = tf.placeholder(tf.float32,shape=(None, image_height, image_width, 3))


conv1_weights = tf.Variable(tf.truncated_normal([5, 5, 3, 32],stddev=0.1,seed=SEED))

conv1_biases = tf.Variable(tf.zeros([32]))

conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.1,seed=SEED))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))

conv3_weights = tf.Variable(tf.truncated_normal([5, 5, 64, 64],stddev=0.1,seed=SEED))
conv3_biases = tf.Variable(tf.constant(0.1, shape=[64]))

fc1_weights = tf.Variable(tf.truncated_normal([image_height // 8 * image_width // 8 * 64, 512],stddev=0.1,seed=SEED))
fc1_biases = tf.Variable(tf.random_normal([512]))

fc2_weights = tf.Variable(tf.truncated_normal(
                                    [512, 512],
                                    stddev=0.1,
                                    seed=SEED))
fc2_biases = tf.Variable(tf.random_normal([512]))

fc3_weights = tf.Variable(tf.truncated_normal([512, num_classes],
                            stddev=0.1,
                            seed=SEED))
fc3_biases = tf.Variable(tf.random_normal([num_classes]))


def model(data, train=False):
    """The Model definition."""

    data = data / 255.0

    conv = tf.nn.conv2d(data,conv1_weights,strides=[1, 1, 1, 1],padding='SAME')

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases))

    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

    conv = tf.nn.conv2d(pool,conv2_weights,strides=[1, 1, 1, 1],padding='SAME')

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv2_biases))

    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

    conv = tf.nn.conv2d(pool,conv3_weights,strides=[1, 1, 1, 1],padding='SAME')

    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases))

    pool = tf.nn.max_pool(relu,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')

    pool_shape = pool.get_shape().as_list()

    reshape = tf.reshape(pool,[-1, pool_shape[1] * pool_shape[2] * pool_shape[3]])

    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)

    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

    hidden = tf.nn.relu(tf.matmul(hidden, fc2_weights) + fc2_biases)

    if train:
        hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)

    return tf.matmul(hidden, fc3_weights) + fc3_biases

train_prediction = model(train_data_node, True)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(train_prediction - train_labels_node), 1))

regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                    tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases) +
                    tf.nn.l2_loss(fc3_weights) + tf.nn.l2_loss(fc3_biases))
loss += 1e-7 * regularizers

eval_prediction = model(eval_data_node)

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(
        1e-3,                      # Base learning rate.
        global_step * batch_size,  # Current index into the dataset.
        train_size,                # Decay step.
        0.95,                      # Decay rate.
        staircase=True)

# train_step = tf.train.AdamOptimizer(5e-3).minimize(loss)
# train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(loss)
# train_step = tf.train.MomentumOptimizer(1e-4, 0.95).minimize(loss)
train_step = tf.train.AdamOptimizer(learning_rate, 0.95).minimize(loss, global_step=global_step)

tf.summary.scalar("Loss", loss)
tf.summary.scalar("Learning rate", learning_rate)

log_dir = "logs_testpy"

if not os.path.exists(log_dir):
    os.mkdir(log_dir)
summary_op = tf.summary.merge_all()
init = tf.global_variables_initializer()
sess = tf.InteractiveSession()
sess.run(init)
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

ckpt = tf.train.get_checkpoint_state(log_dir)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
    print("Model in '" + log_dir + "' successfully restored...")

loss_train_record = list() # np.zeros(n_epoch)
loss_valid_record = list() # np.zeros(n_epoch)

epoch = 1
while True :
    shuffled_index = np.arange(train_size)
    np.random.shuffle(shuffled_index)
    train_dataset = train_dataset[shuffled_index]
    train_labels = train_labels[shuffled_index]

    start_time = time.time()
    for step in range(train_size):
        offset = step * batch_size
        batch_data = train_dataset[offset:(offset + batch_size), ...]
        batch_labels = train_labels[offset:(offset + batch_size)]

        feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
        _, loss_train, current_learning_rate = sess.run([train_step, loss, learning_rate], feed_dict=feed_dict)
    eval_result = eval_in_batches(validation_dataset, sess, eval_prediction, eval_data_node)
    loss_valid = error_measure(eval_result, validation_labels)
    time_elapsed = time.time() - start_time
    summary_str = sess.run(summary_op, feed_dict=feed_dict)
    summary_writer.add_summary(summary_str, epoch)
    print ('Epoch %04d, train loss %.8f, validation loss %.8f, train/validation %0.8f, learning rate %0.8f  : (%0.2f sec/epoch)' % (
        epoch,
        loss_train, loss_valid,
        loss_train / loss_valid,
        current_learning_rate,
        round(time_elapsed,2)
    ))
    loss_train_record.append(np.log10(loss_train))
    loss_valid_record.append(np.log10(loss_valid))
    sys.stdout.flush()

    if epoch %50 == 0:
      print("\n TEST :\n")
      ind = np.random.choice(10-1,1,replace=False)
      test_im = validation_dataset[ind]
      test_lab = validation_labels[ind]
      feed_dict = {eval_data_node: test_im}
      pred = sess.run(eval_prediction, feed_dict=feed_dict)
      print("LABELS\t=>\tPredictions")
      for ki in range(num_classes):
        print("%0.3f\t=>\t%0.03f" % (test_lab[0][ki],pred[0][ki]))
      dataset.show(image=test_im[0],label=test_lab[0],pred=pred[0],save=True,show=False,name="preview.png")

    if epoch %500 ==0:
      saver.save(sess, os.path.join(log_dir,"model.ckpt"), epoch)

    epoch+=1
