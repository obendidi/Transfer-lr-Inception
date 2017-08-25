import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.platform import tf_logging as logging
from InceptionResnetV2 import *
from utils import get_split,load_batch
import os
import time
import CheckpointLoader

#================ DATASET INFORMATION ======================
dataset_dir = '../dataset'

log_dir = './logs'

checkpoint_file = '../inception_resnet_v2_2016_08_30.ckpt'


file_pattern = 'pos_eo_coord_%s_*.tfrecord'
file_pattern_for_counting = 'pos_eo_coord'

items_to_descriptions = {
    'image': 'A 3-channel RGB synthetique wind turbine image currently made by blender.',
    'label': 'the (x,y) coordinate of the 6 postions we want to predict, mainly the 3 blade Tips, the hub and mast top/bottom'
}

#================= Layers of inception(where to start training) ==================
inception_layers = ['features/Conv2d_1a_3x3/',
                    'features/Conv2d_2a_3x3/',
                    'features/Conv2d_2b_3x3/',
                    'features/MaxPool_3a_3x3/',
                    'features/Conv2d_3b_1x1/',
                    'features/Conv2d_4a_3x3/',
                    'features/MaxPool_5a_3x3/',
                    'features/Mixed_5b/',
                    'features/Repeat/',
                    'features/Mixed_6a/',
                    'features/Repeat_1/',
                    'features/Mixed_7a/',
                    'features/Repeat_2/',
                    'features/Block8/',
                    'features/Conv2d_7b_1x1/']


#================= TRAINING INFORMATION ==================
num_epochs = 5000

batch_size = 1

num_classes = 12

image_height = 600
image_width = 1060

initial_learning_rate = 1e-4
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

trainFrom = inception_layers[-1]
print("Starting Training from {}".format(trainFrom))

freezeBatchNorm = False

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

#======================= LOADING DATA =========================
with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = get_split('train', dataset_dir, file_pattern, file_pattern_for_counting, items_to_descriptions, num_classes)
    images, _, labels = load_batch(dataset, batch_size, num_classes,height=image_height, width=image_width)

    num_batches_per_epoch = int(dataset.num_samples / batch_size)
    num_steps_per_epoch = num_batches_per_epoch
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

#======================= ADD-ON INCEPTION MODEL =========================

    googleNet = InceptionResnetV2("Inception", images, trainFrom=trainFrom, freezeBatchNorm=freezeBatchNorm)

    net = googleNet.getOutput("Conv2d_7b_1x1")

    net = slim.conv2d(net, 2600, 3, stride=2, padding='VALID', scope='added_Conv2d_1_3x3')

    net = slim.conv2d(net, 4096, 3, stride=2, padding='VALID', scope='added_Conv2d_2_3x3')

    net = slim.flatten(net,scope = "Flatten_layer")

    net = slim.fully_connected(net, 4096, scope='added_FC')

    net = slim.fully_connected(net,num_classes, scope='added_output')

#======================= TRAINING PROCESS =========================

    loss = tf.reduce_mean(tf.losses.mean_squared_error(predictions=net,labels=labels))
    total_loss = tf.losses.get_total_loss()

    global_step = get_or_create_global_step()

    lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)

    train_op = slim.learning.create_train_op(total_loss, optimizer)

    MSE, MSE_update = tf.contrib.metrics.streaming_mean_squared_error(net, labels)
    metrics_op = tf.group(MSE_update, net)

    tf.summary.scalar('losses/Total_Loss', total_loss)
    tf.summary.scalar('MSE', MSE)
    tf.summary.scalar('learning_rate', lr)
    my_summary_op = tf.summary.merge_all()

    def train_step(sess, train_op, global_step):
        '''
        Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
        '''
        start_time = time.time()
        total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
        time_elapsed = time.time() - start_time

        logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

        return total_loss, global_step_count

    with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8)) as sess:
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        if not CheckpointLoader.loadCheckpoint(sess, log_dir+"/save/",None):
            print("Loading Inception-resnet-v2 ... ")
            googleNet.importWeights(sess, "./inception_resnet_v2_2016_08_30.ckpt",includeTraining=True)
            print("Done.")
        sess.run(tf.local_variables_initializer())
        tf.train.start_queue_runners(sess=sess)
        for step in range(num_steps_per_epoch * num_epochs):
            #At the start of every epoch, show the vital information:
            if step % num_batches_per_epoch == 0:
                logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
                learning_rate_value, accuracy_value = sess.run([lr, MSE])
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming MSE: %s', accuracy_value)

                predictions_value, labels_value = sess.run([net, labels])
                print('predictions: \n', predictions_value)
                print('Labels:\n:', labels_value)
            if step % 2 == 0:
                loss,_ = train_step(sess, train_op, global_step)
                summaries = sess.run(my_summary_op)
                summary_writer.add_summary(summaries, step)
            if step % 1000 :
                saver.save(sess, log_dir + "/model.ckpt",step)
        logging.info('Final Loss: %s', loss)
        logging.info('Final MSE: %s', sess.run(MSE))
        logging.info('Finished training! Saving model to disk now.')
