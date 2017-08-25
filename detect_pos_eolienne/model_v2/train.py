import tensorflow as tf
from utils import *
from InceptionResnetV2 import *
import CheckpointLoader
import tensorflow.contrib.slim as slim

data_dir = "."
filename = "labelsxdata.txt"

image_height = 500
image_width = 500

dataset = Dataset(data_dir,filename,image_height,image_width)

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

trainFrom = inception_layers[0]
print("Starting Training from {}".format(trainFrom))

freezeBatchNorm = False
num_classes = 12
batch_size = 1
images = tf.placeholder(tf.uint8,[None,image_height,image_width,3],name="image")
keep_prob = tf.placeholder("float",name="dropout")
labels = tf.placeholder(tf.float32,[None,num_classes],name="labels")

print("Building the Model over Inception V2")
images_processed = preprocess(images)

googleNet = InceptionResnetV2("Inception", images_processed, trainFrom=trainFrom, freezeBatchNorm=freezeBatchNorm)

net = googleNet.getOutput("Conv2d_7b_1x1")

#conv1 = conv_layer(net,[3, 3, 1536, 1536],name="added_conv1")
conv1 = slim.conv2d(net, 2000, 3,
                        stride=1,
                        padding='SAME',
                        scope='added_Conv2d_1_3x3',
                        weights_initializer=tf.contrib.layers.xavier_initializer())
pool1 = slim.max_pool2d(conv1,2,stride=2,padding='SAME',scope='added_Pool_1')

conv2 = slim.conv2d(pool1, 3000, 3,
                        stride=1,
                        padding='SAME',
                        scope='added_Conv2d_2_3x3',
                        weights_initializer=tf.contrib.layers.xavier_initializer())
pool2 = slim.max_pool2d(conv2,2,stride=2,padding='SAME',scope='added_Pool_2')

conv3 = slim.conv2d(pool2, 4096, 3,
                            stride=1,
                            padding='SAME',
                            scope='added_Conv2d_3_3x3',
                            weights_initializer=tf.contrib.layers.xavier_initializer())
pool3 = slim.max_pool2d(conv3,2,stride=2,padding='SAME',scope='added_Pool_3')

flatten1 = slim.flatten(pool3,scope = "Flatten_layer")

fc1 = slim.fully_connected(flatten1, 4096, scope='added_FC_1',weights_initializer=tf.contrib.layers.xavier_initializer())
fc1 = slim.dropout(fc1,keep_prob=keep_prob)

fc2 = slim.fully_connected(fc1, 4096, scope='added_FC_2',weights_initializer=tf.contrib.layers.xavier_initializer())
fc2 = slim.dropout(fc2,keep_prob=keep_prob)


logits = slim.fully_connected(fc2,num_classes, scope='added_output',activation_fn=None,weights_initializer=tf.contrib.layers.xavier_initializer())

global_step = tf.Variable(0, trainable=False)

learning_rate = tf.train.exponential_decay(
        1e-5,                      # Base learning rate.
        global_step * batch_size,  # Current index into the dataset.
        dataset.shape[0]*15,                # Decay step.
        0.95,                      # Decay rate.
        staircase=True)


loss = tf.reduce_mean(tf.reduce_sum(tf.square(labels - logits), 1))


optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss,global_step=global_step)
#optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


MSE, MSE_update = tf.contrib.metrics.streaming_mean_squared_error(logits, labels)
metrics_op = tf.group(MSE_update, logits)

# Name operations, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')
MSE = tf.identity(MSE, name='MSE')



tf.summary.scalar("MSE", MSE)
tf.summary.scalar("Loss", loss)
tf.summary.scalar("Learning rate", learning_rate)

log_dir = "logs_2"

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

summary_op = tf.summary.merge_all()

sess = tf.Session()

sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())


print("Setting up saver and summary writer !")
saver = tf.train.Saver()
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

if not CheckpointLoader.loadCheckpoint(sess, log_dir,None):
    print("Loading Inception-resnet-v2 ... ")
    googleNet.importWeights(sess, "./inception_resnet_v2_2016_08_30.ckpt",includeTraining=True)
    print("Done.")

step = 1

while True :
    im,lab = dataset.get_example(train=True)
    if step % 10 == 0:
        start_time = time.time()
        l,mse,summary_str = sess.run([loss,MSE,summary_op],feed_dict={images:im,
                                               labels:lab,
                                               keep_prob:1.})
        time_elapsed = time.time() - start_time
        print("TRAINING : Iter",step,"Loss",l,"MSE",mse,"  ({} sec/step)".format(round(time_elapsed,2)))
        summary_writer.add_summary(summary_str, step)
    if step %100 ==0 :
        ls=[]
        mses = []
        ims_test,labs_test = dataset.get_all_records(test=True)
        start_time = time.time()
        for i in range(len(ims_test)):
            l,mse = sess.run([loss,MSE],feed_dict={images:ims_test[i].reshape(1,ims_test[i].shape[0],ims_test[i].shape[1],3),
                                               labels:labs_test[i].reshape(1,num_classes),
                                               keep_prob:1.})
            ls.append(l)
            mses.append(mse)
        time_elapsed = time.time() - start_time
        print("\nTEST on",len(ims_test),"Images : Iter",step,"Loss",np.mean(np.array(ls)),"MSE",np.mean(np.array(mses)),"  ({} sec/step)".format(round(time_elapsed,2)))

        ind = np.random.choice(len(ims_test),1,replace=False)

        pred = sess.run(logits,feed_dict={images:ims_test[ind],
                                               keep_prob:1.})
        lb = labs_test[ind]

        dataset.show(image=ims_test[ind][0],label=lb[0],pred=pred[0],save=True,show=False,name="test_preview.png")

        train_pred = sess.run(logits,feed_dict={images:im,
                                               keep_prob:1.})
        print("\nTEST LABELS\t=>\tTEST Predictions\t\t|\t\tTRAIN LABELS\t=>\tTRAIN Predictions")
        for ki in range(num_classes):
                print("%0.3f\t=>\t%0.03f\t\t\t\t|\t\t%0.3f\t=>\t%0.03f" % (lb[0][ki],pred[0][ki],lab[0][ki],train_pred[0][ki]))
        dataset.show(image=im[0],label=lab[0],pred=train_pred[0],save=True,show=False,name="train_preview.png")
    if step%1000 == 0 :
        print("\nSAVE : Iter",step)
        saver.save(sess, os.path.join(log_dir,"model.ckpt"), step)

    sess.run([optimizer,metrics_op,learning_rate], feed_dict={images:im,
                                labels:lab,
                                keep_prob:0.8})
    step+=1
