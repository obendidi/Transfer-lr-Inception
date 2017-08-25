import scipy.misc as misc
import cv2
import numpy as np
import tensorflow as tf
import os
import time


initializer = tf.contrib.layers.xavier_initializer()

def output_layer(x,shape,name='output'):
    with tf.name_scope(name) as scope:
        Weights = tf.get_variable("weights_"+name, shape=shape)
        bias = tf.Variable(tf.constant(0.1, shape=[shape[-1]]), name = 'bias_'+name)
        return tf.matmul(x, Weights) + bias

def fully_connected_layer(x,shape,keep_prob,dropconnect=False,name ="Fully_Connected"):
    with tf.name_scope(name) as scope:
        Weights = tf.get_variable("weights_"+name, shape=shape)
        bias = tf.Variable(tf.constant(0.1, shape=[shape[-1]]), name = 'bias_'+name)
        fc = tf.matmul(x, Weights) + bias
        fc = tf.nn.relu(fc)
        if dropconnect :
            return tf.nn.dropout(fc, keep_prob)*keep_prob
        else : return tf.nn.dropout(fc, keep_prob)

def flatten(x,shape,name='Flatten'):
    with tf.name_scope(name) as scope:
        flattened = tf.reshape(x, [-1, shape])
        return flattened

def batch_norm_layer(x,train,name='batch_norm'):
    with tf.name_scope(name) as scope:
        conv = tf.contrib.layers.batch_norm(x,is_training=train,updates_collections=None)
        return tf.nn.relu(conv)

def conv_layer(x, filt_shape, name="conv",strides = [1,1,1,1],padding = "SAME"):
    with tf.name_scope(name) as scope:
        Weights = tf.get_variable("weights_"+name, shape=filt_shape)
        bias = tf.Variable(tf.constant(0.1, shape=[filt_shape[-1]]), name = 'bias_'+name)
        conv = tf.add(tf.nn.conv2d(x, Weights, strides=strides, padding=padding),bias)
        return conv
def max_pool_layer(x,k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

def reshape_data(x,shape,name="reshape_data"):
    with tf.name_scope(name) as scope:
        x = tf.reshape(x, shape)
        return x

def preprocess(x,name="Preprocess_Image"):
    with tf.name_scope(name) as scope:
        tf.summary.image('image',x)
        x = tf.cast(x,tf.float32)/255.0
        x = tf.subtract(x, 0.5)
        x = tf.multiply(x, 2.0)
        return x

class Dataset(object):
    """
    Dataset
    """
    def __init__(self,data_dir,filename,image_height,image_width,train_ratio=0.9):
        """
        Initalize the class
        :param dataset_name: Database name
        :param data_files: List of files in the database
        """
        data_file = os.path.join(data_dir,filename)
        print("Reading data into memory ! (this a very bad way to do it) it might take a while ( ͡° ͜ʖ ͡°)")
        print("in the meanwhile you can check how to make tf records with the script in dataset/")
        start_time = time.time()
        files = [line.rstrip() for line in open(data_file)]
        self.data = [ misc.imresize(misc.imread(filename.split(" ")[-1],mode="RGB"),
                                                [image_height,image_width],interp="nearest") for filename in files]

        self.labels = labels = [[float(filename.split(" ")[0]),
                                float(filename.split(" ")[1]),
                                float(filename.split(" ")[2]),
                                float(filename.split(" ")[3]),
                                float(filename.split(" ")[4]),
                                float(filename.split(" ")[5]),
                                float(filename.split(" ")[6]),
                                float(filename.split(" ")[7]),
                                float(filename.split(" ")[8]),
                                float(filename.split(" ")[9]),
                                float(filename.split(" ")[10]),
                                float(filename.split(" ")[11])] for filename in files]
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.shape = self.data.shape
        dt = (time.time() - start_time)
        print("data loaded succesfully in {} seconds!".format(round(dt,2)))
        print("shape of data is",self.data.shape)
        print("shape of labels is",self.labels.shape)

        if train_ratio is not None :
            print("splitting data for training !")
            train_ratio = int(train_ratio*self.shape[0])
            ind = range(self.shape[0])
            self.X_train = self.data[ind[:train_ratio],:]
            self.X_test = self.data[ind[train_ratio:],:]

            self.y_train = self.labels[ind[:train_ratio]]
            self.y_test = self.labels[ind[train_ratio:]]
            print("shape of train data/labels is",self.X_train.shape,"/",self.y_train.shape)
            print("shape of test data/labels is",self.X_test.shape,"/",self.y_test.shape)

    def get_all_records(self,train=False,test=False):
        if train :
            return self.X_train,self.y_train
        elif test :
            return self.X_test,self.y_test
        else:
            return self.data,self.labels

    def get_example(self,batch_size=1,train=True,test=False):

        if train :
            ind = np.random.choice(self.X_train.shape[0]-1,batch_size,replace=False)
            return self.X_train[ind],self.y_train[ind]
        elif test :
            ind = np.random.choice(self.X_test.shape[0]-1,batch_size,replace=False)
            return self.X_test[ind],self.y_test[ind]
        else:
            ind = np.random.choice(self.shape[0]-1,batch_size,replace=False)
            return self.data[ind],self.labels[ind]

    def show(self,image = None,label = None, pred = None, save=False,show=True,example=True,name=None):
        if image is not None and label is not None and pred is not None:
            if name is None :
                name="preview.png"
            image1 = image.copy()
            for i in range(0,len(label),2) :
                cx,cy = label[i]*self.shape[2],label[i+1]*self.shape[1]
                cv2.circle(image,(int(cx),int(cy)),5,(255,0,0),-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,'Labels',(20,20), font, 1,(255,0,0),2)

            for i in range(0,len(pred),2):
                cx,cy = pred[i]*self.shape[2],pred[i+1]*self.shape[1]
                cv2.circle(image1,(int(cx),int(cy)),5,(0,0,255),-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image1,'Predictions',(20,20), font, 1,(0,0,255),2)

            ov_image = np.concatenate((image,image1),axis=0)
            if save :
                cv2.imwrite(name,ov_image)
            if show :
                cv2.imshow("image",ov_image)
                cv2.waitKey(0)

        elif example :
            if name is None :
                name="exemple.png"
            ind = np.random.choice(self.shape[0]-1,1,replace=False)
            image = self.data[ind][0]
            label = self.labels[ind][0]
            for i in range(0,len(label),2):
                cx,cy = label[i]*self.shape[2],label[i+1]*self.shape[1]
                cv2.circle(image,(int(cx),int(cy)),5,(255,0,0),-1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image,'Labels',(10,20), font, 1,(255,0,0),2)

            if save :
                cv2.imwrite(name,image)
            if show :
                cv2.imshow("image",image)
                cv2.waitKey(0)

        else:
            print("Nothing to show here !")
