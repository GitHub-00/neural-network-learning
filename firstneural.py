import pickle
import numpy as np
import os
import tensorflow as tf

CIRFAR_DIR = os.path.abspath(os.path.join(os.getcwd(),'../cifar-10-batches-py'))
print (os.listdir(CIRFAR_DIR))

def lad_data(filename):
    """read data from file"""
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        return data[b'data'], data[b'labels']


x = tf.placeholder(tf.float32,[None, 3072])
y = tf.placeholder(tf.int64, [None])

#(3072,1)
w = tf.get_variable('w', [x.get_shape()[-1],1],
                          initializer=tf.random_normal_initializer(0,1))

#[1,]
b = tf.get_variable('b' ,[1],
                    initializer=tf.constant_initializer(0.0))

#[None, 3072]*(3072,1)=[None,1]
y_ = tf.matmul(x,w)+b

p_y_l = tf.nn.sigmoid(y_)

y_reshaped = tf.reshape(y,(-1,1))

y_reshaped_float = tf.cast(y_reshaped, tf.float32)

loss = tf.reduce_mean(tf.square(y_reshaped_float-p_y_l))

predict = p_y_l>0.5

correct_prediction = tf.equal(tf.cast(predict, tf.int64), y_reshaped)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

#梯度下降的方法
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)