"""
multi neural hidden layers
"""

import pickle
import numpy as np
import os
import tensorflow as tf

CIRFAR_DIR = os.path.abspath(os.path.join(os.getcwd(),'../cifar-10-batches-py'))
#print (os.listdir(CIRFAR_DIR))

def load_data(filename):
    """read data from file"""
    with open(filename,'rb') as f:
        data = pickle.load(f,encoding='bytes')
        return data[b'data'], data[b'labels']

class CifarData:
    def __init__(self,filenames, need_shuffle):
        all_data = []
        all_labels = []
        for filename in filenames:
            data, labels = load_data(filename)

            all_data.append(data)
            all_labels.append(labels)

        self._data = np.vstack(all_data)
        #归一化
        self._data = self._data / 127.5 - 1
        self._labels = np.hstack(all_labels)
        self._num_examples = self._data.shape[0]
        self._need_shuffle = need_shuffle
        self._indicator = 0
        if self._need_shuffle:
            self._shuffle_data()


    def _shuffle_data(self):
        #[0,1,2,3,4]-->[4,3,2,1,0]
        p = np.random.permutation(self._num_examples)
        self._data = self._data[p]
        self._labels = self._labels[p]

    def next_batch(self,batch_size):
        end_indicator = self._indicator+batch_size
        if end_indicator > self._num_examples:
            if self._need_shuffle:
                self._shuffle_data()
                self._indicator = 0
                end_indicator = batch_size
            else:
                raise Exception("have no more examples")
        if end_indicator > self._num_examples:
            raise Exception("batch size is larget than all examples")
        batch_data = self._data[self._indicator: end_indicator]
        batch_labels = self._labels[self._indicator: end_indicator]
        self._indicator = end_indicator
        return batch_data, batch_labels

train_filenames = [os.path.join(CIRFAR_DIR,'data_batch_%d' % i) for i in range(1,6)]
test_filenames = [os.path.join(CIRFAR_DIR, 'test_batch')]

train_data = CifarData(train_filenames, True)
test_data = CifarData(test_filenames, False)

#batch_data, batch_labels = train_data.next_batch(10)

#print(batch_data)
#print(batch_labels)



x = tf.placeholder(tf.float32,[None, 3072])
y = tf.placeholder(tf.int64, [None])
hidden1 = tf.layers.dense(x, 100, tf.nn.relu)
hidden2 = tf.layers.dense(hidden1, 100, tf.nn.relu)
hidden3 = tf.layers.dense(hidden2, 50, tf.nn.relu)
y_ = tf.layers.dense(hidden3, 10)

#y_ -> sofmax
#y -> not hot
#loss = ylogy_
#交叉熵损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels=y , logits=y_)

"""
#(3072,10)
y_ = tf.layers.dense(x, 10) 等价于下面的代码
w = tf.get_variable('w', [x.get_shape()[-1],10],
                          initializer=tf.random_normal_initializer(0,1))

#[10,]
b = tf.get_variable('b' ,[10],
                    initializer=tf.constant_initializer(0.0))

#[None, 3072]*(3072,1)=[None,10]
y_ = tf.matmul(x,w)+b
"""

"""
p_y_l = tf.nn.softmax(y_)

#[0,0,0,0,0,0,0,0,0,0,1] one hot 结果示例
y_one_hot= tf.one_hot(y, 10 , dtype=tf.float32)

loss = tf.reduce_mean(tf.square(y_one_hot - p_y_l))
"""


predict = tf.argmax(y_, 1)

correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

#梯度下降的方法
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)


init = tf.global_variables_initializer()
batch_size = 20
train_steps = 100000
test_steps = 100

with tf.Session() as sess:
    sess.run(init)
    for i in range(train_steps):

        batch_data, batch_labels = train_data.next_batch(batch_size)
        loss_val, acc_val, _ =sess.run(
            [loss,accuracy,train_op],
            feed_dict={
                x : batch_data,
                y : batch_labels})
        if (i+1) % 500 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f '\
                  %(i+1, loss_val, acc_val))
        if (i+1) % 5000 == 0:
            test_data = CifarData(test_filenames, False)
            all_test_acc_val = []
            for j in range(test_steps):
                test_batch_data, test_batch_labels \
                    = test_data.next_batch(batch_size)
                test_acc_val = sess.run(
                    [accuracy],
                    feed_dict = {
                        x: test_batch_data,
                        y: test_batch_labels
                    }
                )
                all_test_acc_val.append(test_acc_val)
            tes_acc = np.mean(all_test_acc_val)
            print('[Test] Step: %d, loss: %4.5f, acc: %4.5f ' \
                  % (i + 1, loss_val, acc_val))


