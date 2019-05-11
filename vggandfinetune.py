
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


x = tf.placeholder(tf.float32,[None, 3072])
y = tf.placeholder(tf.int64, [None])

x_image = tf.reshape(x, [-1, 3, 32, 32] )
x_image = tf.transpose(x_image, perm=[0, 2, 3, 1] )

#32*32*3
conv1_1 = tf.layers.conv2d(x_image,
                          32, #output channel number
                          kernel_size=(3,3), #kernel size
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv1_1')
conv1_2 = tf.layers.conv2d(conv1_1,
                          32, #output channel number
                          kernel_size=(3,3), #kernel size
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv1_2')

pooling1 = tf.layers.max_pooling2d(conv1_2,
                                   (2,2),
                                   (2,2),
                                   name='pool1')

conv2_1 = tf.layers.conv2d(pooling1,
                          32, #output channel number
                          kernel_size=(3,3), #kernel size
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv2_1')
conv2_2 = tf.layers.conv2d(conv2_1,
                          32, #output channel number
                          kernel_size=(3,3), #kernel size
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv2_2')
#8*8
pooling2 = tf.layers.max_pooling2d(conv2_2,
                                   (2,2),
                                   (2,2),
                                   name='pool2')

conv3_1 = tf.layers.conv2d(pooling2,
                         32, #output channel number
                         kernel_size=(3,3), #kernel size
                         padding='same',
                         activation=tf.nn.relu,
                         name='conv3_1')
conv3_2 = tf.layers.conv2d(conv3_1,
                          32, #output channel number
                          kernel_size=(3,3), #kernel size
                          padding='same',
                          activation=tf.nn.relu,
                          name='conv3_2')
#4*4*32
pooling3 = tf.layers.max_pooling2d(conv3_2,
                                   (2,2),
                                   (2,2),
                                   name='pool3')
#
flatten = tf.layers.flatten(pooling3)

y_ = tf.layers.dense(flatten, 10)

#y_ -> sofmax
#y -> not hot
#loss = ylogy_展平
#交叉熵损失函数
loss = tf.losses.sparse_softmax_cross_entropy(labels=y , logits=y_)


predict = tf.argmax(y_, 1)

correct_prediction = tf.equal(predict, y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

#梯度下降的方法
with tf.name_scope('train_op'):
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#应用到各个节点看输出
def variable_summayr(var, name):
    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

        tf.summary.scalar('mean', mean)
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.histogram('histogrm', var)

with tf.name_scope('summary'):
    variable_summayr(conv1_1 , 'conv1_1')
    variable_summayr(conv1_2 , 'conv1_2')
    variable_summayr(conv2_1 , 'conv2_1')
    variable_summayr(conv2_2 , 'conv2_2')
    variable_summayr(conv3_1 , 'conv3_1')
    variable_summayr(conv3_2 , 'conv3_2')



#tensorboard
#指定显示的变量
loss_summary  = tf.summary.scalar("loss", loss)
accuracy_summary = tf.summary.scalar("accurary", accuracy)

#输入图片
source_image = (x_image+1)*127.5
inputs_summary = tf.summary.image("inputs_image", source_image)

merged_summary = tf.summary.merge_all()
merged_summary_test = tf.summary.merge([loss_summary, accuracy_summary])

#tesnsorboard输出指定文件夹
LOG_DIR = '.'
run_label = 'run_vgg_tensorboard'
run_dir = os.path.join(LOG_DIR, run_label)
if not os.path.exists(run_dir):
    os.mkdir(run_dir)

train_log_dir = os.path.join(run_dir, 'train')
test_log_dir = os.path.join(run_dir, 'test')

if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)

if not os.path.join(test_log_dir):
    os.mkdir(test_log_dir)

"""
fine tune
1, 保存模型
2, 还原节点（相当于断点恢复）
3, keep some layera fixed 利用参数trainable
"""
#fine tune
model_dir = os.path.join(run_dir, 'model')
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

saver = tf.train.Saver()
model_name = 'ckp-10000' #eg
model_path = os.path.join(model_dir, model_name)


init = tf.global_variables_initializer()
batch_size = 20
train_steps = 10000
test_steps = 100

output_summary_every_steps = 100
output_model_every_steps = 100

with tf.Session() as sess:
    sess.run(init)
    #计算图
    train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
    test_writer = tf.summary.FileWriter(test_log_dir)

    fixed_test_batch_data, fixed_test_batch_labels \
        =test_data.next_batch(batch_size)

    if os.path.exists(model_path + '.index'):
        saver.restore(sess, model_path)
        print('model restore from %s' , model_path)
    else:
        print('model %s does not exist' ,  model_path)

    for i in range(train_steps):

        batch_data, batch_labels = train_data.next_batch(batch_size)

        eval_ops = [loss, accuracy, train_op]

        should_output_summary = (i+1) % output_summary_every_steps == 0

        if should_output_summary:
            eval_ops.append(merged_summary)

        eval_ops_results = sess.run(
             eval_ops,
             feed_dict={
                 x: batch_data,
                 y: batch_labels
             }
         )

        loss_val, acc_val = eval_ops_results[0:2]

        if should_output_summary:
            train_summary_str = eval_ops_results[-1]
            train_writer.add_summary(train_summary_str, i+1)
            test_summary_str = sess.run(
                [merged_summary_test],
                feed_dict={
                    x:fixed_test_batch_data,
                    y:fixed_test_batch_labels
                }
            )[0]
            test_writer.add_summary(test_summary_str, i+1)

        """
        loss_val, acc_val, _ =sess.run(
            [loss,accuracy,train_op],
            feed_dict={
                x : batch_data,
                y : batch_labels})
        """
        if (i+1) % 100 == 0:
            print('[Train] Step: %d, loss: %4.5f, acc: %4.5f '\
                  %(i+1, loss_val, acc_val))
        if (i+1) % 1000 == 0:
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
        if (i+1) % output_model_every_steps == 0:
            saver.save(sess, os.path.join(model_dir, 'ckp-%05d' %(i+1)))
            print('model save to ckp-%05d' %(i+1))


