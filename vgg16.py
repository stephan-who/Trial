########################################################################################
# Davi Frossard, 2016                                                                  #
# VGG16 implementation in TensorFlow                                                   #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  confusion_matrix
import os
import sys
import datetime
import gc
from tensorflow.contrib.layers.python.layers import batch_norm

# 修改
tf.set_random_seed(10)
tf_dtype = tf.float32
def batch_normal(input , scope="scope" , reuse=False):
    return batch_norm(input , epsilon=1e-5, decay=0.9 , scale=True, scope=scope , reuse = reuse , updates_collections=None)



class vgg16:
    def __init__(self, imgs, label, class_num, learning_rate,keep_prob,count):
        self.imgs = imgs
        self.label = label
        self.count = count
        self.classes = class_num
        self.lr = learning_rate
        self.keep_prob = keep_prob
        self.convlayers()
        self.fc_layers()
        self.probs = self.fc3l
        print('probs', self.probs.get_shape())
        self.optimize()

    def convlayers(self):
        tf.glorot_normal_initializer()
        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,1,64],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.imgs, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(batch_normal(out,scope="c_bn1"), name=scope)
            self.conv1_1 = tf.nn.dropout(conv1_1,keep_prob=self.keep_prob)

            # tf.summary.histogram("conv1_1",self.conv1_1)
        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,64,64],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(batch_normal(out,scope="c_bn2"), name=scope)
            self.conv1_2 = tf.nn.dropout(conv1_2,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv1_2", self.conv1_2)
        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool1')

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,64,128],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(batch_normal(out,scope="c_bn3"), name=scope)
            self.conv2_1 = tf.nn.dropout(conv2_1,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv2_1", self.conv2_1)
        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,128,128],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(batch_normal(out,scope="c_bn4"), name=scope)
            self.conv2_2 = tf.nn.dropout(conv2_2,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv2_2", self.conv2_2)
        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,128,256],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(batch_normal(out,scope="c_bn5"), name=scope)
            self.conv3_1 = tf.nn.dropout(conv3_1,keep_prob=self.keep_prob)

            # tf.summary.histogram("conv3_1", self.conv3_1)
        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,256,256],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(batch_normal(out,scope="c_bn6"), name=scope)
            self.conv3_2 = tf.nn.dropout(conv3_2,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv3_2", self.conv3_2)
        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,256,256],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(batch_normal(out,scope="c_bn7"), name=scope)
            self.conv3_3 = tf.nn.dropout(conv3_3,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv3_3", self.conv3_3)
        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,256,512],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(batch_normal(out,scope="c_bn8"), name=scope)
            self.conv4_1 = tf.nn.dropout(conv4_1,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv4_1", self.conv4_1)
        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,512,512],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(batch_normal(out,scope="c_bn9"), name=scope)
            self.conv4_2 = tf.nn.dropout(conv4_2,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv4_2", self.conv4_2)
        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,512,512],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(batch_normal(out,scope="c_bn10"), name=scope)
            self.conv4_3 = tf.nn.dropout(conv4_3,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv4_3", self.conv4_3)
        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,512,512],initializer=tf.glorot_normal_initializer())

            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(batch_normal(out,scope="c_bn11"), name=scope)
            self.conv5_1 = tf.nn.dropout(conv5_1,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv5_1", self.conv5_1)
        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,512,512],initializer=tf.glorot_normal_initializer())
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(batch_normal(out,scope="c_bn12"), name=scope)
            self.conv5_2 = tf.nn.dropout(conv5_2,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv5_2", self.conv5_2)
        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            # kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf_dtype,
            #                                          stddev=1e-1), name='weights')
            kernel = tf.get_variable('weights',shape=[3,3,512,512],initializer=tf.glorot_normal_initializer())

            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf_dtype),
                                 trainable=True, name='biases')
            out = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(batch_normal(out,scope="c_bn13"), name=scope)
            self.conv5_3 = tf.nn.dropout(conv5_3,keep_prob=self.keep_prob)
            # tf.summary.histogram("conv5_3", self.conv5_3)
        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        # fc1
        with tf.name_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            # fc1w = tf.Variable(tf.truncated_normal([shape, 512],
            #                                              dtype=tf_dtype,
            #                                              stddev=1e-1), name='weights')
            fc1w = tf.get_variable('weights',shape=[shape,512],initializer=tf.glorot_normal_initializer())
            fc1b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf_dtype),
                                 trainable=True, name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            #self.fc1 = tf.nn.relu(fc1l)
            fc1 = tf.nn.relu(batch_normal(fc1l,scope="fc_bn1"))
            self.fc1 = tf.nn.dropout(fc1,keep_prob=self.keep_prob)
            # tf.summary.histogram("fc1", self.fc1)
        # fc2
        # with tf.name_scope('fc2') as scope:
        #     fc2w = tf.Variable(tf.truncated_normal([512, 512],
        #                                                  dtype=tf_dtype,
        #                                                  stddev=1e-1), name='weights')
        #     fc2b = tf.Variable(tf.constant(1.0, shape=[512], dtype=tf_dtype),
        #                          trainable=True, name='biases')
        #     fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
        #     #self.fc2 = tf.nn.relu(fc2l)
        #     fc2 = tf.nn.relu(batch_normal(fc2l,scope="fc_bn2"))
        #     self.fc2 = tf.nn.dropout(fc2, keep_prob=self.keep_prob)
            # tf.summary.histogram("fc2", self.fc2)
        # fc3
        with tf.name_scope('fc3') as scope:
            # fc3w = tf.Variable(tf.truncated_normal([512, self.classes],
            #                                              dtype=tf_dtype,
            #                                              stddev=1e-1), name='weights')
            fc3w = tf.get_variable('weights',shape=[512,self.classes],initializer=tf.glorot_normal_initializer())

            fc3b = tf.Variable(tf.constant(1.0, shape=[self.classes], dtype=tf_dtype),
                                 trainable=True, name='biases')
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc1, fc3w), fc3b)
            # tf.summary.histogram("fc3", self.fc3l)

    def optimize(self):
        # 学习率不断变化
        with tf.variable_scope("learn_param") as scope:
            lr = tf.get_variable("learning_rate", shape=[], dtype=tf_dtype,
                                 initializer=tf.constant_initializer(self.lr), trainable=False)
            self.lr_dynamic = lr
        # self.lr_dynamic = tf.get_variable("dynamic_learning_rate", shape=[], dtype=tf_dtype,
        #                  initializer=tf.constant_initializer(lr), trainable=False)

        # 定义损失函数和metric
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.probs, labels=self.label)
        self.cross_entropy_loss = tf.reduce_mean(cross_entropy, name="cross_entropy_loss")
        # self.regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name="regular_loss")
        # self.total_loss = tf.add(self.cross_entropy_loss, self.regular_loss, name="total_loss")
        self.total_loss = self.cross_entropy_loss
        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        # optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
        self.optimize_op = optimizer.minimize(self.total_loss)

def construct_train_imgs(img_dir, classes):
    # 这些类铁皮的根目录
    base_dir = img_dir
    origin_dataset = ['Normal', 'Chip', 'RedIron', 'Hole', 'Wrinkle', 'Dirty']
    label_dict = {'Normal': 0, 'Chip': 1, 'RedIron': 2, 'Hole': 3, 'Wrinkle': 4, 'Dirty': 5}
    # 对正常的随机取3w张
    # i 就是类别的编号
    img_list = []
    label_list = []
    for i, cls in enumerate(origin_dataset):
        imgs = os.listdir(os.path.join(base_dir, cls))
        #调试时先固定一个normal，数量为30000
        # if cls == 'Normal':
        #      imgs = np.random.choice(imgs, 30000)
        for img in imgs:
            if img[-3:] == 'jpg':
                # 把图像和它所在的文件夹连起来,方便后面操作
                img_list.append(os.path.join(cls, img))
                label_list.append(i)

    # 先把测试集划分出来
    total_num = len(img_list)
    all_imgs = np.asarray(img_list)
    all_labels = np.asarray(label_list)

    perm = np.arange(total_num)
    #修改
    # np.random.seed(10)
    # np.random.shuffle(perm)
    all_imgs = all_imgs[perm]
    all_labels = all_labels[perm]

    X = all_imgs
    y = all_labels

    return X,y

    # 10折交叉验证 StratifiedKFold可以确保训练集和验证集中每一类的样本数比较均衡

    #训练数据
    # all_imgs = []
    # all_labels = []
    # test_imgs = []
    # test_labels = []
    # valid_imgs = []
    # valid_labels = []
    # train_dir = img_dir + '/Train'
    # test_dir = img_dir + '/Test'
    # valid_dir = img_dir + '/Valid'
    # for i,cls in enumerate(classes):
    #     imgs = os.listdir(os.path.join(train_dir, cls))
    #     if cls == 'Normal':
    #         imgs = np.random.choice(imgs, 28000)
    #
    #     select_imgs = []
    #     for img in imgs:
    #         if img[-3:]=='jpg':
    #             select_imgs.append(os.path.join(cls, img))
    #     all_imgs += select_imgs
    #     all_labels += [i]*len(select_imgs)
    # assert len(all_imgs) == len(all_labels)
    #
    # for i, cls in enumerate(classes):
    #     imgs = os.listdir(os.path.join(test_dir, cls))
    #     select_imgs = []
    #     for img in imgs:
    #         if img[-3:] == 'jpg':
    #             select_imgs.append(os.path.join(cls, img))
    #     test_imgs += select_imgs
    #     test_labels += [i] * len(select_imgs)
    #
    # for i, cls in enumerate(classes):
    #     imgs = os.listdir(os.path.join(valid_dir, cls))
    #     select_imgs = []
    #     for img in imgs:
    #         if img[-3:] == 'jpg':
    #             select_imgs.append(os.path.join(cls, img))
    #     valid_imgs += select_imgs
    #     valid_labels += [i] * len(select_imgs)
    #
    # total_num = len(all_imgs)
    # all_imgs = np.asarray(all_imgs)
    # all_labels = np.asarray(all_labels)
    # perm = np.arange(total_num)
    # np.random.shuffle(perm)
    # #随机打乱照片顺序
    # all_imgs = all_imgs[perm]
    # all_labels = all_labels[perm]
    #
    # total_test = len(test_imgs)
    # test_imgs = np.asarray(test_imgs)
    # test_labels = np.asarray(test_labels)
    # perm1 = np.arange(total_test)
    # np.random.shuffle(perm1)
    # test_imgs = test_imgs[perm1]
    # test_labels = test_labels[perm1]
    #
    # total_valid = len(valid_imgs)
    # valid_labels = np.asarray(valid_labels)
    # valid_imgs = np.asarray(valid_imgs)
    # perm2 = np.arange(total_valid)
    # np.random.shuffle(perm2)
    # valid_imgs = valid_imgs[perm2]
    # valid_labels = valid_labels[perm2]
    #
    # return all_imgs,all_labels,valid_imgs,valid_labels,test_imgs,test_labels

    # return all_imgs[0:int(0.7*total_num)], all_labels[0:int(0.7*total_num)],\
    #         all_imgs[int(0.7*total_num):int(0.8*total_num)], all_labels[int(0.7*total_num):int(0.8*total_num)],\
    #         all_imgs[int(0.8*total_num):], all_labels[int(0.8*total_num):]

def generate_batch(input_imgs, input_label, step, batch_size,img_dir):
    num_class = 6
    #img_dir = '/home/p40/ssd/Train'
    batch_img = input_imgs[step * batch_size: (step + 1) * batch_size]
    batch_label = input_label[step * batch_size: (step + 1) * batch_size]
    batch_imgs = [np.asarray(imread(os.path.join(img_dir, img)).astype(np.float))/255 for img in batch_img]
    batch_imgs = np.asarray(batch_imgs)
    batch_imgs = batch_imgs.reshape((batch_size, 32, 195, 1))
    # print('batch_imgs', batch_imgs)
    zero_matrix = np.zeros((batch_size, num_class))

    for i, bl in enumerate(batch_label):
        zero_matrix[i, bl] = 1
    # print('zero_matrix', zero_matrix)
    return batch_imgs, zero_matrix

def train_shuffle(train_imgs, train_labels):
    perm = np.arange(len(train_imgs))
    np.random.seed(10)
    np.random.shuffle(perm)
    return  train_imgs[perm], train_labels[perm]


def plot_confusion_matrix(cls_pred):
    # This is called from print_test_accuracy() below.
    # cls_pred is an array of the predicted class-number for
    # all images in the test-set.

    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

def train(img_dir,classes,X_train_final,y_train_final,X_valid,y_valid,count):
    #构建训练样本,按batch输入,正常类别随机取3w张图像
    # img_dir = '/home/p40/ssd/Train/'
    summary_path = '/home/p40/git/Trial/summary'
    # classes = ['Normal','Chip','RedIron','Hole','Wrinkle','Dirty']
    Train_Epochs = 100
    #先将lr放大十倍，即0.002
    Learning_rate = 0.00017
    Learning_rate_Decay = 0.999
    batch_size = 256
    # Keep_prob = 0.9

    train_imgs = np.asarray(X_train_final)
    train_labels = np.asarray(y_train_final)
    valid_imgs = np.asarray(X_valid)
    valid_label = np.asarray(y_valid)
    #train_imgs, train_labels, valid_imgs, valid_label, test_imgs, test_label = construct_train_imgs(img_dir, classes)
    imgs = tf.placeholder(tf_dtype, [batch_size, 32, 195, 1])
    label = tf.placeholder(tf_dtype, [batch_size, len(classes)])
    Keep_prob = tf.placeholder(tf_dtype)
    vgg_name = 'vgg' + str(count)
    vgg_name = vgg16(imgs=imgs, label=label, class_num=len(classes), learning_rate=Learning_rate,keep_prob=Keep_prob,count=count)
    print('model done')
    print(vgg_name)
    saver = tf.train.Saver(max_to_keep=10)
    max_iter = int(len(train_imgs) / batch_size - 1)
    max_valid_iter = int(len(valid_imgs) / batch_size - 1)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    tf.summary.scalar("train_loss", vgg_name.total_loss)
    tf.summary.scalar("train_accuarcy", vgg_name.accuracy)
    tf.summary.scalar("learning_rate",vgg_name.lr_dynamic)


    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(summary_path+'/train', graph=sess.graph)
    summary_valid_writer = tf.summary.FileWriter(summary_path+'/test',graph= sess.graph)
    stop_count = 0
    all_best_accuracy = 0
    for epoch in range(Train_Epochs):
        # y_pred = np.zeros(shape=1024, dtype=np.int)
        # y_true = np.zeros(shape=1024, dtype=np.int)
        train_imgs, train_labels = train_shuffle(train_imgs, train_labels)
        valid_imgs, valid_label = train_shuffle(valid_imgs,valid_label)
        current_epoch_best_accuracy = 0
        print('train_imgs',len(train_imgs))
        print('split: ', count, '  epoch: ', epoch)
        for step in range(max_iter):
            trimgs, trlabels = generate_batch(train_imgs, train_labels, step, batch_size,img_dir)
            #训练
            _, total_loss, accuarcy,summary= sess.run([vgg_name.optimize_op, vgg_name.total_loss,
                                                       vgg_name.accuracy,merged_summary_op],
                                                       feed_dict={vgg_name.imgs:trimgs, vgg_name.label:trlabels,vgg_name.keep_prob:0.4})

            print('total_loss:', total_loss,' accuarcy:', accuarcy)
            summary_writer.add_summary(summary,(epoch)*max_iter+step)
            if (step % 100 == 0 and step != 0 ):
                # 验证集上验证
                valid_step = int(step/100)
                vimgs, vlabels = generate_batch(valid_imgs, valid_label, valid_step%max_valid_iter, batch_size, img_dir)
                # tf.summary.histogram("conv1_1", vgg_name.conv1_1)
                # tf.summary.histogram("conv1_2", vgg_name.conv1_2)
                # tf.summary.histogram("conv2_1", vgg_name.conv2_1)
                # tf.summary.histogram("conv2_2", vgg_name.conv2_2)
                # tf.summary.histogram("conv3_1", vgg_name.conv3_1)
                # tf.summary.histogram("conv3_2", vgg_name.conv3_2)
                # tf.summary.histogram("conv3_3", vgg_name.conv3_3)
                # tf.summary.histogram("conv4_1", vgg_name.conv4_1)
                # tf.summary.histogram("conv4_2", vgg_name.conv4_2)
                # tf.summary.histogram("conv4_3", vgg_name.conv4_3)
                # tf.summary.histogram("conv5_1", vgg_name.conv5_1)
                # tf.summary.histogram("conv5_2", vgg_name.conv5_2)
                # tf.summary.histogram("conv5_3", vgg_name.conv5_3)
                # tf.summary.histogram("fc1", vgg_name.fc1)
                # tf.summary.histogram("fc2", vgg_name.fc2)
                # y_true_cls = tf.argmax(vlabels, 1)
                # y_pred_cls = tf.argmax(vgg_name.probs,1)
                print('vlabels:', len(vlabels))
                total_loss, accuarcy,summary_valid = sess.run([vgg_name.total_loss, vgg_name.accuracy,merged_summary_op],
                                                   feed_dict={vgg_name.imgs:vimgs, vgg_name.label:vlabels,vgg_name.keep_prob:1.0})
                summary_valid_writer.add_summary(summary_valid,valid_step)
                if(accuarcy > current_epoch_best_accuracy):
                    current_epoch_best_accuracy = accuarcy
                print('split: ',count, '        epoch: ', epoch)
                print('evaluated on valid set')
                print('valid_total_loss:', total_loss, ' valid_accuarcy:', accuarcy)
                #降低学习率
                with tf.variable_scope("learn_param", reuse=True):
                    lr = tf.get_variable("learning_rate", shape=[], dtype=tf_dtype,
                                                 initializer=tf.constant_initializer(Learning_rate), trainable=False)
                    lr = tf.assign(lr, lr * Learning_rate_Decay)
                    sess.run(lr)
                #保存模型
                saver.save(sess, ("./log/iron_vgg16_model"+ str(count)), global_step=epoch)

            # 修改
            if epoch >75:
                if (step == (max_iter - 1)):
                    if current_epoch_best_accuracy > all_best_accuracy:
                        all_best_accuracy = current_epoch_best_accuracy
                        stop_count = 0
                    else:
                        stop_count = stop_count + 1

                    # 修改
                    if (stop_count == 10) or (epoch==99):
                        print("Accuracy doesn't improve on Valid Set in 10 epochs or epoch = 99!")
                        y_pred = np.zeros(shape=len(valid_imgs), dtype=np.int)
                        y_true = np.zeros(shape=len(valid_imgs), dtype=np.int)
                        for vstep in range(max_valid_iter):
                            vimgs, vlabels = generate_batch(valid_imgs, valid_label, vstep,
                                                            batch_size, img_dir)
                            y_pred_cls = tf.argmax(vgg_name.probs, 1)
                            y_true_cls = tf.argmax(vlabels, 1)
                            y_pred[vstep * batch_size:(vstep + 1) * batch_size], \
                            y_true[vstep * batch_size:(vstep + 1) * batch_size], = sess.run \
                                ([y_pred_cls, y_true_cls],feed_dict={vgg_name.imgs: vimgs, vgg_name.label: vlabels,vgg_name.keep_prob:1.0})
                        cm = confusion_matrix(y_true=y_true,y_pred=y_pred)
                        f = open('/home/p40/git/Trial/log/confusion_matrix.txt', 'a')
                        now = datetime.datetime.now()
                        f.write(now.strftime('%Y-%m-%d %H:%M:%S')+ "\n")
                        f.write("第" + str(count) + "折分类混淆矩阵：\n")
                        f.write(str(cm))
                        f.write("\n\n")
                        f.close()
                        del vgg_name
                        gc.collect()
                        break
        else:continue
        break
            #当epoch=100时，用测试集进行测试，并输出各类的分类准确率
            # f.write("第"+str(count)+"分类混淆矩阵：\n\n")
            # f.write(confusion_matrix(y_train_final,y_pred))
            # f.write("\n\n")
        # f.close()
if __name__ == '__main__':
    img_dir = '/home/p40/ssd/Train/'
    classes = ['Normal', 'Chip', 'RedIron', 'Hole', 'Wrinkle', 'Dirty']
    X,y = construct_train_imgs(img_dir,classes)
    #十折交叉
    # 修改
    skf = StratifiedKFold(n_splits=10, random_state=10, shuffle=True)
    count = 0
    gpu_num = sys.argv[1]
    print("gpu_num:",gpu_num)
    for train_index, valid_index in skf.split(X, y):
        count = count + 1
        print("count:",count)
        if gpu_num == str(0):
            if count == 3:
                break
        if gpu_num == str(1):
            if (count < 3):
                print("less than 3")
                continue
            if count == 5:
                break
        if gpu_num == str(2):
            if count < 5:
                print("less than 5")
                continue
            if count == 7:
                break
        if gpu_num == str(3):
            if count < 7:
                print("less than 5")
                continue
            if count == 9:
                break
            # 这就是每一折的训练集和验证集
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        file_open = open('/home/p40/git/Trial/log/splits_content.txt','a')
        now = datetime.datetime.now()
        file_open.write(now.strftime('%Y-%m-%d %H:%M:%S'))
        file_open.write("第" + str(count) + "折分类结果：\n\n")
        file_open.write("训练集及标签：\n\n")
        for x1,y1 in zip(X_train,y_train):
            file_open.write(str(x1) + " , label: " + str(y1) + "\n")

        file_open.write("验证集及标签：\n\n")
        for x2,y2 in zip(X_valid,y_valid):
            file_open.write(str(x2) + " , label: "+str(y2)+ "\n")
        file_open.close()
        y_train_num = len(y_train)
        y_valid_num = len(y_valid)
        print('y_train', y_train_num)
        print('y_valid', y_valid_num)

        for i in range(6):
            print(str(i), np.sum(y_train==i),np.sum(y_valid==i))
        # break

        X_train_aug = []
        y_train_aug = []
        # 然后需要把扩增的样本都加入到训练集中
        for x in X_train:
            # 判断其是否是磕碰 褶皱 和 孔洞其中之一
            x_class, x_img = x.split('/')
            if x_class == 'Chip':
                # 去磕碰增强的目录下面把增强的图像都加到aug里面
                for i in range(4):  # 4是增强的倍数
                    X_train_aug.append(os.path.join('ChipAug', x_img.split('.')[0] + '_' + str(i) + '.jpg'))
                    y_train_aug.append(1)
                X_train_aug.append(os.path.join('ChipAug',x_img.split(".")[0]+'_'+'lr'+ '_' + '0.jpg'))
                y_train_aug.append(1)
                X_train_aug.append(os.path.join('ChipAug', x_img.split(".")[0] + '_' + 'up' + '_' + '0.jpg'))
                y_train_aug.append(1)
            # 后面两类类似
            if x_class == 'Wrinkle':
                for i in range(4):  # 4是增强的倍数
                    X_train_aug.append(os.path.join('WrinkleAug', x_img.split(".")[0] + '_' + str(i) + '.jpg'))
                    y_train_aug.append(4)
                X_train_aug.append(os.path.join('WrinkleAug', x_img.split(".")[0] + '_' + 'lr' + '_' + '0.jpg'))
                y_train_aug.append(4)
                X_train_aug.append(os.path.join('WrinkleAug', x_img.split(".")[0] + '_' + 'up' + '_' + '0.jpg'))
                y_train_aug.append(4)
            if x_class == 'Hole':
                for i in range(2):  # 4是增强的倍数
                    X_train_aug.append(os.path.join('HoleAug', x_img.split(".")[0] + '_' + 'lr_0_noise_' + str(i) + '.jpg'))
                    y_train_aug.append(3)
                    X_train_aug.append(os.path.join('HoleAug', x_img.split(".")[0] + '_' + 'up_0_noise_' + str(i) + '.jpg'))
                    y_train_aug.append(3)
                    X_train_aug.append(os.path.join('HoleAug', x_img.split(".")[0] + '_' + 'noise_' + str(i) + '.jpg'))
                    y_train_aug.append(3)
                X_train_aug.append(os.path.join('HoleAug', x_img.split(".")[0] + '_' + 'lr_0' + '.jpg'))
                y_train_aug.append(3)
                X_train_aug.append(os.path.join('HoleAug', x_img.split(".")[0] + '_' + 'up_0' + '.jpg'))
                y_train_aug.append(3)

        # 最后进行合并, 就是这一折所构建的训练集和测试集
        X_train_final = list(X_train) + X_train_aug
        y_train_final = list(y_train) + y_train_aug

        # 后面可以用pickle把它们都保存下来, 以后用到的时候直接用pickle load进去就行
        with tf.Graph().as_default():
            train(img_dir,classes,X_train_final,y_train_final,X_valid,y_valid,count)
