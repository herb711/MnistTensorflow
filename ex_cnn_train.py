# ex_cnn_train.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
忽略下面的提示
Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
上述提示告诉我们可以更好的提升tensorflow的运行速度，加入下面的代码可以不再出现该提示
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import time
import numpy as np

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载ex_cnn_inference.py中定义的常量和前向传播的函数
import ex_cnn_inference

# 配置神经网络的参数
BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARAZTION_RATE = 0.0001
TRAINING_STEPS = 200 # 30000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
MODEL_SAVE_PATH = "model/LeNet5/EX"
MODEL_NAME = "EX_LeNet5_model"

def train(mnist):
    # 定义输入输出placeholder
    # 调整输入数据placeholder的格式，输入为一个四维矩阵
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,                             # 第一维表示一个batch中样例的个数
        ex_cnn_inference.IMAGE_SIZE,             # 第二维和第三维表示图片的尺寸
        ex_cnn_inference.IMAGE_SIZE,
        ex_cnn_inference.NUM_CHANNELS],          # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, ex_cnn_inference.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 直接使用ex_cnn_inference.py中定义的前向传播过程
    y = ex_cnn_inference.inference(x, True, regularizer)
    global_step = tf.Variable(0, trainable=False)

    #定义损失函数、学习率、滑动平均操作以及训练过程
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        t_start = time.time()
        for train_one in range(10):
            print('start...train %d'%(train_one))
            # 初始化Tensorflow持久化类
            saver = tf.train.Saver(max_to_keep=1) # 对保存的数目限制为1，设置为None为无限制
            tf.global_variables_initializer().run() # 初始化参数
            # 验证和测试的过程将会有一个独立的程序来完成
            for i in range(TRAINING_STEPS):
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                #类似地将输入的训练数据格式调整为一个四维矩阵，并将这个调整后的数据传入sess.run过程
                reshaped_xs = np.reshape(xs, (BATCH_SIZE, ex_cnn_inference.IMAGE_SIZE, ex_cnn_inference.IMAGE_SIZE, ex_cnn_inference.NUM_CHANNELS))
                reshaped_ys = [change_labels(train_one,y) for y in np.argmax(ys, 1)] #转换为二维矩阵            
                # 运行
                _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: reshaped_ys})
                #每1000轮保存一次模型。
                if (i==0) | ((i+1)%100==0): # 1000
                    # 输出当前的训练情况。这里只输出了模型在当前训练batch上的损失函数大小。通过损失函数的大小可以大概了解训练的情况。
                    # 在验证数据集上的正确率信息会有一个单独的程序来生成。
                    print("After %d training step(s), loss on training batch is %f." % (step, loss_value))
                    # 保存当前的模型。注意这里隔出了global_step参数，这样可以让每个被保存模型的文件名末尾加上训练的轮数，比如“model.ckpt-1000”表示训练1000轮后得到的模型
                    modelone_path = MODEL_SAVE_PATH + str(train_one)
                    if not os.path.exists(modelone_path):
                        os.makedirs(modelone_path)
                    modelone_name = MODEL_NAME + str(train_one)
                    saver.save(sess, os.path.join(modelone_path, modelone_name), global_step=global_step)
        sess.close()
        print('end...{0:.1f}s'.format(time.time()-t_start))

def change_labels(one,y):
    e = np.zeros((2))
    if int(y) == one:
        e[1] = 1.0 # 数组下标1，表示有效
    else:
        e[0] = 1.0 # 数组下标0，表示无效
    return e

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST-data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()