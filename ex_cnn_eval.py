#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载ex_cnn_inference.py 和 ex_cnn_train.py中定义的常量和函数
import ex_cnn_inference
import ex_cnn_train

# 每10秒加载一次最新的模型， 并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(tf.float32, [
            mnist.test.num_examples,                 # 第一维表示样例的个数
            ex_cnn_inference.IMAGE_SIZE,             # 第二维和第三维表示图片的尺寸
            ex_cnn_inference.IMAGE_SIZE,
            ex_cnn_inference.NUM_CHANNELS],          # 第四维表示图片的深度，对于RBG格式的图片，深度为5
                       name='x-input')
        y_ = tf.placeholder(tf.float32, [None, ex_cnn_inference.OUTPUT_NODE], name='y-input')

        # 直接通过调用封装好的函数来计算前向传播的结果。
        # 因为测试时不关注正则损失的值，所以这里用于计算正则化损失的函数被设置为None。
        y = ex_cnn_inference.inference(x, False, None)

        # 使用前向传播的结果计算正确率。
        # 如果需要对未知的样例进行分类，那么使用tf.argmax(y, 1)就可以得到输入样例的预测类别了。
        #correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动平均的函数来获取平局值了。
        # 这样就可以完全共用ex_cnn_inference.py中定义的前向传播过程
        variable_averages = tf.train.ExponentialMovingAverage(ex_cnn_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            t_start = time.time()
            label_ones = []
            for train_one in range(10):
                modelone_path = ex_cnn_train.MODEL_SAVE_PATH + str(train_one)
                # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名
                ckpt = tf.train.get_checkpoint_state(modelone_path)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    
                    reshaped_ys = [change_labels(train_one,y) for y in np.argmax(mnist.test.labels, 1)] #转换为二维矩阵
                    validate_feed = {
                        x: np.reshape(mnist.test.images, 
                            (mnist.test.num_examples, 
                            ex_cnn_inference.IMAGE_SIZE, 
                            ex_cnn_inference.IMAGE_SIZE, 
                            ex_cnn_inference.NUM_CHANNELS)),
                        y_: reshaped_ys}
                    # 运行得到预测值y
                    ys = sess.run(y, feed_dict = validate_feed)
                    
                    # [1,2]转换为[1,10]
                    label_one = [get_labels(train_one,y) for y in ys]
                    label_ones.append(label_one)
                    print("Use %s modle, get label = %d" % (ckpt.model_checkpoint_path, train_one))
                else:
                    print("No checkpoint file found")
                    return
            # 将10个label的数据整合起来再进行对比
            label_ys = sum(np.array(label_ones))
            ys = [np.argmax(y) for y in label_ys]
            ys_ = [np.argmax(y) for y in mnist.test.labels]
            correct = sum(int(x == y) for (x, y) in zip(ys,ys_))
            #correct = sum(int(x == y) for (x, y) in (np.argmax(label_ys), np.argmax(mnist.test.labels)))
            accuracy_score = correct / mnist.test.num_examples  
            print("validation accuracy = %f" % (accuracy_score))  
            sess.close()
            print('end...{0:.1f}s'.format(time.time()-t_start))

def change_labels(one,y):
    e = np.zeros((2))
    if int(y) == one:
        e[1] = 1.0 # 数组下标1，表示有效
    else:
        e[0] = 1.0 # 数组下标0，表示无效
    return e

def get_labels(one,y):
    e = np.zeros((10))
    e[one] = y[1] # 只将确实有效的概率记录下来
    return e

def main(argv=None):
    mnist = input_data.read_data_sets("MNIST-data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()