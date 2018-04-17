import tensorflow as tf  
import time  
import math  
import numpy as np  
import tensorflow as tf  
from tensorflow.examples.tutorials.mnist import input_data  
  
import cnn_inference  
import cnn_train  
  
def evaluate(mnist):  
    with tf.Graph().as_default() as g:  
        x = tf.placeholder(tf.float32, [  
            mnist.test.num_examples,  
            cnn_inference.IMAGE_SIZE,  
            cnn_inference.IMAGE_SIZE,  
            cnn_inference.NUM_CHANNELS],  
                           name='x-input')  
        y_ = tf.placeholder(tf.float32, [None,cnn_inference.OUTPUT_NODE],name='y-input')  
        validate_feed = {x:mnist.test.images,  
                         y_:mnist.test.labels}  
        global_step =tf.Variable(0,trainable=False)  
        regularizer = tf.contrib.layers.l2_regularizer(cnn_train.REGULARAZTION_RATE)  
  
        # 直接调用封装好的函数来计算前向传播的结果。  
        y = cnn_inference.inference(x, False, regularizer)  
  
        # 使用前向传播的结果计算正确率。使用tf.argmax(y,1)得到输入样例的预测类别  
        correct_prediction = tf.equal(tf.argmax(y, 1),tf.argmax(y_, 1))  
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
  
        # 通过变量重命名的方式来加载模型，这样在前向传播的过程中就不需要调用求滑动 #  
        # 平均的函数来获取平均值了。这样就可以完全公用mnist_inference.py中定义的 #  
        # 前向传播过程 #  
        variable_averages = tf.train.ExponentialMovingAverage(  
            cnn_train.MOVING_AVERAGE_DECAY)  
        variable_to_restore = variable_averages.variables_to_restore()  
        saver = tf.train.Saver(variable_to_restore)  
  
        # 每隔EVAL_INTERVAL_SECS秒调用一次计算正确率的过程以检测训练过程中正确率的变化  
        n = math.ceil(mnist.test.num_examples / mnist.test.num_examples)  
        for i in range(n):  
            with tf.Session() as sess:  
                ckpt = tf.train.get_checkpoint_state(cnn_train.MODEL_SAVE_PATH)  
                if ckpt and ckpt.model_checkpoint_path:  
                    saver.restore(sess, ckpt.model_checkpoint_path)  
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  
                    xs, ys = mnist.test.next_batch(mnist.test.num_examples)  
                    # xs, ys = mnist.test.next_batch(cnn_train.BATCH_SIZE)  
                    reshaped_xs = np.reshape(xs, (  
                        mnist.test.num_examples,  
                        # cnn_train.BATCH_SIZE,  
                        cnn_inference.IMAGE_SIZE,  
                        cnn_inference.IMAGE_SIZE,  
                        cnn_inference.NUM_CHANNELS))  
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_xs, y_: ys})  
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))  
                else:  
                    print('No checkpoint file found')  
                    return  
  
  
# 主程序  
def main(argv=None):  
    mnist = input_data.read_data_sets("MNIST-data", one_hot=True)  
    evaluate(mnist)  
  
  
if __name__ == '__main__':  
    main()