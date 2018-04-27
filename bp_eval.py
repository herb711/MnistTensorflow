# bp_eval.py
# 用训练好的模型进行预测和验证
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import bp_inference as inf
import bp_train as tra

# 模型预测结果
def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出placeholder
        x = tf.placeholder(tf.float32,[None,inf.INPUT_NODE],name='x-input')
        y_ = tf.placeholder(tf.float32,[None,inf.OUTPUT_NODE],name='y-input')
        # 准备预测数据
        validate_feed = {x: mnist.test.images,
                        y_: mnist.test.labels}
        # 获取前向传播结果，因为预测不关注正则化损失的值，因此这里设置为None
        y = inf.inference(x, None)

        # 使用向前传播的结果计算正确率
        correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        # 使用前向传播预测
        #tf.argmax(y, 1) # 输出预测类别  

        # 设置滑动平均的系数
        variable_averages = tf.train.ExponentialMovingAverage(tra.MOVING_AVERAGE_DECAY)
        # 通过变量重命名的方式加载模型，这样在向前传播过程中不需要调用求滑动平均的函数来获取平均值
        saver = tf.train.Saver(variable_averages.variables_to_restore())
        
        # 执行
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(tra.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 加载模型
                #saver.restore(sess, "./model/model_bp.ckpt-5001")
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 计算正确率
                accuracy_score = sess.run(accuracy,feed_dict=validate_feed)
                print("validation accuracy using average model is %g"%(accuracy_score))
            else:
                print("No checkpoint file found")
                return 


# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST-data",one_hot=True)
    evaluate(mnist)
# TensofFlow提供一个主程序入口，tf.app.run会调用定义的main函数
if __name__ == '__main__':
    tf.app.run()    