# tensorflow_bp.py
# 训练bp神经网络并进行验证
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#mnist 数据集相关的常数
INPUT_NODE = 784 #输入层的节点数 对于mnist就是图片的像素
OUTPUT_NODE = 10 #输出层的节点数 

#配置的神经网络参数
LAYER1_NODE = 500 #隐藏层节点数
BATCH_SIZE = 100  #一个训练batch中的训练数据个数

LEARNING_RATE_BASE = 0.8     #基础的学习率
LEARNING_RATE_DECAY = 0.99   #学习率的衰减
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 5000        #训练轮数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率

#定义一个 使用ReLU激活函数的 三层 全连接 前向传播 神经网络
#ReLU激活函数实现去线性化
def inference(input_tensor,avg_class,weights1,biases1,weights2,biases2):
    #当没有提供滑动平均类时，直接使用参数的原始值
    if avg_class == None:
        #计算隐藏层的前向传播结果，使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights1)+biases1)
        #计算输出层的前向传播结果
        #因为在计算损失函数时会一并计算softmax函数，因此这里不需要加入激活函数
        return tf.matmul(layer1,weights2)+biases2
    else:
        #首先使用avg_class.average函数来计算得出变量的滑动平均值
        #然后再计算
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor,avg_class.average(weights1)) +
            avg_class.average(biases1))
        return tf.matmul(layer1,avg_class.average(weights2)) + \
                avg_class.average(biases2)

#训练模型过程
def train(mnist):
    #建立x，y的格式
    x = tf.placeholder(tf.float32,[None,INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,OUTPUT_NODE],name='y-input')

    #生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE,LAYER1_NODE],stddev=0.1)) #truncated_normal从正态分布中输出随机值
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))  #constant生成常量节点 初值为0.1
    #生成输出层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE,OUTPUT_NODE],stddev=0.1)) #truncated_normal从正态分布中输出随机值
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))  #constant生成常量节点 初值为0.1

    #计算在当前参数下神经网络向前传播的结果
    #因为滑动平均类的参数为None，所以函数不会进行滑动平均
    y = inference(x,None,weights1,biases1,weights2,biases2)

    # 定义存储训练轮数的变量 该变量为不可训练变量trainable=False
    # global_step为网络中迭代的轮数，可用于动态控制衰减率
    global_step = tf.Variable(0, trainable=False)
    # 定义一个滑动平均的类，初始化给定了衰减率0.99，和控制衰减的变量global_step
    # 给定训练轮数的变量global_step可以加快训练早期变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    # 在所有代表神经网络参数的变量上使用滑动平均
    # tf.trainalbe_variables返回的就是图上的集合
    variable_averages_op = variable_averages.apply(tf.trainable_variables())
    # 滑动平均不会改变变量本身的取值，会维护一个影子变量来记录其滑动平均值
    # 所以调用时需要明确调用average函数
    average_y = inference(x,variable_averages,weights1,biases1,weights2,biases2)
    
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    # 当分类问题只有一个答案时，可以使用下面函数来加速交叉熵的计算
    # 第一个参数时神经网络前向传播结果，第二个是训练数据的正确答案
    # 因为标准答案是一个长度为10的一维数组，所以需要tf.argmax函数得到正确答案对应数组下标
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失，一般只计算神经网络边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    # 计算总损失 = 交叉熵损失 + 正则化损失
    loss = cross_entropy_mean + regularization
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, # 基础的学习率，随着迭代的进行，学习率在这个基础上递减
        global_step,        # 当前迭代的轮数
        BATCH_SIZE,         # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY)# 学习率衰减速度
    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(loss, global_step=global_step)
    # 在训练模型时，每过一遍数据既需要通过反向传播更新网络参数，又要更新参数的滑动平均值
    train_op = tf.group(train_step, variable_averages_op)
    # 判断两个张量的每一维是否相等，如果相等返回True
    correct_prediction = tf.equal(tf.argmax(average_y,1),tf.argmax(y_,1))
    # 将上面的布尔型数值转换为实数型，然后计算平均值。这个平均值就是模型在这一组数据上的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        # 准备验证数据
        validate_feed = {x: mnist.validation.images,
                        y_: mnist.validation.labels}
        # 准备测试数据 用于评价模型优劣
        test_feed = {x: mnist.test.images,
                    y_: mnist.test.labels}
        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value = sess.run([train_op, loss], feed_dict={x: xs, y_: ys})
            # 计算滑动平均模型在验证集和测试集上的正确率
            if i % 1000 == 0:# 每1000轮输出一次
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average"
                    "model is %g, test is %g, loss is %g"%(i,validate_acc,test_acc,loss_value))
       
# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST-data",one_hot=True)
    train(mnist)
# TensofFlow提供一个主程序入口，tf.app.run会调用定义的main函数
if __name__ == '__main__':
    tf.app.run()