# bp_train.py
# 主训练类，用于训练逻辑
import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 加载前向传播函数
import bp_inference as inf

# 配置神经网络的参数
BATCH_SIZE = 100  #一个训练batch中的训练数据个数
LEARNING_RATE_BASE = 0.8     #基础的学习率
LEARNING_RATE_DECAY = 0.99   #学习率的衰减
REGULARIZATION_RATE = 0.0001 #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 5000        #训练轮数
MOVING_AVERAGE_DECAY = 0.99  #滑动平均衰减率

# 模型保存的路径和文件名
MODEL_SAVE_PATH = "./model/BP/"
MODEL_NAME = "BP_model" 

def train(mnist):
    # 定义输入输出placeholder
    x = tf.placeholder(tf.float32,[None,inf.INPUT_NODE],name='x-input')
    y_ = tf.placeholder(tf.float32,[None,inf.OUTPUT_NODE],name='y-input')
 
    # 正则化，用于防止过拟合
    # 当模型过于复杂之后，就会出现模型太过于匹配训练样本，而不能很好地适应测试样本，
    # 称这种现象为过拟合现象；
    # 正则化的思想是在损失函数中加入刻画模型复杂度的指标，这里成为正则化函数
    # 这样整个损失函数就是J(wi) + r*R(wi);这里的wi是指所有的权重w,
    # 显然当w过于复杂时，R(wi)就会越大，制约损失函数的值，反过来限制R(wi)的值
    # 参考链接：http://blog.csdn.net/jinping_shi/article/details/52433975
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)# REGULARIZATION_RATE是正则化权重

    # 通过输入数据和正则化函数向前传播
    y = inf.inference(x, regularizer)

    # 滑动平均模型：我的理解，为了防止权重等变量可能出现大的突升或者突降，我们使用了一个"缓兵之计"，
    # 即，使得变量变化不要太大，这样模型将更加稳定健壮。
    # variable2'=shadow_variable2=decay×shadow_variable1+(1−decay)×variable2
    # 这里的shadow_variable1是变量variable的初始值,表示为v1;公式中的variable2为variable改变后得值，为v2
    # decay为衰减率，variable2'-variable2为采用滑动后的-未采用的；
    # 显然，|variable2'-variable1|<|variable2-variable1|
    # 另外为了使模型在训练前期更新能够尽可能快，我们又对decay进行了函数处理；使得变化更大；
    # 比如变量X从0-10改变，decay=0.9；
    # 只单独采用滑动平均之后的值是0-1；在采用decay函数处理后，就是0-9，显然后面的变化更快一些
    # 想法：其实真这样，感觉滑动平均就有点多此一举了
    # decay=min{decay,1+num_updates / 10+num_updates}
    # 这里的num_updates参数就是下面的global_step，一般初始化为0，不可更改
    global_step = tf.Variable(0, trainable=False)
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    # 将定义的滑动平均variable_averages应用到所有的参数中，variables_averages_op即为参数的更新动作；
    # 即每执行一次variables_averages_op，就会更新一次全部的参数
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # 损失函数：损失函数有很多，这里使用交叉熵作为损失函数，H(p,q)=−∑xp(x)log(q(x))
    # 通俗的解释是交叉熵表示的是p和q之间的相似的，其中q是输出值所对应的概率，p是该样本的正确输出
    # 另外，q的曲解需要使用到softmax函数，这个函数主要使用在多分类中，目的是将分类结果转化为所出现的概率q
    # 所有的概率和为1
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 求均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    # 计算总损失 = 交叉熵损失 + 正则化损失
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))

    # 定义学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,                    # 基础学习率，之后的在这个基础上递减
        global_step,                           # 当前迭代轮数
        mnist.train.num_examples / BATCH_SIZE, # 训练完所有的样本需要的迭代次数
        LEARNING_RATE_DECAY,                   # 学习率衰减率
        staircase=False)                       # 默认为False表示连续衰减，True表示阶梯状衰减

    # 使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                    .minimize(loss, global_step=global_step)
    # control_dependenciesshi实现两个过程处理：反向传播更新参数train_step、滑动平均更新参数variables_averages_op
    # with指如果执行成功，就执行内部逻辑
    with tf.control_dependencies([train_step,variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化tensorflow持久化类
    saver = tf.train.Saver()

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        # 初始化变量
        tf.global_variables_initializer().run()

        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):
            # 产生这一轮使用的一个batch的训练数据，并运行训练过程
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = \
                sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出当前训练情况
                print("After %d training step(s), loss on training"
                    "batch is %g."%(step,loss_value))
                # 保存当前模型
                #save_path = os.getcwd() + MODEL_SAVE_PATH
                if not os.path.exists(MODEL_SAVE_PATH): 
                    os.makedirs(MODEL_SAVE_PATH) 
                # saver.save(sess, 'my-model', global_step=0) ==>filename: 'my-model-0'  
                saver.save(sess, os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)
                
# 主程序入口
def main(argv=None):
    # 声明处理MNIST数据集的类，这个类在初始化时会自动下载数据
    mnist = input_data.read_data_sets("MNIST-data",one_hot=True)
    train(mnist)
# TensofFlow提供一个主程序入口，tf.app.run会调用定义的main函数
if __name__ == '__main__':
    tf.app.run()       