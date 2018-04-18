# cnn_inference.py
# 卷积神经网络 LeNet5 前向传播函数
'''
INPUT: [28x28x1]           weights: 0
CONV5-32: [28x28x32]       weights: (5*5*1+1)*32
POOL2: [14x14x32]          weights: 0
CONV5-64: [14x14x64]       weights: (5*5*32+1)*64
POOL2: [7x7x64]          weights: 0
FC: [1x1x512]              weights: (7*7*64+1)*512
FC: [1x1x10]              weights: (1*1*512+1)*10
'''
import tensorflow as tf  
  
# 1.配置参数  
INPUT_NODE = 784  
OUTPUT_NODE = 10  
  
IMAGE_SIZE = 28  
NUM_CHANNELS = 1  
NUM_LABELS = 10  
# 第一层卷积层的尺寸和深度  
CONV1_DEEP = 32  
CONV1_SIZE = 5  
# 第二层卷积层的尺寸和深度  
CONV2_DEEP = 64  
CONV2_SIZE = 5  
# 全连接层的节点个数  
FC_SIZE = 512  
  
# 2.定义前向传播的过程  
# 这里添加了一个新的参数train，用于区分训练过程和测试过程。  
# 在这个程序中将用到dropout方法，dropout可以进一步提升模型可靠性并防止过拟合  
# dropout过程只在训练时使用  
def inference(input_tensor, train, regularizer):  
    # 声明第一层卷积层的变量并实现前向传播过程。  
    # 通过使用不同的命名空间来隔离不同层的变量  
    # 这可以让每一层中的变量命名只需要考虑在当前层的作用。不需要担心重名的问题  
    # 定义的卷积层输入为28*28*1的原始MNIST图片像素，使用全0填充后，输出为28*28*32  
    with tf.variable_scope('layer1-conv1'):  
        conv1_weights = tf.get_variable(  
            "weight",[CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_DEEP],  
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        conv1_biases = tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))  
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1,1,1,1],padding='SAME')  
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1,conv1_biases))  
  
    # 实现第二层池化层的前向传播过程。这一层输入为14*14*32  
    with tf.name_scope('layer2-pool1'):  
        pool1 = tf.nn.max_pool(relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')  
  
    # 实现第三层卷积层  
    with tf.variable_scope('layer3-conv2'):  
        conv2_weights = tf.get_variable(  
            "weight",[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],  
            initializer=tf.truncated_normal_initializer(stddev=0.1))  
        conv2_biases = tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))  
  
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1,1,1,1], padding='SAME')  
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))  
  
    # 实现第四层池化层  
    with tf.name_scope('layer4-pool2'):  
        pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  
        # 将第四层池化层的输出转化为第五层全连接的输入格式。  
        pool_shape = pool2.get_shape().as_list()  
        # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长度及深度的乘积。  
        nodes = pool_shape[1]* pool_shape[2]*pool_shape[3]  
        # 通过tf.reshape函数将第四层的输出变成一个batch的向量  
        reshaped = tf.reshape(pool2, [pool_shape[0],nodes])  
  
    #声明第五层全连接层的变量并实现前向传播过程  
    with tf.variable_scope('layer5-fc1'):  
        fc1_weights = tf.get_variable(  
            "weight", [nodes, FC_SIZE],  
            initializer=tf.truncated_normal_initializer(stddev=0.1)  
        )  
        if regularizer != None:  
            tf.add_to_collection('losses', regularizer(fc1_weights))  
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))  
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights)+fc1_biases)  
        if train:fc1 = tf.nn.dropout(fc1,0.5)  
  
    # 声明第六层全连接层的变量并实现前向传播过程  
    with tf.variable_scope('layer6-fc2'):  
        fc2_weights = tf.get_variable(  
            "weight", [FC_SIZE,NUM_CHANNELS], initializer=tf.truncated_normal_initializer(stddev=0.1)  
        )  
        if regularizer != None:  
            tf.add_to_collection('losses',regularizer(fc2_weights))  
        fc2_biases = tf.get_variable(  
            "bias",[NUM_LABELS],initializer=tf.constant_initializer(0.1)  
        )  
        logit = tf.matmul(fc1, fc2_weights)+ fc2_biases  
        return logit