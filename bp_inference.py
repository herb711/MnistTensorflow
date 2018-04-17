# bp_inference.py
# BP神经网络 前向传播函数
import tensorflow as tf

#mnist 数据集相关的常数
INPUT_NODE = 784 #输入层的节点数 对于mnist就是图片的像素
OUTPUT_NODE = 10 #输出层的节点数 

#配置的神经网络参数
LAYER1_NODE = 500 #隐藏层节点数

# 通过tf.get_variable函数来获取变量，在训练神经网络时会创建这些变量
def get_weight_variable(shape, regularizer):
    weights = tf.get_variable("weights",shape,
                initializer=tf.truncated_normal_initializer(stddev=0.1))
    # 当给出正则化生成函数时，将当前变量的正则化损失加入losses集合
    # losses是自定义集合，不在tensorflow自动管理的集合列表中
    if regularizer != None: 
        # regularization = regularizer(weights1) + regularizer(weights2)
        tf.add_to_collection('losses',regularizer(weights))
    return weights

def inference(input_tensor, regularizer):
    # 声明第一层隐层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable([INPUT_NODE,LAYER1_NODE],regularizer)
        biases = tf.get_variable("biases",[LAYER1_NODE],
                    initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor,weights)+biases)
    # 输出层
    with tf.variable_scope('layer2'):
        weights = get_weight_variable([LAYER1_NODE,OUTPUT_NODE],regularizer)
        biases = tf.get_variable("biases",[OUTPUT_NODE],
                    initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1,weights)+biases
    return layer2
    