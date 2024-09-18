#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a implementation of our LPNet structure:
# X. Fu, B. Liang, Y. Huang, X. Ding and J. Paisley. “Lightweight Pyramid Networks for Image Deraining”, IEEE Transactions on Neural Networks and Learning Systems, 2019.
# author: Xueyang Fu (xyfu@ustc.edu.cn)

import numpy as np

import tf_slim as slim
import tensorflow.compat.v1 as tf1
import tensorflow as tf
import vgg16
num_pyramids = 5  # number of pyramid levels
num_blocks = 5    # number of recursive blocks
num_feature = 16  # number of feature maps
num_channels = 3  # number of input's channels 
batch_size = 2

# leaky ReLU
def lrelu(x, leak = 0.2, name = 'lrelu'):   
    with tf1.variable_scope(name):
         return tf.maximum(x, leak*x, name = name)   



######## Laplacian and Gaussian Pyramid ########
def lap_split(img,kernel):
    with tf.name_scope('split'):
        low = tf.nn.conv2d(img, kernel, [1,2,2,1], 'SAME')
        low_upsample = tf.nn.conv2d_transpose(low, kernel*4, tf.shape(img), [1,2,2,1])
        high = img - low_upsample
    return low, high

def LaplacianPyramid(img,kernel,n):
    levels = []
    for i in range(n):
        img, high = lap_split(img, kernel)
        levels.append(high)
    levels.append(img)
    return levels[::-1]

def GaussianPyramid(img,kernel,n):
    levels = []
    low = img
    for i in range(n):
        low = tf.nn.conv2d(low, kernel, [1,2,2,1], 'SAME')
        levels.append(low)
    return levels[::-1]
######## Laplacian and Gaussian Pyramid ######## 



# create kernel
def create_kernel(name, shape, initializer=tf.keras.initializers.glorot_normal()):#

    regularizer = tf.keras.regularizers.l2(l = 1e-4)
    new_variables = tf1.get_variable(name=name, shape=shape, initializer=initializer,regularizer=regularizer)

    return new_variables


# sub network
def generator_model(images,patch_size):
    print(images.shape)

    #tf1.get_variable_scope().reuse_variables()
    with tf1.variable_scope('1st_layer',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         #tf1.get_variable_scope().reuse_variables()
         kernel0 = create_kernel(name='weights_0', shape=[3, 3, num_channels, num_feature])
         conv0 = tf.nn.conv2d(images, kernel0, [1, 1, 1, 1], padding='SAME')
         out_block = tf.nn.relu(conv0) # leaky ReLU
    print(out_block.shape)

  #  recursive blocks
    with tf1.variable_scope('DRBone',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
             #Dense Block
         kernel1 = create_kernel(name='weights_1', shape=[3, 3, num_feature, num_feature])
         conv1 = tf.nn.conv2d(out_block, kernel1, [1, 1, 1, 1], padding='SAME')
         bias1 = tf.nn.relu(conv1)
         bias2 = tf.reduce_sum([out_block, bias1],axis= 0)
         print(bias1.shape)
         print(bias2.shape)
         kernel2 = create_kernel(name='weights_2', shape=[3, 3, num_feature, num_feature])
         conv2 = tf.nn.conv2d(bias2, kernel2, [1, 1, 1, 1], padding='SAME')
         bias3 = tf.nn.relu(conv2)
         bias4 = tf.reduce_sum([out_block,bias3,bias1],axis=0)
         print(bias3.shape)
         kernel3 = create_kernel(name='weights_3', shape=[3, 3, num_feature, num_feature])
         conv3 = tf.nn.conv2d(bias4, kernel3, [1, 1, 1, 1], padding='SAME')
         bias5 = tf.nn.relu(conv3)
         print(bias5.shape)
         bias6 = tf.reduce_sum([out_block,bias5,bias3, bias1],axis=0)
         kernel7 = create_kernel(name='weights_7', shape=[1, 1, num_feature, num_feature])
         conv4 = tf.nn.conv2d(bias6, kernel7, [1, 1, 1, 1], padding='SAME')

    with tf1.variable_scope('DRBtwo',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         kernel4 = create_kernel(name='weights_4', shape=[5, 5, num_feature, num_feature])
         conv_1 = tf.nn.conv2d(out_block, kernel4, [1, 1, 1, 1], padding='SAME')
         bias_1 = tf.nn.relu(conv_1)
         print(bias_1.shape)
         bias_2 = tf.reduce_sum([out_block, bias_1],axis=0)
         kernel5 = create_kernel(name='weights_5', shape=[5, 5, num_feature, num_feature])
         conv_2 = tf.nn.conv2d(bias_2, kernel5, [1, 1, 1, 1], padding='SAME')
         bias_3 = tf.nn.relu(conv_2)
         print(bias_3.shape)
         bias_4 = tf.reduce_sum([out_block, bias_1, bias_3],axis=0)
         kernel6 = create_kernel(name='weights_6', shape=[5, 5, num_feature, num_feature])
         conv_3 = tf.nn.conv2d(bias_4, kernel6, [1, 1, 1, 1], padding='SAME')
         #bias_5 = tf.add(conv3, bias2, bias1)
         bias_5 = tf.nn.relu(conv_3)
         print(bias_5.shape)
         bias_6 = tf.reduce_sum([out_block,bias_1,bias_3,bias_5],axis=0) #  shortcut
         print(bias_6.shape)
         kernel8 = create_kernel(name='weights_8', shape=[1, 1, num_feature, num_feature])
         conv_4 = tf.nn.conv2d(bias_6, kernel8, [1, 1, 1, 1], padding='SAME')
         print(conv_4.shape)
         print("xin")
         #Feature Concat
         kernel9 = create_kernel(name='weights_9', shape=[3, 3, num_feature*2, num_feature])
         #Feature Fusion
         convff4 = tf.reduce_sum([conv4,out_block],axis=0)
         conv_ff4 = tf.reduce_sum([conv_4,out_block],axis=0)
         print(conv_ff4.shape)
         ronghe = tf.concat([convff4,conv_ff4],axis=3)
         print(ronghe.shape)
         ronghe1 = tf.nn.conv2d(ronghe, kernel9, [1, 1, 1, 1], padding='SAME')
         print(ronghe1.shape)
         kernel10 = create_kernel(name='weights_10', shape=[1, 1, num_feature, num_feature/2])
         ronghe2 = tf.nn.conv2d(ronghe1, kernel10, [1, 1, 1, 1], padding='SAME')
         #Adaptive Attention Module
         kernel11 = create_kernel(name='weights_11', shape=[1, 1, num_feature/2, 1])
         kernel12 = create_kernel(name='weights_12', shape=[3, 3, num_feature/2, 1])
         kernel13 = create_kernel(name='weights_13', shape=[5, 5, num_feature/2, 1])
         kernel14 = create_kernel(name='weights_14', shape=[7, 7, num_feature/2, 1])
         conv_attention_1 = tf.nn.conv2d(ronghe2, kernel11, [1, 1, 1, 1], padding='SAME')
         conv_attention_2 = tf.nn.conv2d(ronghe2, kernel12, [1, 1, 1, 1], padding='SAME')
         conv_attention_3 = tf.nn.conv2d(ronghe2, kernel13, [1, 1, 1, 1], padding='SAME')
         conv_attention_4 = tf.nn.conv2d(ronghe2, kernel14, [1, 1, 1, 1], padding='SAME')

         kernel15 = create_kernel(name='weights_15', shape=[1, 1, 1, 1])
         kernel16 = create_kernel(name='weights_16', shape=[1, 1, 1, 1])
         kernel17 = create_kernel(name='weights_17', shape=[1, 1, 1, 1])
         kernel18 = create_kernel(name='weights_18', shape=[1, 1, 1, 1])
         conv_weight_1 = tf.nn.conv2d(conv_attention_1, kernel15, [1, 1, 1, 1], padding='SAME')
         conv_weight_2 = tf.nn.conv2d(conv_attention_2, kernel16, [1, 1, 1, 1], padding='SAME')
         conv_weight_3 = tf.nn.conv2d(conv_attention_3, kernel17, [1, 1, 1, 1], padding='SAME')
         conv_weight_4 = tf.nn.conv2d(conv_attention_4, kernel18, [1, 1, 1, 1], padding='SAME')

         jiaquan = tf.reduce_sum([conv_attention_1 * tf.nn.softmax(conv_weight_1),conv_attention_2 * tf.nn.softmax(conv_weight_2),conv_attention_3 * tf.nn.softmax(conv_weight_3),conv_attention_4 * tf.nn.softmax(conv_weight_4)],axis=0)
         print(jiaquan.shape)
         #MCF
         pool_avg = tf.nn.avg_pool(input=ronghe1,ksize=[batch_size,patch_size,patch_size,num_feature], strides=[1,1,1,1], padding="VALID")
         print("xin")
         print(pool_avg.shape)
         mlp = multilayer_perceptron(pool_avg)
         print(mlp.shape)
         mean, var = tf.nn.moments(mlp, [1, 2])
         print(mean.shape)
         _,_,_, c = mlp.get_shape().as_list()
         print(c)
         scale = tf.ones((c), dtype=tf.float32)
         offset = tf.zeros((c), dtype=tf.float32)
         BN = tf.nn.batch_normalization(mlp, mean, var, offset, scale, variance_epsilon=0.01)
    #Reconstruction Module
    with tf1.variable_scope('recons',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         kernel19 = create_kernel(name='weights_19', shape=[3, 3, num_feature, 3])
         conv10 = tf.nn.conv2d(BN * jiaquan, kernel19, [1, 1, 1, 1], padding='SAME')
    print("xinxin")
    print(conv10.shape)
     #  shortcut
    final = tf.reduce_sum([conv10,images],axis=0)
    return final

def discriminator_model(images):
    #BN,RELU,MaxP
    #tf1.get_variable_scope().reuse_variables()
    with tf1.variable_scope('4st_layer',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         #tf1.get_variable_scope().reuse_variables()
         kernel_00 = create_kernel(name='weights_09', shape=[3, 3, num_channels, num_feature])
         #tf1.get_variable_scope().reuse_variables()
         conv0 = tf.nn.conv2d(images, kernel_00, [1, 1, 1, 1], padding='SAME')
         #bias0 = tf.nn.bias_add(conv0, biases0)
         mean, var = tf.nn.moments(conv0, [1, 2])
         _,_,_, c = conv0.get_shape().as_list()
         scale = tf.ones((c), dtype=tf.float32)
         offset = tf.zeros((c), dtype=tf.float32)
         conv1 = tf.nn.batch_normalization(conv0, mean, var, offset, scale, variance_epsilon=0.01)
         conv2 = tf.nn.relu(conv1)
         out_block = tf.nn.max_pool(conv2,ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")
         print(out_block.shape)
    # BN,RELU,MaxP
    with tf1.variable_scope('2st_layer',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         #out_block = tf.nn.relu(conv0) # leaky ReLU
         kernel_01 = create_kernel(name='weights_01', shape=[3, 3, num_feature, num_feature])
         conv3 = tf.nn.conv2d(out_block, kernel_01, [1, 1, 1, 1], padding='SAME')
         mean1, var1 = tf.nn.moments(conv3, [1, 2])
         _,_,_, c1 = conv3.get_shape().as_list()
         scale1 = tf.ones((c1), dtype=tf.float32)
         offset1 = tf.zeros((c1), dtype=tf.float32)
         conv4 = tf.nn.batch_normalization(conv3, mean1, var1, offset1, scale1, variance_epsilon=0.01)
         conv5 = tf.nn.relu(conv4)
         out_block1 = tf.nn.max_pool2d(conv5,ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")
    #Muti-scale module
    with tf1.variable_scope('3st_layer',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         kernel_03 = create_kernel(name='weights_03', shape=[3, 3, num_feature, num_feature])
         kernel_04 = create_kernel(name='weights_04', shape=[3, 3, num_feature, num_feature])
         kernel_05 = create_kernel(name='weights_05', shape=[3, 3, num_feature, num_feature])
         conv6 = tf.nn.atrous_conv2d(out_block1, kernel_03, rate=1, padding='SAME')
         conv7 = tf.nn.atrous_conv2d(out_block1, kernel_04, rate=3, padding='SAME')
         conv8 = tf.nn.atrous_conv2d(out_block1, kernel_05, rate=5, padding='SAME')
         """conv6 = tf.nn.conv2d(out_block1, kernel_03, [1, 1, 1, 1], padding='SAME',rate=1)
         conv7 = tf.nn.conv2d(out_block1, kernel_04, [1, 1, 1, 1], padding='SAME',rate=3)
         conv8 = tf.nn.conv2d(out_block1, kernel_05, [1, 1, 1, 1], padding='SAME',rate=5)"""
         conv9 = tf.concat([conv6,conv7,conv8],axis=3)
    #BN,ReLu,MaxP
    with tf1.variable_scope('5st_layer',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         kernel_02 = create_kernel(name='weights_02', shape=[1, 1, num_feature * 3, num_feature])
         conv10 = tf.nn.conv2d(conv9, kernel_02, [1, 1, 1, 1], padding='SAME')
         mean2, var2 = tf.nn.moments(conv10, [1, 2])
         _,_,_, c2 = conv10.get_shape().as_list()
         scale2 = tf.ones((c2), dtype=tf.float32)
         offset2 = tf.zeros((c2), dtype=tf.float32)
         conv11 = tf.nn.batch_normalization(conv10, mean2, var2, offset2, scale2, variance_epsilon=0.01)
         conv12 = tf.nn.relu(conv11)
         out_block2 = tf.nn.max_pool2d(conv12,ksize=[1,3,3,1], strides=[1,1,1,1], padding="SAME")
    #BN，Relu
    with tf1.variable_scope('6st_layer',tf1.get_variable_scope(), reuse=tf1.AUTO_REUSE):
         kernel_06 = create_kernel(name='weights_05', shape=[3, 3, num_feature, num_feature])
         conv13 = tf.nn.conv2d(out_block2, kernel_06, [1, 1, 1, 1], padding='SAME')
         mean3, var3 = tf.nn.moments(conv13, [1, 2])
         _,_,_, c3 = conv13.get_shape().as_list()
         scale3 = tf.ones((c3), dtype=tf.float32)
         offset3 = tf.zeros((c3), dtype=tf.float32)
         conv14 = tf.nn.batch_normalization(conv13, mean3, var3, offset3, scale3, variance_epsilon=0.01)
         conv15 = tf.nn.relu(conv14)
    #tf1.get_variable_scope().reuse_variables()
    #fully connected layer
    discriminator = model_layer()
    return discriminator(conv15)
#fully connected layer
def model_layer():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(1024,use_bias=False))#输入一个10维的随机向量
    model.add(tf.keras.layers.BatchNormalization())#使用了BN就不用偏置
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(1))
    return model

# Create model
def multilayer_perceptron(x):
    n_hidden_1 = 32  # 1st layer number of neurons
    n_hidden_2 = 16  # 2nd layer number of neurons
    n_classes = 16  # MNIST total classes (0-9 digits)

    weights = {
        'h1': tf.Variable(tf.random.normal([x.shape[3], n_hidden_1])),
        'h2': tf.Variable(tf.random.normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random.normal([n_hidden_2, n_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random.normal([n_hidden_1])),
        'b2': tf.Variable(tf.random.normal([n_hidden_2])),
        'out': tf.Variable(tf.random.normal([n_classes]))
    }
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)#二分类的交叉损失熵


