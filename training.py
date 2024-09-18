#!/usr/bin/env python2
# -*- coding: utf-8 -*-


# This is a implementation of training code of this paper:
# X. Fu, B. Liang, Y. Huang, X. Ding and J. Paisley. “Lightweight Pyramid Networks for Image Deraining”, IEEE Transactions on Neural Networks and Learning Systems, 2019.
# author: Xueyang Fu (xyfu@ustc.edu.cn)


import os
import re
import time
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import matplotlib.pyplot as plt
import math
import model_xin
from vgg16 import vgg16
import cv2
from skimage import transform,data
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


num_pyramids = 5       # number of pyramid levels
learning_rate = 1e-3   # learning rate
iterations = int(3e5)  # iterations
batch_size = 2        # batch size
num_channels = 3       # number of input's channels 
patch_size = 80        # patch size 
save_model_path = './model/'  # path of saved model
model_name = 'model-epoch'    # name of saved model


input_path = './TrainData/input/' # rainy images
gt_path = './TrainData/label/'    # ground truth
 

# randomly select image patches
def _parse_function(input_path, gt_path, patch_size = patch_size):   
    image_string = tf.io.read_file(input_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    print(image_decoded)
    rainy = tf.cast(image_decoded, tf.float32)/255.0
    print(rainy)
    image_string = tf.io.read_file(gt_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)  
    label = tf.cast(image_decoded, tf.float32)/255.0
          
    t = time.time()
    Data = tf.random_crop(rainy, [patch_size, patch_size ,3],seed = t)   # randomly select patch
    Label = tf.random_crop(label, [patch_size, patch_size ,3],seed = t)
    return Data, Label


def _parse_function_xin(input_path, gt_path, patch_size=patch_size):
    image_string = tf.io.read_file(input_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    rainy = tf.cast(image_decoded, tf.float32) / 255.0

    image_string = tf.io.read_file(gt_path)
    image_decoded = tf.image.decode_png(image_string, channels=3)
    label = tf.cast(image_decoded, tf.float32) / 255.0

    t = time.time()
    Data = tf.random_crop(rainy, [patch_size//2, patch_size//2, 3], seed=t)  # randomly select patch
    Label = tf.random_crop(label, [patch_size//2, patch_size//2, 3], seed=t)
    return Data, Label
#sigmoid
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)#二分类的交叉损失熵
#判别器的损失
def discriminator_loss(real_out,fake_out):
    #希望tf.ones_like(real_out)不断地接近于1，tf.zeros_like(fake_out)fake与
    image_real_loss = cross_entropy(tf.ones_like(real_out),real_out)#cross_entropy(y_true,y_pred)
    image_fake_loss = cross_entropy(tf.zeros_like(fake_out),fake_out)
    return image_real_loss+image_fake_loss

#生成器的损失
def generator_loss(fake_out):
    #fake_out希望生成数据的数据为1
    return cross_entropy(tf.ones_like(fake_out),fake_out)


"""def train_step(images,labels):
    with tf.GradientTape() as gen_tape,tf.GradientTape() as disc_tape:
        real_out = model_xin.discriminator_model(labels,training=True)#（64，1）
        images_xin = model_xin.generator_model(images, training=True)
        fake_out = model_xin.discriminator_model(images_xin, training=True)#对生成数据进行检测
        #以上都是为了得到loss,通过记录loss的获得流程，最终通过调整参数利用优化器
        gen_loss = generator_loss(fake_out)#生成器希望让图像被判断为真
        disc_loss = discriminator_loss(real_out,fake_out)

    #通过损失函数得到梯度，
    gradient_gen = gen_tape.gradient(gen_loss,model_xin.generator_model().trainable_variables)#计算生成器损失和变量之间的梯度
    gradient_disc = disc_tape.gradient(disc_loss,model_xin.discriminator_model().trainable_variables)#计算生成器损失和变量之间的梯度

    #更新参数  通过结合梯度和改变自变量
    generator_optimizer.apply_gradients(zip(gradient_gen,model_xin.generator_model().trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradient_disc,model_xin.discriminator_model().trainable_variables))"""

if __name__ == '__main__':    
    tf.compat.v1.disable_v2_behavior()#reset_default_graph()


    input_files = os.listdir(input_path)
    for i in range(len(input_files)):
        input_files[i] = input_path + input_files[i]
    label_files = os.listdir(gt_path)       
    for i in range(len(label_files)):
        label_files[i] = gt_path + label_files[i]
    print(cv2.imread(input_files[0]).shape)
    """imagelist = []
    for r in input_files:
        imagelist.append(cv2.resize(cv2.imread(r),(int(cv2.imread(r).shape[0]//2),int(cv2.imread(r).shape[1]//2))))
    print(cv2.imread(label_files[i]))"""

    input_files = tf.convert_to_tensor(input_files, dtype=tf.string)
    label_files = tf.convert_to_tensor(label_files, dtype=tf.string)
    print(input_files[1])
    dataset = tf.data.Dataset.from_tensor_slices((input_files, label_files))
    datasets = tf.data.Dataset.from_tensor_slices((input_files, label_files))

    dataset = dataset.map(_parse_function)
    datasets = datasets.map(_parse_function_xin)

    dataset = dataset.prefetch(buffer_size = batch_size)
    datasets = datasets.prefetch(buffer_size=batch_size)

    dataset = dataset.batch(batch_size).repeat()
    datasets = datasets.batch(batch_size).repeat()
    #print(dataset)
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
    #print(iterator)
    iterators = tf.compat.v1.data.make_one_shot_iterator(datasets)

    inputs, labels = iterator.get_next()
    input, label = iterators.get_next()

    print(inputs)
    print(inputs.shape)
    print("jiege")
    print(inputs.shape[1])

    #print(cv2.resize(inputs,(int(inputs.shape[0]),int(inputs.shape[1]//2),int(inputs.shape[2]//2),int(inputs.shape[3]))))
    #print(cv2.resize(inputs,(int(inputs.shape[1]/2),int(inputs.shape[2]/2))).shape)#图片规模的一半
    print("xinyongjie")
    print("xinyongjie")
    #vgg16
    loss1 = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(vgg16(model_xin.generator_model(inputs,patch_size)).probs,vgg16(labels).probs))))
    #LE
    loss2 = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(model_xin.generator_model(inputs,patch_size),labels))))

    images_xin = model_xin.generator_model(inputs,patch_size)
    fake_out = model_xin.discriminator_model(images_xin)
    real_out = model_xin.discriminator_model(labels)  # （64，1）
    # 对生成数据进行检测
    # 以上都是为了得到loss,通过记录loss的获得流程，最终通过调整参数利用优化器
    gen_loss = generator_loss(fake_out)  # 生成器希望让图像被判断为真
    disc_loss = discriminator_loss(real_out, fake_out)
    #通过损失函数得到梯度
    #LA
    loss3 = gen_loss + disc_loss
    #loss4 = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(model_xin.generator_model(input),label))))

    """images_xins = model_xin.generator_model(input)
    fake_outs = model_xin.discriminator_model(images_xins)
    real_outs = model_xin.discriminator_model(label)  # （64，1）
    # 对生成数据进行检测
    # 以上都是为了得到loss,通过记录loss的获得流程，最终通过调整参数利用优化器
    gen_losses = generator_loss(fake_outs)  # 生成器希望让图像被判断为真
    disc_losses = discriminator_loss(real_outs, fake_outs)"""
    #1/2scale Lp
    loss4 = tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(vgg16(model_xin.generator_model(input,patch_size//2)).probs,vgg16(label).probs))))

    loss = loss1 + loss2 + loss3 + loss4

    g_optim =  tf1.train.AdamOptimizer(learning_rate).minimize(loss) # Optimization method: Adam
    all_vars = tf1.trainable_variables()
    saver = tf1.train.Saver(var_list=all_vars, max_to_keep = 5)
    config = tf1.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf1.Session(config=config) as sess:
      
       sess.run(tf1.group(tf1.global_variables_initializer(),tf1.local_variables_initializer()))
       tf1.get_default_graph().finalize()
              
       if tf.train.get_checkpoint_state(save_model_path):   # load previous trained model 
          ckpt = tf.train.latest_checkpoint(save_model_path)
          saver.restore(sess, ckpt)  
          ckpt_num = re.findall(r'(\w*[0-9]+)\w*',ckpt)
          start_point = int(ckpt_num[0]) + 1     
          print("loaded successfully")
       else:  # re-training when no models found
          print("re-training")
          start_point = 0  
          
       check_input, check_label =  sess.run([inputs,labels])
       print("check patch pair:")  
       plt.subplot(1,3,1)     
       plt.imshow(check_input[0,:,:,:])
       plt.title('input')         
       plt.subplot(1,3,2)    
       plt.imshow(check_label[0,:,:,:])
       plt.title('ground truth')      
       plt.show()    
     
       start = time.time()    
       #迭代学习
       for j in range(start_point,iterations):
           #train_step(inputs, labels)
           _, Training_Loss = sess.run([g_optim,loss])  # training
           if np.mod(j+1,100) == 0 and j != 0:    
              end = time.time() 
              print ('%d / %d iteraions, Training Loss  = %.4f, running time = %.1f s' % (j+1, iterations, Training_Loss, (end-start)))
              save_path_full = os.path.join(save_model_path, model_name) 
              saver.save(sess, save_path_full, global_step = j+1) # save model every 100 iterations
              start = time.time()  
              
       print ('training finished') 
    sess.close()