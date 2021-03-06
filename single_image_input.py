# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:00:07 2018

@author: lh248
"""

from __future__ import division
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from scipy.ndimage.filters import convolve
import scipy as sci
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
os.environ['CUDA_VISIBLE_DEVICES']='0'

# default parameters
beta = 1   # weight of the l1 loss for generator

def lrelu(x):
    return tf.maximum(0.2*x,x)

def Generator(label,sp):
    dim=512
    input = tf.image.resize_area(label,(sp,sp),align_corners=True)
    net=slim.conv2d(input,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    sp *= 2
    dim = 512
    net=tf.image.resize_bilinear(net,(sp,sp),align_corners=True)
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net1=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
    net1=(net1+1.0)/2.0*255.0

    sp *= 2
    dim = 256
    net=tf.image.resize_bilinear(net,(sp,sp),align_corners=True)
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net2=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv200')
    net2=(net2+1.0)/2.0*255.0
        
    sp *= 2
    dim = 256
    net=tf.image.resize_bilinear(net,(sp,sp),align_corners=True)
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net3=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv300')
    net3=(net3+1.0)/2.0*255.0
    return net1, net2, net3

def Discriminator_img(image,dim):
    # input image is 256 x 256 x input_dim            
    net=slim.conv2d(image,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_img'+str(dim)+'_conv1')
    net=slim.max_pool2d(net,[2, 2],scope='d_img'+str(dim)+'_pool1')    
    # 128 x 128 x dim
    net=slim.conv2d(net,dim*2,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_img'+str(dim*2)+'_conv2')
    net=slim.max_pool2d(net,[2, 2],scope='d_img'+str(dim*2)+'_pool2') 
    # 64 x 64 x dim*2
    net=slim.conv2d(net,dim*4,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_img'+str(dim*4)+'_conv3')
    net=slim.max_pool2d(net,[2, 2],scope='d_img'+str(dim*4)+'_pool3') 
    # 32 x 32 x dim*4
    net=slim.conv2d(net,dim*8,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_img'+str(dim*8)+'_conv4')
    net=slim.max_pool2d(net,[2, 2],scope='d_img'+str(dim*8)+'_pool4') 
    # 16 x 16 x dim*8
    net=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='d_img'+str(dim*8)+'_conv5')
    net=(net+1.0)/2.0
    # 16 x 16 x 1
    return net

def abs_criterion(real,fake):
    return tf.reduce_mean(tf.abs(fake-real))

def mae_criterion(real,fake):
    return tf.reduce_mean((real-fake)**2)

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

sess=tf.Session()
is_training=True
sp=32#spatial resolution: 32x32 -> 256x256
with tf.variable_scope(tf.get_variable_scope()):
    real_image = tf.placeholder(tf.float32,[None,None,None,1])

    im64, im128, im256 = Generator(real_image,sp)
    Dis_fake_img = Discriminator_img(im256,32)
    g_loss_img_real = sce_criterion(Dis_fake_img,tf.ones_like(Dis_fake_img))
    l1 = abs_criterion(tf.image.resize_area(real_image,(sp*2,sp*2)), im64)
    l2 = abs_criterion(tf.image.resize_area(real_image,(sp*4,sp*4)), im128)
    l3 = abs_criterion(real_image, im256)
    content_loss = l1+l2+l3
    G_loss = g_loss_img_real + content_loss*beta

    fake_image = tf.placeholder(tf.float32,[None,None,None,1])
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        Dis_fake_img = Discriminator_img(fake_image,32)
        Dis_real_img = Discriminator_img(real_image,32)
    d_loss_img_real = sce_criterion(Dis_real_img,tf.ones_like(Dis_real_img))
    d_loss_img_fake = sce_criterion(Dis_fake_img,tf.zeros_like(Dis_fake_img))
    D_img_loss = d_loss_img_real + d_loss_img_fake
    
    g_loss_img_real_sum = tf.summary.scalar("g_loss_img_real",g_loss_img_real)
    G_loss_sum = tf.summary.scalar("G_loss",G_loss)
    d_loss_img_real_sum = tf.summary.scalar("d_loss_img_real",d_loss_img_real)
    d_loss_img_fake_sum = tf.summary.scalar("d_loss_img_fake",d_loss_img_fake)
    D_img_loss_sum = tf.summary.scalar("D_img_loss",D_img_loss)
    
    weight=tf.placeholder(tf.float32)
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
D_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(D_img_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('d_img')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("result_256p")


if is_training:
    g_loss=np.zeros(10010,dtype=float)
    d_img_loss=np.zeros(10010,dtype=float)
    label_images=[None]*10010
    for epoch in range(1,101):
        if os.path.isdir("result_256p/%05d"%epoch):
            continue
        cnt=0
        for ind in np.random.permutation(10000)+1:
            st=time.time()
            cnt+=1
            if label_images[ind] is None:
                label_images[ind]=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256_All48/%05d.png"%ind)),axis=0)#training image
                label_images[ind]=np.expand_dims(label_images[ind],axis=3)
            _,G_current,gen_img,gen_loss_img_real,l1_loss=sess.run([G_opt,G_loss,im256,g_loss_img_real,content_loss],feed_dict={real_image:label_images[ind],lr:1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            _,D_img_current,dis_loss_img_real,dis_loss_img_fake=sess.run([D_opt,D_img_loss,d_loss_img_real,d_loss_img_fake],feed_dict={real_image:label_images[ind],fake_image:gen_img,lr:1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            g_loss[ind]=G_current
            d_img_loss[ind]=D_img_current
            if cnt%10==0:
                print("ep:%d cnt:%d g:%.2f d_img:%.2f g_img_r:%.2f l1:%.2f d_img_r:%.2f d_img_f:%.2f tm:%.2f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(d_img_loss[np.where(d_img_loss)]),np.mean(gen_loss_img_real),np.mean(l1_loss),np.mean(dis_loss_img_real),np.mean(dis_loss_img_fake),time.time()-st))
        os.makedirs("result_256p/%05d"%epoch)
        target=open("result_256p/%05d/score.txt"%epoch,'w')
        target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
        target.write("%f"%np.mean(d_img_loss[np.where(d_img_loss)]))
        target.close()
        saver.save(sess,"result_256p/model.ckpt")
        if epoch%100==0:
            saver.save(sess,"result_256p/%05d/model.ckpt"%epoch)
        for ind in range(10501,11501):
            if not os.path.isfile("data/DICPC/PC256/%05d.png"%ind):#test label
                continue
            test_image=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256_All48/%05d.png"%ind)),axis=0)#test image
            test_image=np.expand_dims(test_image,axis=3)
            output=sess.run(im256,feed_dict={real_image:test_image})
            output=np.minimum(np.maximum(output,0.0),255.0)
            scipy.misc.toimage(output[0,:,:,0],cmin=0,cmax=255).save("result_256p/%05d/%06d_output.jpg"%(epoch,ind))

if not os.path.isdir("result_256p/final"):
    os.makedirs("result_256p/final")
for ind in range(10001,10501):
    if not os.path.isfile("data/DICPC/PC256_All48/%05d.png"%ind):#test label
        continue
    test_image=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256_All48/%05d.png"%ind)),axis=0)#test image
    test_image=np.expand_dims(test_image,axis=3)
    output=sess.run(im256,feed_dict={real_image:test_image})
    output=np.minimum(np.maximum(output, 0.0), 255.0)
    scipy.misc.toimage(output[0,:,:,0],cmin=0,cmax=255).save("result_256p/final/%06d_output.jpg"%ind)
