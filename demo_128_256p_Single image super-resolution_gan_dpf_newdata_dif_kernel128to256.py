# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 10:30:28 2018

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
radius1 = 2  # radius of kernel for 128 x 128 image
radius2 = 4  # radius of kernel for 256 x 256 image
R = 4       # Outer radius of phase ring
W = 0.8     # width of phase ring
zetap = 0.8 # amplitude attenuation factors caused by phase ring
M = 6       # size of dictionary
beta = 1   # weight of the l1 loss for generator

def somb(x):
    z = np.ones(x.shape, dtype=float)
    ind = x.nonzero()
    z[ind] = 2.0*sci.special.jn(1, np.pi*x[ind]) / (np.pi*x[ind])
    return z

def get_kernel(radius,R,W,zetap,ang):
    ran = np.arange(-radius,radius+1,1)
    X,Y = np.meshgrid(ran,ran)
    rr = np.sqrt(X.astype(float)**2 + Y.astype(float)**2)
    kernel1 = np.pi*(R**2)*somb(2*R*rr)
    kernel2 = np.pi*((R-W)**2)*somb(2*(R-W)*rr)
    kernelr = kernel1 - kernel2
    kerneli = (zetap*math.cos(ang)-math.sin(ang))*kernelr
    kernel = kerneli
    kernel[radius,radius] += math.sin(ang)
    kernel /= np.linalg.norm(kernel,ord=2)
    return kernel

kernel_1 = []
for m in range(1,M+1):
    ang = 2*m*math.pi/M
    kern = get_kernel(radius1,R,W,zetap,ang)
    kernel_1.append(kern)
Kernel1 = np.expand_dims(kernel_1, axis=3)
Kernel1 = np.expand_dims(Kernel1, axis=4)

kernel_2 = []
for m in range(1,M+1):
    ang = 2*m*math.pi/M
    kern = get_kernel(radius2,R,W,zetap,ang)
    kernel_2.append(kern)
Kernel2 = np.expand_dims(kernel_2, axis=3)
Kernel2 = np.expand_dims(Kernel2, axis=4)

def pattern_filter_tf(im,kernel):
    im_DPF = []
    for i in range(0,M):
        im_filter = tf.nn.conv2d(im,kernel[i],strides=[1,1,1,1],padding='SAME')
        im_DPF.append(im_filter)
    return im_DPF

def lrelu(x):
    return tf.maximum(0.2*x,x)

def Generator(label,sp):
    dim=256
    input = tf.image.resize_area(label,(sp,sp),align_corners=True)
    im_DPF = pattern_filter_tf(input,Kernel2)
    im_DPF = tf.squeeze(im_DPF,[4])    
    im_DPF = tf.transpose(im_DPF,[1,2,3,0])
    net=slim.conv2d(im_DPF,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    sp *= 2
    dim = 256
    net=tf.image.resize_bilinear(net,(sp,sp),align_corners=True)
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net1=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv110')
    net1=(net1+1.0)/2.0*255.0
    
    return net1

def Discriminator3(image,dim):
    # input image is 256 x 256 x input_dim            
    net=slim.conv2d(image,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_3'+str(dim)+'_conv1')
    net=slim.max_pool2d(net,[2, 2],scope='d_3'+str(dim)+'_pool1')    
    # 128 x 128 x dim
    net=slim.conv2d(net,dim*2,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_3'+str(dim*2)+'_conv2')
    net=slim.max_pool2d(net,[2, 2],scope='d_3'+str(dim*2)+'_pool2') 
    # 64 x 64 x dim*2
    net=slim.conv2d(net,dim*4,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_3'+str(dim*4)+'_conv3')
    net=slim.max_pool2d(net,[2, 2],scope='d_3'+str(dim*4)+'_pool3') 
    # 32 x 32 x dim*4
    net=slim.conv2d(net,dim*8,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='d_3'+str(dim*8)+'_conv4')
    net=slim.max_pool2d(net,[2, 2],scope='d_3'+str(dim*8)+'_pool4') 
    # 16 x 16 x dim*8
    net=slim.conv2d(net,1,[1,1],rate=1,activation_fn=None,scope='d_3'+str(dim*8)+'_conv5')
    net=(net+1.0)/2.0
    # 16 x 16 x 1
    return net

def abs_criterion(real,fake):
    real = tf.squeeze(real,[4])    
    real = tf.transpose(real,[1,2,3,0])
    fake = tf.squeeze(fake,[4])    
    fake = tf.transpose(fake,[1,2,3,0])
    return tf.reduce_mean(tf.abs(fake-real))

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

sess=tf.Session()
is_training=True
sp=128#spatial resolution: 128x128 -> 256x256
with tf.variable_scope(tf.get_variable_scope()):
    real_image = tf.placeholder(tf.float32,[None,None,None,1])

    im256 = Generator(real_image,sp)
    D3_fake = Discriminator3(im256,32)
    g3_loss = sce_criterion(D3_fake,tf.ones_like(D3_fake))
    l1 = abs_criterion(pattern_filter_tf(real_image,Kernel2), pattern_filter_tf(im256,Kernel2))
    content_loss = l1
    G_loss = g3_loss + content_loss*beta

    fake_image = tf.placeholder(tf.float32,[None,None,None,1])
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        D3_fake = Discriminator3(fake_image,32)
        D3_real = Discriminator3(real_image,32)
    d3_loss_real = sce_criterion(D3_real,tf.ones_like(D3_real))
    d3_loss_fake = sce_criterion(D3_fake,tf.zeros_like(D3_fake))
    D3_loss = d3_loss_real + d3_loss_fake
    
    weight=tf.placeholder(tf.float32)
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
D3_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(D3_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('d_3')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("result_256p")


if is_training:
    g_loss=np.zeros(10010,dtype=float)
#    d1_loss=np.zeros(10010,dtype=float)
#    d2_loss=np.zeros(10010,dtype=float)
    d3_loss=np.zeros(10010,dtype=float)
    label_images=[None]*10010
    for epoch in range(1,51):
        if os.path.isdir("result_256p/%04d"%epoch):
            continue
        cnt=0
        for ind in np.random.permutation(10000)+1:
            st=time.time()
            cnt+=1
            if label_images[ind] is None:
                label_images[ind]=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256_All48/%05d.png"%ind)),axis=0)#training image
                label_images[ind]=np.expand_dims(label_images[ind],axis=3)
            _,G_current,gen_img256,gen_loss3,l1_loss=sess.run([G_opt,G_loss,im256,g3_loss,content_loss],feed_dict={real_image:label_images[ind],lr:1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            _,D3_current,dis3_loss_real,dis3_loss_fake=sess.run([D3_opt,D3_loss,d3_loss_real,d3_loss_fake],feed_dict={real_image:label_images[ind],fake_image:gen_img256,lr:1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            g_loss[ind]=G_current
            d3_loss[ind]=D3_current
            if cnt%10==0:
                print("ep:%d cnt:%d G:%.2f D:%.2f g:%.2f d_r:%.2f d_f:%.2f l1:%.2f tm:%.2f"%(epoch, cnt, np.mean(g_loss[np.where(g_loss)]), np.mean(d3_loss[np.where(d3_loss)]), np.mean(gen_loss3), np.mean(dis3_loss_real), np.mean(dis3_loss_fake), np.mean(l1_loss), time.time()-st))
        os.makedirs("result_256p/%04d"%epoch)
        target=open("result_256p/%04d/score.txt"%epoch,'w')
        target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
        target.write("%f"%np.mean(d3_loss[np.where(d3_loss)]))
        target.close()
        saver.save(sess,"result_256p/model.ckpt")
        if epoch%100==0:
            saver.save(sess,"result_256p/%04d/model.ckpt"%epoch)
        for ind in range(10001,10051):
            if not os.path.isfile("data/DICPC/PC256_All48/%05d.png"%ind):#test label
                continue
            test_image=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256_All48/%05d.png"%ind)),axis=0)#test image
            test_image=np.expand_dims(test_image,axis=3)
            output1=sess.run(im256,feed_dict={real_image:test_image})
            output1=np.minimum(np.maximum(output1,0.0),255.0)
            scipy.misc.toimage(output1[0,:,:,0],cmin=0,cmax=255).save("result_256p/%04d/%05d_output1.jpg"%(epoch,ind))

if not os.path.isdir("result_256p/final"):
    os.makedirs("result_256p/final")
for ind in range(10001,10501):
    if not os.path.isfile("data/DICPC/PC256_All48/%05d.png"%ind):#test label
        continue
    test_image=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256_All48/%05d.png"%ind)),axis=0)#test image
    test_image=np.expand_dims(test_image,axis=3)
    output1=sess.run(im256,feed_dict={real_image:test_image})
    output1=np.minimum(np.maximum(output1,0.0),255.0)
    scipy.misc.toimage(output1[0,:,:,0],cmin=0,cmax=255).save("result_256p/final/%06d_output1.jpg"%ind)
