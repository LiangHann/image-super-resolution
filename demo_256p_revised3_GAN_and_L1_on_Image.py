# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:11:16 2018

@author: lh248
"""

from __future__ import division
import os,helper,time,scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import scipy as sci
import math

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
os.environ['CUDA_VISIBLE_DEVICES']='0'

# default parameters
radius = 2  # radius of kernel
R = 4       # Outer radius of phase ring
W = 0.8     # width of phase ring
zetap = 0.8 # amplitude attenuation factors caused by phase ring
M = 6       # size of dictionary

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

kernel = []
for m in range(1,M+1):
    ang = 2*m*math.pi/M
    kern = get_kernel(radius,R,W,zetap,ang)
    kernel.append(kern)
Kernel = np.expand_dims(kernel, axis=3)
Kernel = np.expand_dims(Kernel, axis=4)

def pattern_filter(im,kernel):
    im_DPF = []
    for i in range(0,M):
        im_filter = tf.nn.conv2d(im,kernel[i],strides=[1,1,1,1],padding='SAME')
        im_DPF.append(im_filter)
    return im_DPF

def lrelu(x):
    return tf.maximum(0.2*x,x)

def Generator(label,sp):
    dim=512
    input = tf.image.resize_area(label,(sp,sp),align_corners=True)
    im_DPF = pattern_filter(input,Kernel)
    im_DPF = tf.squeeze(im_DPF,[4])
    im_DPF = tf.transpose(im_DPF,[1,2,3,0])
    
    net=slim.conv2d(im_DPF,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    sp *= 2
    dim = 512
    net=tf.image.resize_bilinear(net,(sp,sp),align_corners=True)
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net11=slim.conv2d(net,M,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv100')
    net12=slim.conv2d(net11,1,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv110')
    net11=(net11+1.0)/2.0*255
    net12=(net12+1.0)/2.0*255

    sp *= 2
    dim = 256
    net=tf.image.resize_bilinear(net,(sp,sp),align_corners=True)
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net21=slim.conv2d(net,M,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv200')
    net22=slim.conv2d(net21,1,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv210')
    net21=(net21+1.0)/2.0*255
    net22=(net22+1.0)/2.0*255
        
    sp *= 2
    dim = 256
    net=tf.image.resize_bilinear(net,(sp,sp),align_corners=True)
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv1')
    net=slim.conv2d(net,dim,[3,3],rate=1,normalizer_fn=slim.layer_norm,activation_fn=lrelu,scope='g_'+str(sp)+'_conv2')
    net31=slim.conv2d(net,M,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv300')
    net32=slim.conv2d(net31,1,[1,1],rate=1,activation_fn=None,scope='g_'+str(sp)+'_conv310')
    net31=(net31+1.0)/2.0*255
    net32=(net32+1.0)/2.0*255
    return net11, net12, net21, net22, net31, net32

def compute_error(real,fake):
    return tf.reduce_mean(tf.abs(fake-real))#diversity loss


sess=tf.Session()
is_training=True
sp=32#spatial resolution: 32x32 -> 256x256
with tf.variable_scope(tf.get_variable_scope()):
    label = tf.placeholder(tf.float32,[None,None,None,1])
    real_image = tf.placeholder(tf.float32,[None,None,None,1])
    im64_DPF, im64, im128_DPF, im128, im256_DPF, im256 = Generator(label,sp)
    weight=tf.placeholder(tf.float32)
    l11 = compute_error(pattern_filter(tf.image.resize_area(real_image,(sp*2,sp*2)),Kernel), im64_DPF)
    l12 = compute_error(tf.image.resize_area(real_image,(sp*2,sp*2)), im64)
    l21 = compute_error(pattern_filter(tf.image.resize_area(real_image,(sp*4,sp*4)),Kernel), im128_DPF)
    l22 = compute_error(tf.image.resize_area(real_image,(sp*4,sp*4)), im128)
    l31 = compute_error(pattern_filter(real_image,Kernel), im256_DPF)
    l32 = compute_error(real_image, im256)
    content_loss1 = l11+l21+l31
    content_loss2 = l12+l22+l32
    G_loss=tf.reduce_sum(tf.reduce_mean(content_loss1))*0.3+tf.reduce_sum(tf.reduce_mean(content_loss2))*0.7
#    G_loss=tf.reduce_sum(tf.reduce_min(content_loss,reduction_indices=0))*0.999+tf.reduce_sum(tf.reduce_mean(content_loss,reduction_indices=0))*0.001
lr=tf.placeholder(tf.float32)
G_opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss,var_list=[var for var in tf.trainable_variables() if var.name.startswith('g_')])
saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())
ckpt=tf.train.get_checkpoint_state("result_256p")


if is_training:
    g_loss=np.zeros(8010,dtype=float)
    label_images=[None]*8010
    for epoch in range(1,101):
        if os.path.isdir("result_256p/%04d"%epoch):
            continue
        cnt=0
        for ind in np.random.permutation(8000)+1:
            st=time.time()
            cnt+=1
            if label_images[ind] is None:
                label_images[ind]=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256/%04d.png"%ind)),axis=0)#training image
                label_images[ind]=np.expand_dims(label_images[ind],axis=3)
            _,G_current,loss11,loss12,loss21,loss22,loss31,loss32=sess.run([G_opt,G_loss,l11,l12,l21,l22,l31,l32],feed_dict={label:label_images[ind],real_image:label_images[ind],lr:1e-4})#may try lr:min(1e-6*np.power(1.1,epoch-1),1e-4 if epoch>100 else 1e-3) in case lr:1e-4 is not good
            g_loss[ind]=G_current
            if cnt%10==0:
                print("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f"%(epoch,cnt,np.mean(g_loss[np.where(g_loss)]),np.mean(loss11),np.mean(loss12),np.mean(loss21),np.mean(loss22),np.mean(loss31),np.mean(loss32),time.time()-st))
        os.makedirs("result_256p/%04d"%epoch)
        target=open("result_256p/%04d/score.txt"%epoch,'w')
        target.write("%f"%np.mean(g_loss[np.where(g_loss)]))
        target.close()
        saver.save(sess,"result_256p/model.ckpt")
        if epoch%100==0:
            saver.save(sess,"result_256p/%04d/model.ckpt"%epoch)
        for ind in range(8001,8051):
            if not os.path.isfile("data/DICPC/PC256/%04d.png"%ind):#test label
                continue
            test_image=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256/%04d.png"%ind)),axis=0)#test image
            test_image=np.expand_dims(test_image,axis=3)
            output=sess.run(im256,feed_dict={label:test_image})
            output=np.minimum(np.maximum(output,0.0),255.0)
            scipy.misc.toimage(output[0,:,:,0],cmin=0,cmax=255).save("result_256p/%04d/%06d_output.jpg"%(epoch,ind))

if not os.path.isdir("result_256p/final"):
    os.makedirs("result_256p/final")
for ind in range(8001,8501):
    if not os.path.isfile("data/DICPC/PC256/%04d.png"%ind):#test label
        continue
    test_image=np.expand_dims(np.float32(scipy.misc.imread("data/DICPC/PC256/%04d.png"%ind)),axis=0)#test image
    test_image=np.expand_dims(test_image,axis=3)
    output=sess.run(im256,feed_dict={label:test_image})
    output=np.minimum(np.maximum(output, 0.0), 255.0)
    scipy.misc.toimage(output[0,:,:,0],cmin=0,cmax=255).save("result_256p/final/%06d_output.jpg"%ind)
