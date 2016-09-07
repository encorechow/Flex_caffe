import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import matplotlib.cm as cm
import scipy.misc
from PIL import Image
import scipy.io
import os

caffe_root = "../"
data_root = "./data/"

import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

with open('test.lst') as f:
    test_list = f.readlines()

image_list = []


for s in test_list:
    s = s.strip()
    im = Image.open(s)
    im_data = np.array(im, dtype=np.float32)
    im_data = im_data[:,:,::-1]
    mean = np.array((104.00698793,116.66876762,122.67891434))
    im_data -= mean
    image_list.append(im_data)

model_root = './'
net = caffe.Net(model_root + 'test_prototxt/testnet_8_28_2block.prototxt', model_root + 'model/bsds500_new_2block2_iter_4000.caffemodel', caffe.TEST)
net1 = caffe.Net(model_root + 'test_prototxt/testnet_8_28_reverse_2block.prototxt', model_root + 'model/bsds500_new_2block2_iter_4000.caffemodel', caffe.TEST)
idx = 4

im_data = image_list[idx]
im_data = im_data.transpose((2,0,1))
caffe.set_mode_gpu()
caffe.set_device(0)

if im_data.shape[2] > im_data.shape[1]:
    net.blobs['data'].reshape(1, *im_data.shape)
    net.blobs['data'].data[...] = im_data
    net.forward()
    out = net.blobs['weightconv2'].data[0][0,:,:]
    print(out)
    #out = net.blobs['sum4'].data[0][0,:,:]
    plt.figure("aaa")
    #plt.imshow(out,cmap = cm.Greys_r)
    # plt.subplot(121)
    # plt.imshow(out, cmap = cm.Greys_r)
    # plt.subplot(122)
    plt.imshow(out, cmap = cm.Greys_r)
    plt.show()
    # np.save('./single_test/' + str(idx) + '.txt', out )
else:
    net1.blobs['data'].reshape(1, *im_data.shape)
    net1.blobs['data'].data[...] = im_data
    net1.forward("bbb")
    out = net1.blobs['weightconv3'].data[0][0,:,:]
    #out = net1.blobs['sum4'].data[0][0,:,:]


    plt.figure()
    #plt.imshow(out,cmap = cm.Greys_r)
    # plt.subplot(121)
    # plt.imshow(out, cmap = cm.Greys_r)
    # plt.subplot(122)
    plt.imshow(out, cmap = cm.Greys_r)
    plt.show()
    # np.save('./single_test/' + str(idx) + '.txt', out )
