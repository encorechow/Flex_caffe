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
net = caffe.Net(model_root + 'test_prototxt/testnet.prototxt', model_root + 'model/bsds500_iter_6000.caffemodel', caffe.TEST)
net1 = caffe.Net(model_root + 'test_prototxt/testnet1.prototxt', model_root + 'model/bsds500_iter_6000.caffemodel', caffe.TEST)
for idx in xrange(0, len(image_list)):

    im_data = image_list[idx]
    im_data = im_data.transpose((2,0,1))
    caffe.set_mode_gpu()
    caffe.set_device(0)

    if im_data.shape[2] > im_data.shape[1]:
        net.blobs['data'].reshape(1, *im_data.shape)
        net.blobs['data'].data[...] = im_data
        net.forward()
        out = net.blobs['flexconv1'].data[0][0,:,:]
        np.save('./out_image/' + str(idx) + '.txt', out )
    else:
        net1.blobs['data'].reshape(1, *im_data.shape)
        net1.blobs['data'].data[...] = im_data
        net1.forward()
        out = net1.blobs['flexconv1'].data[0][0,:,:]
        np.save('./out_image/' + str(idx) + '.txt', out )
