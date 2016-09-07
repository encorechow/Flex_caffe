from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.misc
import scipy.io
import sys
import os

source_root = "../data/images/train/"
ground_truth = "../data/groundTruthImage/train/"
#
# for filename in os.listdir(source_root):
#     if filename[-4:] == '.jpg':
#         im = Image.open(source_root+filename)
#         im_data = np.array(im, dtype=np.float32)
#         if im_data.shape[0] > im_data.shape[1]:
#             im_data = im_data.transpose([1, 0, 2])
#         scipy.misc.imsave(source_root + filename, im_data)

for filename in os.listdir(ground_truth):
    if filename[-4:] == '.png':
        im = Image.open(ground_truth+filename)
        im_data = np.array(im, dtype=np.float32)
        
        if im_data.shape[0] > im_data.shape[1]:
            im_data = im_data.transpose([1, 0])
        scipy.misc.imsave(ground_truth + filename, im_data)


#plt.figure()
#plt.imshow(im_data)
#plt.show()
