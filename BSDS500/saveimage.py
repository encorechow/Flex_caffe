import numpy as np
import scipy.misc


root = './out_image/'
target = './output/'
for i in xrange(0, 200):
     temp = np.load(root + str(i) + '.txt.npy')
     scipy.misc.imsave(target + str(i) + '.png',  temp)
