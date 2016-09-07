import numpy as np
import scipy.misc
import sys


temp = np.load(sys.argv[1] + '.txt.npy')
scipy.misc.imsave(sys.argv[1] + '.png',  temp)
