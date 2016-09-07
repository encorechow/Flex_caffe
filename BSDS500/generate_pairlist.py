source_path_gt = './data/groundTruthImage/'
source_path_img = './data/images/'

type_img = '.jpg'
type_gt = '.png'

import os

def generate_pair(category):
    xpath = source_path_img + category + '/'
    ypath = source_path_gt + category + '/'

    filename_x = []
    filename_y = []

    for file in os.listdir(xpath):
        if file.endswith(type_img):
            filename_x.append(int(file[0:-4]))
    for file in os.listdir(ypath):
        if file.endswith(type_gt):
            filename_y.append(int(file[0:-4]))

    filename_x.sort()
    filename_y.sort()

    for x, y in zip(filename_x, filename_y):
        if x != y:
            print 'error'
            return
    with open(category + '_pair.lst', 'w') as f:
        for x, y in zip(filename_x, filename_y):
            f.write('{0}{1}{2} {3}{4}{5}\n'.format('.' + xpath, x, type_img, '.' + ypath, y, type_gt))


if __name__ == "__main__":
    generate_pair('train')
    generate_pair('test')
