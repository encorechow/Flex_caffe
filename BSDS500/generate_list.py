source_path_gt = './data/groundTruthImage/'
source_path_img = './data/images/'

type_img = '.jpg'
type_gt = '.png'

import os

def generate_pair(category):
    xpath = source_path_img + category + '/'

    filename_x = []

    for file in os.listdir(xpath):
        if file.endswith(type_img):
            filename_x.append(int(file[0:-4]))

    filename_x.sort()

    with open(category + '.lst', 'w') as f:
        for x in filename_x:
            f.write('{0}{1}{2}\n'.format('.' + xpath, x, type_img))


if __name__ == "__main__":
    generate_pair('train')
    generate_pair('test')
