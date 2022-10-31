import skimage
import imageio.v2 
from pathlib import Path
import numpy as np
import argparse

from torch import preserve_format
# cv2 : BGR , ndarry
# imageio.v2 : RGB, ndarry

parser = argparse.ArgumentParser()
parser.add_argument('--num_of_parts', required=True)
parser.add_argument('--current_num', required=True)
args = parser.parse_args()
n_parts = int(args.num_of_parts)
cur_part = int(args.current_num)


#------------------------- Set up the training data ---------------------------------
## [collect imgs]
dir = Path.cwd()
img_path_neg = dir/"INRIAPerson/train_64x128_H96/neg_crop.lst"
with open(img_path_neg, 'r') as f:
    n_paths = [i[:-1] for i in f]
img_path_pos = dir/"INRIAPerson/train_64x128_H96/pos_crop.lst"
with open(img_path_pos, 'r') as f:
    p_paths = [i[:-1] for i in f]


def get_cur_lst(lst, n_parts, cur_part):
    print(len(lst))
    step = len(lst)//n_parts
    start_idx = step * cur_part
    if cur_part != n_parts-1:
        new_lst = lst[start_idx:start_idx+step]
    else:
        new_lst = lst[start_idx:]
    return new_lst

n_paths = get_cur_lst(n_paths, n_parts=n_parts, cur_part=cur_part)
p_paths = get_cur_lst(p_paths, n_parts=n_parts, cur_part=cur_part)
## [collect imgs]



## [collect data]
n_features = []
for path in n_paths:
    print('open',path)
    img = imageio.v2.imread(path)
    fd = skimage.feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    n_features.append(fd)
    

p_features = []
for path in p_paths:
    print('open',path)
    img = imageio.v2.imread(path)
    fd = skimage.feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    p_features.append(fd)

n_labels = np.ones(len(n_features))
p_labels = np.ones(len(p_features))*2

features = np.array(n_features+p_features)
features = features.astype(np.float32)
labels = np.concatenate((n_labels, p_labels), axis=0)
labels = labels.astype(int)
labels = labels.reshape((-1,1))
print('Features and labels computed')

np.save(f'data/features_{cur_part}.npy', features)
np.save(f'data/labels_{cur_part}.npy', labels)

print('Features and labels saved')
## [collect data]