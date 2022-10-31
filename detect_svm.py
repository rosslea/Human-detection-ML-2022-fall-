import imageio
import numpy as np
import skimage
import math
import cv2
from tqdm import tqdm
import copy
from pathlib import Path
import torchvision.transforms as T
import torch
import time


def detect_multiscale(img, scale, svm, hitThreshold, winStride=(2,2)):
    h, w, c = img.shape
    h_copy, w_copy = h, w
    n = math.log10(h/128)/math.log10(scale)
    levels = int(n) if (n-int(n))<1e-1 else int(n)+1
    rst = []
    for level in range(levels): 
        if h>=128 and w>=64:
            img_resized = skimage.transform.resize(img, (int(h), int(w), c), preserve_range=True)
            # print(img_resized.shape)
            output = slide_window(img_resized, svm=svm, hitThreshold=hitThreshold, winStride=winStride)
            for (rect, score) in output:
                s = h_copy/h
                rect = (rect[0]*s, rect[1]*s, 128*s, 64*s)
                rst.append((rect, score))     
        if level != (levels-2):
            h, w = h/scale, w/scale
        else:
            h, w = 128, w*128//h
    return rst
    

def slide_window(img, svm, hitThreshold=0, winStride=(2,2)):
    h, w, c = img.shape
    cell_group = h//8, w//8
    block_group = cell_group[0] - 1, cell_group[1] - 1
    fd = skimage.feature.hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False, channel_axis=-1)
    feature = fd.astype(np.float32)[np.newaxis].reshape((block_group[0], block_group[1], 36))
    # hog : block 15*7, cell 16*8
    rst = []
    for i in range(0, block_group[0]-14, winStride[0]):
        # print('slide in h direction i=', i)
        for j in range(0, block_group[1]-6, winStride[1]):
            patch = feature[i:i+15,j:j+7]
            rect = (i*8,j*8) # upper left pixel corrd in resized img
            patch = np.expand_dims(patch.flatten(), axis=0)
            score = svm.predict(patch, flags=1)[1]
            rst.append((rect,score))
    return rst


print('detecting')

def detect_neg():
    print('--------------- detecting neg ---------------')
    n_lst = 'INRIAPerson/test_64x128_H96/neg.lst'
    with open(n_lst) as f:
        n_paths = [Path('INRIAPerson')/str(i[:-1]).capitalize() for i in f] 
    
    scores = []
    for path in n_paths[:]:
        print('open', path)
        img = imageio.v2.imread(path)
        svm = cv2.ml.SVM.load('model/svm_data_1.dat')
        rects = detect_multiscale(img, scale=1.05, svm=svm, hitThreshold=0, winStride=(2, 2))

        for (rect, score) in rects:
            scores.append(score)
            # x, y, h, w = map(int, rect)
            # cv2.rectangle(img_copy, (y,x), (y+w,x+h), (0,0,255), 2)
            # cv2.imwrite('predict.png', img_copy)
    return scores

def detect_pos():
    print('--------------- detecting pos ---------------')
    p_lst = 'INRIAPerson/test_64x128_H96/pos.lst'
    with open(p_lst) as f:
        p_paths = [Path('INRIAPerson')/i[:-1] for i in f]
    
    for i in range(len(p_paths)):
        if not p_paths[i].exists():   
            p_paths[i] = Path("INRIAPerson/70X134H96/Test/pos")/p_paths[i].name
            assert p_paths[i].exists(), f"{p_paths[i]} " + 'not exist'
    
    cropper = T.CenterCrop(size=(128, 64))
    scores = []
    for path in p_paths[:]:
        print('open', path)
        img = imageio.v2.imread(path)
        data = torch.tensor(img)
        img = cropper(data).numpy()
        svm = cv2.ml.SVM.load('model/svm_data_1.dat')
        rects = detect_multiscale(img, scale=1.05, svm=svm, hitThreshold=0, winStride=(2, 2))

        for (rect, score) in rects:
            scores.append(score)
            # x, y, h, w = map(int, rect)
            # cv2.rectangle(img_copy, (y,x), (y+w,x+h), (0,0,255), 2)
            # cv2.imwrite('predict.png', img_copy)
    return scores

neg_scores = np.array(detect_neg())
pos_scores = np.array(detect_pos())

np.save('data/neg_scores.npy', neg_scores)
np.save('data/pos_scores.npy', pos_scores)


print("scores saved")
