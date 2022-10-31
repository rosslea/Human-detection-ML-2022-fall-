from torch import nn
import numpy as np
import math
import skimage
import torch

def get_model():
    net = nn.Sequential(nn.Linear(3780,1))
    return net

def detect_multiscale(img, scale, clf, type='svm', winStride=(2,2)):
    h, w, c = img.shape
    h_copy, w_copy = h, w
    n = math.log10(h/128)/math.log10(scale)
    levels = int(n) if (n-int(n))<1e-1 else int(n)+1
    rst = []
    for level in range(levels): 
        if h>=128 and w>=64:
            img_resized = skimage.transform.resize(img, (int(h), int(w), c), preserve_range=True)
            # print(img_resized.shape)
            output = slide_window(img_resized, clf=clf, type=type, winStride=winStride)
            for (rect, score) in output:
                s = h_copy/h
                rect = (rect[0]*s, rect[1]*s, 128*s, 64*s)
                rst.append((rect, score))     
        if level != (levels-2):
            h, w = h/scale, w/scale
        else:
            h, w = 128, w*128//h
    return rst
    

def slide_window(img, clf, type, winStride=(2,2)):
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
            if type == 'svm':
                score = clf.predict(patch, flags=1)[1]
            elif type == 'l_r':
                patch = torch.tensor(np.expand_dims(patch, axis=0))
                score = clf(patch)
            rst.append((rect,score))
    return rst




























## [credit for NMS https://gist.github.com/JeremyPai/]

def non_maximum_suppression_fast(boxes, overlapThresh=0.3):

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""
        Non-Maximum Suppression (Malisiewicz et al.)
        
        The only difference compared to Felzenszwalb's method
        is that the inner loop is eliminated to speed up the 
        algorithm.
    
        Return value
            boxes[pick]     Remaining bounding boxes
    
        Arguments
            boxes           Detection bounding boxes
            overlapThresh   Overlap threshold for suppression
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    # If there is no bounding box, then return an empty list
    if len(boxes) == 0:
        return []
    # boxes = np.array([list(tp) for tp in boxes])
    boxes = np.array([[y,x,y+h,x+w] for (x,y,w,h) in boxes])    
    # Initialize the list of picked indexes
    pick = []
    
    # Coordinates of bounding boxes
    x1 = boxes[:,0].astype("float")
    y1 = boxes[:,1].astype("float")
    x2 = boxes[:,2].astype("float")
    y2 = boxes[:,3].astype("float")
    
    # Calculate the area of bounding boxes
    bound_area = (x2-x1+1) * (y2-y1+1)
    
    # Sort the bounding boxes by the bottom-right y-coordinate of the bounding box
    sort_index = np.argsort(y2)
    
    # Looping until nothing left in sort_index
    while sort_index.shape[0] > 0:
        # Get the last index of sort_index
        # i.e. the index of bounding box having the biggest y2
        last = sort_index.shape[0]-1
        i = sort_index[last]
        
        # Add the index to the pick list
        pick.append(i)
        
        # Compared to every bounding box in one sitting
        xx1 = np.maximum(x1[i], x1[sort_index[:last]])
        yy1 = np.maximum(y1[i], y1[sort_index[:last]])
        xx2 = np.minimum(x2[i], x2[sort_index[:last]])
        yy2 = np.minimum(y2[i], y2[sort_index[:last]])        

        # Calculate the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Compute the ratio of overlapping
        overlap = (w*h) / bound_area[sort_index[:last]]
        
        # Delete the bounding box with the ratio bigger than overlapThresh
        sort_index = np.delete(sort_index, 
                               np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        
    # return only the bounding boxes in pick list        
    return boxes[pick]
