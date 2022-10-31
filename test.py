import skimage
import cv2
import torch
import imageio.v2
from utils import detect_multiscale, get_model, non_maximum_suppression_fast
import copy
import numpy as np

def detect(path=None, scale=1.05, th=0, type_='svm', IoU=0.3):
    if path is None:
        path = 'INRIAPerson/Test/pos/crop001514.png'
        # path = 'crop001514(1).png'
    img = imageio.v2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_copy = copy.deepcopy(img)
    svm = cv2.ml.SVM.load('model/svm_data_1.dat')
    net = get_model()
    net.load_state_dict(torch.load('model/net.pt'))

    if type_ =='svm':
        rst = detect_multiscale(img, scale=1.05, clf=svm, type=type_)
        rects = [r for (r,s) in rst if s < th]
        scores = np.array([s for (r,s) in rst if s < th]).flatten()

    else:
        rst = detect_multiscale(img, scale=1.05, clf=net, type=type_)
        rects = [r for (r,s) in rst if s < th]

    for rect in rects:
        x, y, h, w = map(int, rect)
        # print(rect)
        cv2.rectangle(img_copy, (y,x), (y+w,x+h), (0,0,255), 2)

    pick = non_maximum_suppression_fast(rects, IoU)
    
    for rect in pick:
        x, y, h, w = map(int, rect)
        # print(rect)
        cv2.rectangle(img, (x,y), (h,w), (0,0,255), 2)
    
    cv2.imwrite("Before NMS 2.jpg", img_copy)
    cv2.imwrite("After NMS 2.jpg", img)    

scores = detect(path=None, scale=1.05, th=-2.2, type_='svm', IoU=0.5)
# detect(path=None, scale=1.05, th=0.4, type_='l_r', IoU=0.2)


        