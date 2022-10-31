import skimage
import imageio.v2 
import cv2
from pathlib import Path
import numpy as np
import math
import argparse

from torch import feature_dropout, preserve_format
# cv2 : BGR , ndarry
# imageio.v2 : RGB, ndarry

features = np.load('data/features_0.npy')
labels = np.load('data/labels_0.npy')

#------------------------ Train the svm ----------------------------------------------------
## [init]
print('Starting training process')
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setC(0.1)
svm.setKernel(cv2.ml.SVM_LINEAR)
svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, int(1e7), 1e-6))
## [init]

## [train]
svm.train(features, cv2.ml.ROW_SAMPLE, labels)
print('Finished training process')
## [train]

svm.save('model/svm_data_1.dat')
print('Svm saved in file svm_data_1.dat')