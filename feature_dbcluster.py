# -*- coding: UTF-8 -*-


import numpy as np
from sklearn.cluster import DBSCAN
import scipy.spatial
import scipy.cluster.vq

features = np.load("./features_imagenet_vgg16/features_imagenet_vgg16.npy")

dbcluster = DBSCAN(eps=70, min_samples=108)

# 100 images of tiger shark and 5 images of electric ray
temp = np.vstack((features[0:100], features[400:500]))
predict = dbcluster.fit_predict(temp)

print(predict)






