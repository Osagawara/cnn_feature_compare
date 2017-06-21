# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import matplotlib.pyplot as plt

from scipy.cluster import hierarchy
from hcluster_statistics import *

path = "./features_imagenet_vgg16/"
# 准备数据, 600张图片选自n02113186(corgi), 其余选自noise
target_set = path + "tweenty sets/features_imagenet_n02687172_vgg16.npy"
print(target_set)
target_num = 200
samples = []
samples.append(np.load(target_set)[0:(0+target_num)])
groundtruth = np.zeros(1000, dtype=np.int8)
groundtruth[0:target_num] = 1

noise_set = path + "features_imagenet_noise_vgg16_5000.npy"
samples.append(np.load(noise_set)[2217:(2217+(1000-target_num))])

samples = np.vstack(samples)

Z = hierarchy.linkage(samples, method='average', metric='euclidean')

cut_threshold = np.linspace(0.3, 0.8, 11)
cum_threshold = 5
cluster_size, error_num = accumulate_images(Z, cut_threshold, cum_threshold, groundtruth)


X = np.arange(len(cut_threshold))
wanted = cluster_size - error_num
xtick = [str(i) for i in cut_threshold]

# the number of wanted image and error image in the first cluster
plt.subplot(2, 2, 1)
plt.bar(X-0.4, wanted[:, 0], label='wanted')
plt.bar(X-0.4, error_num[:, 0], bottom=wanted[:, 0], label='error', color='r')
plt.xticks(X, xtick)
plt.xlabel('Threshold')
plt.legend(loc='upper left')

'''
# the number of wanted image and error image in the second cluster
plt.subplot(2, 2, 2)
plt.bar(X-0.4, wanted[:, 0]+wanted[:, 1], label='wanted')
plt.bar(X-0.4, error_num[:, 0]+error_num[:, 1], bottom=wanted[:, 0]+wanted[:, 1], label='error', color='r')
plt.xticks(X, xtick)
plt.xlabel('Threshold')
plt.legend(loc='best')
'''

# the difference of the size of the biggest cluster at different threshold
plt.subplot(2, 2, 2)
plt.plot(cut_threshold[1:11], np.diff(cluster_size[:, 0]), "b*--", mec="b")

# the error rate and recall rate of the biggest cluster
plt.subplot(2, 2, 3)
error_rate = error_num[:, 0] / cluster_size[:, 0]
recall_rate = (cluster_size[:, 0] - error_num[:, 0]) / target_num
plt.plot(cut_threshold, error_rate, 'ro-', mec='r', label="error rate")
plt.plot(cut_threshold, recall_rate, 'b*--', mec='b', label="recall rate")
plt.legend(loc='best')

'''
# the error-rate and recall rate of the biggest and second biggest
plt.subplot(2, 2, 4)
error_rate = np.cumsum(error_num, axis=1)[:, 1] / np.cumsum(cluster_size, axis=1)[:, 1]
call_back_rate = np.cumsum(cluster_size - error_num, axis=1)[:, 1] / target_num
plt.plot(cut_threshold, error_rate, 'ro-', label = "error rate")
plt.plot(cut_threshold, recall_rate, 'b*--', label = "recall rate")
plt.legend(loc = 'best')
'''

# the difference of the error rate and recall rate
plt.subplot(2, 2, 4)
plt.plot(cut_threshold[1:11], np.diff(error_rate), 'ro-', mec='r', label="error rate diff")
plt.plot(cut_threshold[1:11], np.diff(recall_rate), 'b*--', mec='b', label="recall rate diff")
plt.legend(loc='best')

plt.show()


'''
first_collection = np.zeros((len(cut_threshold), len(cut_threshold)))
second_collection = np.zeros((len(cut_threshold), len(cut_threshold)))
first_error = np.zeros((len(cut_threshold), len(cut_threshold)))
second_error = np.zeros((len(cut_threshold), len(cut_threshold)))

for i in range(len(cut_threshold)):
    for j in range(len(cut_threshold)):
        first_collection[i][j], second_collection[i][j], first_error[i][j], second_error[i][j] = \
            two_stage_accumulate(samples, \
                                 first_cut=cut_threshold[i], \
                                 second_cut=cut_threshold[j],\
                                 first_size=10, second_size=10, \
                                 groundtruth=groundtruth)

error_rate = (first_error+second_error)/(first_collection+second_collection)
callback_rate = (first_collection+second_collection - first_error - second_error)/target_num

print(error_rate)
print(callback_rate)
print("\n")

callback_threshold = [0.7, 0.75, 0.8, 0.85, 0.9]
best_ratio = np.zeros(len(callback_threshold))
for i in range(len(callback_threshold)):
    temp = np.min(error_rate[callback_rate > callback_threshold[i]])
    re = np.where((error_rate == temp).reshape(error_rate.shape))
    print(re)
    best_ratio[i] = first_collection[re[0][0]][re[1][0]]/second_collection[re[0][0]][re[1][0]]

print("\n")
print(best_ratio)

'''



