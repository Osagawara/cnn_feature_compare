# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = "./features_imagenet_vgg16/"
# 准备数据, 600张图片选自n02113186(corgi), 其余选自noise
wanted_set = './features_imagenet_vgg16/twenty sets/features_imagenet_n03617480_vgg16.npy'
wanted_num = 666
samples = []
samples.append(np.load(wanted_set)[0:(0+wanted_num)])
groundtruth = np.zeros(1000, dtype=np.int8)
groundtruth[0:wanted_num] = 1

noise_set = path + "features_imagenet_noise_vgg16_5000.npy"
samples.append(np.load(noise_set)[2046:(2046+(1000-wanted_num))])
samples = np.vstack(samples)

# 假设买方给予10张样张
examples = random.sample(range(wanted_num), 5)
centroid = np.mean(samples[examples], axis=0)


pca = PCA(n_components=10)
pca.fit(samples)
mix = pca.transform(samples)
print(np.sum(pca.explained_variance_ratio_))

# 假设买方给予10张样张
examples = random.sample(range(wanted_num), 10)
centroid = np.mean(samples[examples], axis=0)


distance = np.linalg.norm(samples-centroid, axis=1)

min_dist = np.min(distance)
max_dist = np.max(distance)

print(min_dist, max_dist)

radius = np.linspace(0, 200, 21)
count_in_radius = np.zeros(21)
right_image = np.zeros(21)
recall_ratio = np.zeros(21)
error_ratio = np.zeros(21)


for i in range(21):
    count_in_radius[i] = np.sum(distance < radius[i])
    right_image[i] = np.sum(groundtruth[distance < radius[i]])
    recall_ratio[i] = right_image[i]/wanted_num
    if(count_in_radius[i] == 0):
        error_ratio[i] = 0
    else:
        error_ratio[i] = 1 - right_image[i]/count_in_radius[i]

count_diff = np.hstack([[0], np.diff(count_in_radius)])
print(count_in_radius)
print('\n')
print(count_diff)
print('\n')
print(right_image)
print('\n')
print(recall_ratio)
print('\n')
print(error_ratio)

plt.subplot(2, 1, 1)
plt.plot(radius, count_in_radius, 'ro-', mec='r', lw=1.5,label="count")
plt.plot(radius, count_diff, 'b*--', mec='b', lw=1.5, label="count diff")
plt.legend(loc='best')

plt.subplot(2, 1, 2)
plt.plot(radius, error_ratio, 'ro-', mec='r', lw=1.5,label="error rate")
plt.plot(radius, recall_ratio, 'b*--', mec='b', lw=1.5, label="recall rate")
plt.legend(loc='best')

plt.show()

