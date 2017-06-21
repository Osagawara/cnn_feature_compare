# -*- coding: utf-8 -*-

# show the difference of the change points of wanted class and the noise
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

wanted_set = './features_imagenet_vgg16/twenty sets/features_imagenet_n02114367_vgg16.npy'
wanted_num = 500
x = np.arange(1000, dtype=np.int16)
x = np.random.permutation(x)
samples = np.load(wanted_set)[x[0:wanted_num]]

noise_set = "./features_imagenet_vgg16/features_imagenet_noise_vgg16_5000.npy"
noise_num = 500
x = np.arange(5000, dtype=np.int16)
x = np.random.permutation(x)
noise = np.load(noise_set)[x[0:noise_num]]

'''
# 假设买方给予10张样张
examples = random.sample(range(wanted_num), 10)
centroid = np.mean(samples[examples], axis=0)
'''

mix = np.vstack((samples, noise))

'''
pca = PCA(n_components=2)
pca.fit(mix)
mix = pca.transform(mix)
print(np.sum(pca.explained_variance_ratio_))
'''

# 假设买方给予10张样张
examples = random.sample(range(wanted_num), 10)
centroid = np.mean(mix[examples], axis=0)
# print(centroid)

distance = np.linalg.norm(mix - centroid, axis=1)
print(np.max(distance), np.min(distance))
radius = np.linspace(0, 200, 21)

sample_distance = distance[0:wanted_num]
noise_distance = distance[wanted_num:(wanted_num+noise_num)]

sample_count = np.zeros(21)
noise_count = np.zeros(21)
for i in range(21):
    sample_count[i] = np.sum(sample_distance < radius[i])
    noise_count[i] = np.sum(noise_distance < radius[i])

plt.plot(radius, noise_count, 'ro-', mec='r', lw=1.5, label="noise")
plt.plot(radius, sample_count, 'b*--', mec='b', lw=1.5, label="wanted")
plt.legend(loc='best')
plt.xlabel('R')
plt.ylabel('count')


plt.show()

