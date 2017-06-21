# -*- coding: utf-8 -*-

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

path = "./features_imagenet_vgg16/"
wanted_set = './features_imagenet_vgg16/twenty sets/features_imagenet_n01518878_vgg16.npy'
wanted_num = 400
samples = []
samples.append(np.load(wanted_set)[200:(200+wanted_num)])

noise_set = path + "features_imagenet_noise_vgg16_5000.npy"
samples.append(np.load(noise_set)[2046:(2046+(500-wanted_num))])
samples = np.vstack(samples)

pca = PCA(n_components=8)
pca.fit(samples)
print(np.sum(pca.explained_variance_ratio_))
a = pca.transform(samples)

plt.scatter(a[0:wanted_num, 0], a[0:wanted_num, 1], c='b', marker='*', edgecolors='none')
plt.scatter(a[wanted_num:500, 0], a[wanted_num:500, 1], c='r', marker='+')
plt.show()
