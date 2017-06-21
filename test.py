# -*- coding: UTF-8 -*-

import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def recall_error(wanted_set, noise_set, total_num, ratio):
    '''
    总图片数一定，噪声图片按照不同的比例，以50作为半径，计算收上来的图片的召回率和错误率
    :param wanted_set:
    :param noise_set:
    :param total_num: 图片总数目
    :param ratio: 噪声占总数目的比例
    :return:
    '''
    noise_num = int(total_num * ratio)
    wanted_num = int(total_num - noise_num)

    x = np.arange(1000, dtype=np.int8)
    x = np.random.permutation(x)
    samples = np.load(wanted_set)[x[0:wanted_num]]

    x = np.arange(5000, dtype=np.int8)
    x = np.random.permutation(x)
    noise = np.load(noise_set)[x[0:noise_num]]

    mix = np.vstack((samples, noise))

    '''
    pca = PCA(n_components=2)
    pca.fit(mix)
    mix = pca.transform(mix)
    '''

    # 假设买方给予10张样张
    examples = random.sample(range(wanted_num), 10)
    centroid = np.mean(mix[examples], axis=0)

    distance = np.linalg.norm(mix - centroid, axis=1)
    radius = 80
    select_image = np.sum(distance <= radius)
    # print(select_image)
    # np.save('../mechanism/bid_images/selected_features_n07614500.npy', mix[distance <= radius])
    wanted_image = np.sum(distance[0:wanted_num] <= radius)
    unwanted_image = np.sum(distance[wanted_num:total_num] <= radius)

    recall_ratio = float(wanted_image) / wanted_num
    error_ratio = 1 - float(wanted_image) / select_image
    
    return recall_ratio, error_ratio

wanted_set = './features_imagenet_vgg16/twenty sets/features_imagenet_n07614500_vgg16.npy'
noise_set = './features_imagenet_vgg16/features_imagenet_noise_vgg16_5000.npy'

'''
a = np.linspace(0.1, 0.5, 9)
recall_ratio = np.zeros(9)
error_ratio = np.zeros(9)

times = 100
for i in range(times):
    for r in range(9):
        t1, t2 = recall_error(wanted_set, noise_set, 500, a[r])
        recall_ratio[r] += t1
        error_ratio[r] += t2

recall_ratio /= times
error_ratio /= times

plt.plot(a, error_ratio, 'ro-', mec='r', lw=1.5, label="error rate")
plt.plot(a, recall_ratio, 'b*--', mec='b', lw=1.5, label="recall rate")
plt.legend(loc='center left')
plt.xlabel('rate of noise image in total image')
plt.title('The recall and error rate ')
plt.ylim(0, 1.1)
plt.show()

print(recall_ratio)
print(error_ratio)
'''

recall_error(wanted_set, noise_set, 1000, 0.2)