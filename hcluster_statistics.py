# -*- coding: utf-8 -*-


import numpy as np
from scipy.cluster import hierarchy

def accumulate_images(Z, cut_threshold, cum_threshold, groundtruth):
    '''

    :param Z: (n-1)*4 linkage matrix
    :param cut_threshold: 切割聚合树的阈值
    :param cum_threshold: 将聚类按从大到小排列, 分别选取1, 2, ..., cum_threshold个聚类，计算错误率
    :param groundtruth: 0-1向量, 标记图片是否是需要的图片
    :return: 两个len(cut_threshold)*cum_threshold矩阵cluster_size, error_num
             cluster_size(i, j)表示以cut_threshold[i]分割得到的第三个聚簇的size
             error_num(i, j)表示对应的簇中错误的图片数
    '''


    max_dist = np.max(Z[:, 2])
    cluster_size = np.zeros((len(cut_threshold), cum_threshold))
    error_num = np.zeros((len(cut_threshold), cum_threshold))
    for i in range(len(cut_threshold)):
        fcluster = hierarchy.fcluster(Z, cut_threshold[i]*max_dist, criterion='distance')
        hcluster_size = np.bincount(fcluster)

        # sort the hcluster_size in descent order
        sorted_size = np.sort(hcluster_size)[: : -1]
        sorted_indices = np.argsort(hcluster_size)[: : -1]

        temp = np.min((cum_threshold, len(sorted_size)))
        for j in range(temp):
            cluster_size[i][j] = sorted_size[j]
            selected_images = fcluster == sorted_indices[j]
            error_num[i][j] = np.sum(selected_images)-np.sum(groundtruth[selected_images])

    return cluster_size, error_num


def select_images(samples, cut_threshold, size_threshold, groundtruth):
    '''

    :param samples: n*m array, 每一行代表图片的特征
    :param cut_threshold: 切割聚合数的阈值
    :param size_threshold: 将规模大于size的集合选中
    :param groundtruth: 0-1向量, 标记图片是否是需要的图片
    :return: 挑选的图片标记向量
    '''

    Z = hierarchy.linkage(samples, method='average', metric='euclidean')
    max_dist = np.max(Z[:, 2])
    fcluster = hierarchy.fcluster(Z, cut_threshold * max_dist, criterion='distance')
    hcluster_size = np.bincount(fcluster)


    selected_image = np.zeros(len(groundtruth), dtype=np.bool)
    for i in range(len(hcluster_size)):
        if(hcluster_size[i] > size_threshold):
            selected_image[fcluster == i] = True

    return selected_image


def two_stage_accumulate(samples, first_cut, second_cut, first_size, second_size, groundtruth):
    '''

    :param samples: 收集的图片对应的特征
    :param first_cut: 第一次筛选时的阈值
    :param second_cut: 第二次筛选时的阈值
    :param
    :param groundtruth: 0-1向量, 标记图片是否是需要的图片
    :return: 两次筛选收集的图片数量和错误的图片数量
    '''

    first_select = select_images(samples, first_cut, first_size, groundtruth)
    remain = np.arange(len(groundtruth), dtype=np.int)[-first_select]
    second_select = select_images(samples[remain], second_cut, second_size, groundtruth[remain])

    first_collection = np.sum(first_select)
    second_collection = np.sum(second_select)
    first_error = first_collection - np.sum(groundtruth[first_select])
    second_error = second_collection - np.sum(groundtruth[remain[second_select]])

    return first_collection, second_collection, first_error, second_error


