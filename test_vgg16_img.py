# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

import vgg16
import utils

import os
import re


def batch_single_set(path, image_num):
    '''

    :param path: path of image set
    :param image_num: number of images extracted
    :return: features
    '''

    image_array = []
    imlist = list(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPEG'))
    image_count = 0
    i = 0
    while image_count < image_num:
        a = utils.load_image(imlist[i])
        i += 1
        if len(a.shape) == 3:
            image_array.append(a.reshape(1, 224, 224, 3))
            image_count += 1

    batch = np.concatenate(image_array, 0)
    return batch


# create a list of images
# imageset_list has ten elements, which are file directories
# image_list is the list of the image paths

imageset_path = '/raid/workspace/wangjunjun/imagenet/ILSVRC2012_img_train'
imageset_list = [os.path.join(imageset_path, f) for f in os.listdir(imageset_path) if not f.endswith('.tar') and not f.endswith('.txt')]
image_num_per_set = 1000


'''
for path in imageset_list:
    print(path)
    imlist = list(os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPEG'))
    image_count = 0
    i = 0
    while image_count < image_num_per_set:
        a = utils.load_image(imlist[i])
        i += 1
        if len(a.shape) == 3:
            image_array.append(a.reshape(1, 224, 224, 3))
            image_count += 1
            image_list.append(imlist[i])

'''

'''
# output the selected image name
with open('./selected_image.txt', 'w') as f:
    f.write('\n'.join(image_list) + '\n')

batch = np.concatenate(image_array, 0)

print("batch size: {}".format(batch.shape))
'''

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
# tensorflow input batch shouldn't be too large, otherwise it will terminate with error

sub_batch = 50


with tf.device('/gpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [sub_batch, 224, 224, 3])

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        for path in imageset_list:
            batch = batch_single_set(path, image_num_per_set)
            features = []
            for i in range(batch.shape[0] / sub_batch):
                feed_dict = {images: batch[(sub_batch * i): (sub_batch * (i + 1))]}
                fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
                features.append(fc8)
            features = np.concatenate(features, 0)
            npy_path = "features_imagenet_" + re.split('/', path)[-1] + "_vgg16.npy"
            np.save(npy_path, features)
            print(npy_path + "  OK")


'''
# create features from open-image
imageset_path = '../open-images/images/truss_bridge/0'
imlist = [os.path.join(imageset_path, f) for f in os.listdir(imageset_path) ]
with tf.device('/gpu:2'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [sub_batch, 224, 224, 3])

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        image_array = []
        image_count = 0
        i = 0
        while i < len(imlist):
            try:
                a = utils.load_image(imlist[i])
                # i += 1
                if len(a.shape) == 3 and a.shape[2] == 3:
                    image_array.append(a.reshape(1, 224, 224, 3))
                    image_count += 1
                else:
                    print("image {}: ".format(i) + imlist[i])
            except IOError:
                print("image {}: ".format(i) + imlist[i])
            except TypeError:
                print("image {}: ".format(i) + imlist[i])
            except ValueError:
                print("image {}: ".format(i) + imlist[i])
            finally:
                i += 1


        batch = np.concatenate(image_array, 0)
        features = []
        for i in range(batch.shape[0] / sub_batch):
            feed_dict = {images: batch[(sub_batch * i): (sub_batch * (i + 1))]}
            fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
            features.append(fc8)
            print("batch {} OK".format(i))
        features = np.concatenate(features, 0)
        npy_path = "features_openimages_truss_bridge_vgg16_{}.npy".format((i+1)*sub_batch)
        np.save(npy_path, features)
        print(npy_path + "  OK")
'''
