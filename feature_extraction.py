import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os
import re


def feature_extraction(path, imagenum):
    """Extract features from images in given path

    :param path: the path holding the images
    :param imagenum: image number that be extracted
    :return: A numpy matrix
             every row is a feature
    """

    caffe_root = '../../'
    sys.path.insert(0, caffe_root + 'python')
    import caffe

    net_file = caffe_root + 'examples/deploy_test/deploy.prototxt'
    caffe_model = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    mean_file = caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'

    # create a list of images
    imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.JPEG')]

    net = caffe.Net(net_file, caffe_model, caffe.TEST)
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))
    transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
    transformer.set_raw_scale('data', 255)
    transformer.set_channel_swap('data', (2, 1, 0))  # if using RGB instead if BGR

    features = np.zeros((imagenum, 1000), np.float32)

    for i in range(imagenum):
        img = caffe.io.load_image(imlist[i])
        net.blobs['data'].data[...] = transformer.preprocess('data', img)

        out = net.forward()
        features[i] = out['fc8'].flatten()

    np.save("features_{}.npy".format(os.path.split(path)[1]), features)
    print("features_{}  OK".format(os.path.split(path)[1]))

    return features
