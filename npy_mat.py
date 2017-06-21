import numpy as np
import scipy.io
import os

npy_path = './features_imagenet_vgg16/'
npylist = [os.path.join(npy_path, f) for f in os.listdir(npy_path) if f.endswith('.npy')]

mat_path = '/Users/xiaoxiang/Documents/MATLAB/mechanism/features_imagenet_vgg16/'
for i in npylist:
    mat_name = os.path.split(i)[1].split('.')[0]
    temp = np.load(i)
    scipy.io.savemat(mat_path+mat_name, {'noises':temp})
