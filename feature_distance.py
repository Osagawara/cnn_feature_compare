import numpy as np
from scipy.cluster.vq import *
import external_index

obs = np.load("./features_imagenet_vgg16/features_imagenet_vgg16.npy")
obs = whiten(obs)
[codebook, distortion] = kmeans(obs, 10)
[code, dist] = vq(obs, codebook)
for i in range(10):
    print(code[(i*100): (i+1)*100])

reference_model = np.zeros(1000)
for i in range(10):
    reference_model[i*100:(i+1)*100] = np.ones(100)*i

index = external_index.external_index(code, reference_model)
print(index)


