import numpy as np
import tensorflow as tf

import vgg16
import utils

import os






# create a list of images
path = './test_data/'
imlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

image_array = [utils.load_image(img_path).reshape((1, 224, 224, 3)) for img_path in imlist]
#for img_path in imlist:
    #print(img_path)
    #utils.load_image(img_path).reshape((1, 224, 224, 3))
    #print("success")

#img1 = utils.load_image("./test_data/tiger.jpeg")
#img2 = utils.load_image("./test_data/puzzle.jpeg")

#batch1 = img1.reshape((1, 224, 224, 3))
#batch2 = img2.reshape((1, 224, 224, 3))

#batch = np.concatenate((batch1, batch2), 0)

batch = np.concatenate(image_array, 0)

# with tf.Session(config=tf.ConfigProto(gpu_options=(tf.GPUOptions(per_process_gpu_memory_fraction=0.7)))) as sess:
with tf.device('/cpu:0'):
    with tf.Session() as sess:
        images = tf.placeholder("float", [len(image_array), 224, 224, 3])
        feed_dict = {images: batch}

        vgg = vgg16.Vgg16()
        with tf.name_scope("content_vgg"):
            vgg.build(images)

        prob = sess.run(vgg.prob, feed_dict=feed_dict)
        fc8 = sess.run(vgg.fc8, feed_dict=feed_dict)
       
       #utils.print_prob(prob[0], './synset.txt')
       #utils.print_prob(prob[1], './synset.txt')





