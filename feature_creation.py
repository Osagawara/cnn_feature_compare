import sys, os
import feature_extraction

path = '../../../imagenet/ILSVRC2012_img_train'

imageSetList = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.tar') and not f.endswith('.txt')]
for i in range(len(imageSetList)):
    feature_extraction.feature_extraction(imageSetList[i], 100)

