# %%
import os

import cellMorphHelper
import pandas as pd
from matplotlib import pyplot as plt
from skimage.color import label2rgb
from skimage.io import imread, imsave
from skimage.measure import label

# %%
"../data/TJ2201/mask/", ims[3]

# %%
ims = os.listdir("../data/TJ2201/mask")
im = imread("../data/TJ2201/mask/" + ims[3])
plt.figure(figsize=(20, 20))
plt.imshow(im)
# %%
fileDir = "../data/AG2021/label/train"
ncells = 0
for file in os.listdir(fileDir):
    labels = pd.read_csv(os.path.join(fileDir, file))
    ncells += labels.shape[0]
