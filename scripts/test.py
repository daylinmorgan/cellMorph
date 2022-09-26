# %%
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.color import label2rgb

from matplotlib import pyplot as plt
import cellMorphHelper
import os

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils.visualizer import ColorMode

# %%
'../data/TJ2201/mask/', ims[3]

# %%
ims = os.listdir('../data/TJ2201/mask')
im = imread('../data/TJ2201/mask/'+ims[3])
plt.figure(figsize=(20,20))
plt.imshow(im)
