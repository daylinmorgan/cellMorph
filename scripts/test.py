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
phaseContrastPath = '../data/TJ2201/phaseContrast'
maskPath = '../data/TJ2201/mask'

phaseContrastFiles = os.listdir(phaseContrastPath)
phaseFile = phaseContrastFiles[4]
imgBase = cellMorphHelper.getImageBase(phaseFile)
maskFile = 'mask_'+imgBase+'.tif'

mask = imread(maskPath+'/'+maskFile)
img = imread(phaseContrastPath+'/'+phaseFile)


label_image = label(mask)
overlay = label2rgb(label_image, image=img,bg_label=0)

fig, ax = plt.subplots()
fig.set_size_inches(20, 20)
ax.imshow(overlay)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_yticks([])
ax.set_xticks([])

# fig.savefig('../output/sampleSegmentation.png')
# %%
imgSplit = cellMorphHelper.imSplit(img)
# %%
predictor = cellMorphHelper.getSegmentModel('../output/AG2021Split16')
# %%
imNum = 6
imgPath = '../data/TJ2201Split16/phaseContrast/'+'phaseContrast_'+imgBase+f'_{imNum}'+'.png'
# plt.imshow(imread(imgPath))
cellMorphHelper.viewPredictorResult(predictor, imgPath)