# %%
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.color import label2rgb
from skimage.measure import label

import cellMorphHelper
# %% Full segmentation
imFull = imread('../data/TJ2201/segmentedIms/composite_D2_4_2022y04m08d_08h00m.png')
imSeg = np.load('../data/TJ2201/segmentedIms/composite_D2_4_2022y04m08d_08h00m_seg.npy', allow_pickle=True)
imSeg = imSeg.item()

label_img = label(imSeg['masks'])
imgOverlay = label2rgb(label_img, imFull[:,:,0:3])

resFactor = 1408/1040

fontSize = 20
fig, ax = plt.subplots()
imSizeX = 20
fig.set_size_inches(imSizeX, imSizeX*resFactor)
ax.imshow(imgOverlay)
ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.set_yticks([])
ax.set_xticks([])

fig.savefig('../results/figs/exampleFullSeg.png', dpi=600)

# %% Example segmentations
predictorTJ2201 = cellMorphHelper.getSegmentModel('../output/TJ2201Split16')
predictorAG2021 = cellMorphHelper.getSegmentModel('../output/AG2021Split16')

# %%
cellMorphHelper.viewPredictorResult(predictorTJ2201, '../data/TJ2201Split16/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m_13.png')
cellMorphHelper.viewPredictorResult(predictorAG2021, '../data/TJ2201Split16/phaseContrast/phaseContrast_E2_4_2022y04m07d_16h00m_13.png')
