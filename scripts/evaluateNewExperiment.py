# %%
import sys, importlib
importlib.reload(sys.modules['cellMorphHelper'])

from cellMorphHelper import splitExpIms, getSegmentModel, viewPredictorResult
import random

from skimage.io import imread

import matplotlib.pyplot as plt
# %%
predictor = getSegmentModel('../output/AG2021Split16')
# %% Split Experiment Images
experiment = 'TJ2201'
nIms = 16
# splitExpIms(experiment, nIms)
# %% View segmentations
pcPath = os.path.join('../data/'+experiment+'Split'+str(nIms), 'phaseContrast')
ims = os.listdir(pcPath)

for imNum in random.sample(range(len(ims)), 3):
    imPath = os.path.join(pcPath, ims[imNum])
    viewPredictorResult(predictor, imPath)