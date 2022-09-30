# %%
import importlib
import random
import sys

import cellMorphHelper
import matplotlib.pyplot as plt
from cellMorphHelper import getSegmentModel, splitExpIms, viewPredictorResult
from skimage.io import imread

# importlib.reload(sys.modules['cellMorphHelper'])


# %%
predictor = cellMorphHelper.getSegmentModel("../output/AG2021Split16")
# %% Split Experiment Images
experiment = "TJ2201"
nIms = 16
# splitExpIms(experiment, nIms)
# %% View segmentations
pcPath = os.path.join("../data/" + experiment + "Split" + str(nIms), "phaseContrast")
ims = os.listdir(pcPath)

random.seed(1234)
for imNum in random.sample(range(len(ims)), 3):
    imPath = os.path.join(pcPath, ims[imNum])
    cellMorphHelper.viewPredictorResult(predictor, imPath)
# %%
cellMorphHelper.viewPredictorResult(
    predictor,
    "../data/TJ2201Split16/phaseContrast/phaseContrast_D2_1_2022y04m09d_04h00m_14.png",
)
