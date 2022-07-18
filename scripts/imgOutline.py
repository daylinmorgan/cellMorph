# %%
import os
import cv2
import matplotlib.pyplot as plt
from cellMorphHelper import getSegmentModel

from skimage import data, measure

# %%
predictor = getSegmentModel('../output/AG2021Split16')

# %%
allPerims = {}
imPath = '../data/AG2021Split16/phaseContrast'
testIm = 'phaseContrast_C5_1_2020y06m19d_00h33m'

imNum = 6

imPathFull = os.path.join(imPath, testIm+'_'+str(imNum)+'.jpg')

im = cv2.imread(imPathFull)

outputs = predictor(im)['instances'].to("cpu")

for cellNum in range(len(outputs)):
    mask = outputs[cellNum].pred_masks.numpy()[0]
# %%
coords = measure.regionprops_table(measure.label(mask), properties = ['coords'])['coords'][0]


# %%

class cellPerims:
    """
    Assigns properties of cells from phase contrast imaging
    """

    def __init__(self, experiment, imageBase, splitNum, mask):
        self.experiment = experiment
        self.imageBase = imageBase
        self.splitNum = splitNum
        fname = imageBase+'_'+str(splitNum)+'.jpg'
        self.phaseContrast = os.path.join('../data', experiment, 'phaseContrast_'+fname)
        self.composite = os.path.join('../data', experiment, 'composite'+fname)
        self.mask = mask

    def findPerimeter(self):
        c = measure.find_contours(mask)
        assert len(c) == 1
# %%
# phaseContrast_C5_1_2020y06m19d_00h33m_2.jpg