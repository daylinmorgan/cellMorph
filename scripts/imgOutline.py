# %%
import sys, importlib
# importlib.reload(sys.modules['cellMorphHelper'])
import os
import cv2
import numpy as np
from skimage.io import imread
import matplotlib.pyplot as plt
from cellMorphHelper import getSegmentModel, findFluorescenceColor

from skimage import data, measure
from skimage.segmentation import clear_border

# %%
predictor = getSegmentModel('../output/AG2021Split16')

# %%

# %%

class cellPerims:
    """
    Assigns properties of cells from phase contrast imaging
    """
    def findPerimeter(self):
        c = measure.find_contours(self.mask)
        # assert len(c) == 1, "Error for {}".format(self.composite)
        return c[0]
    
    
    def __init__(self, experiment, imageBase, splitNum, mask):
        try:
            self.experiment = experiment
            self.imageBase = imageBase
            self.splitNum = splitNum
            fname = imageBase+'_'+str(splitNum)+'.jpg'
            self.phaseContrast = os.path.join('../data', experiment, 'phaseContrast','phaseContrast_'+fname)
            self.composite = os.path.join('../data', experiment, 'composite', 'composite_'+fname)
            self.mask = mask

            self.perimeter = self.findPerimeter()
            self.color = findFluorescenceColor(self.composite)
        except:
            print(self.composite)
# Testing out cellperimeter class
experiment = 'AG2021Split16'
imageBase = 'C5_1_2020y06m19d_00h33m'
splitNum = 1

fname = 'phaseContrast_'+imageBase+'_'+str(splitNum)+'.jpg'
imPath = os.path.join('../data',experiment,'phaseContrast',fname)
im = imread(imPath)
outputs = predictor(im)['instances'].to("cpu")
for cellNum in range(len(outputs)):
    mask = outputs[cellNum].pred_masks.numpy()[0]

cell1 = cellPerims(experiment, imageBase, splitNum, mask)
RGB = imread(cell1.composite)
# %%
experiment = 'AG2021Split16'
ims = os.listdir(os.path.join('../data',experiment, 'phaseContrast'))
imbases = ['_'.join(im.split('.')[0].split('_')[1:-1]) for im in ims]
splitNums = [int(im.split('.')[0].split('_')[-1]) for im in ims]

cells = []

for imbase, splitNum in zip(imbases, splitNums):
    print(imbase)
    fname = 'phaseContrast_'+imageBase+'_'+str(splitNum)+'.jpg'
    imPath = os.path.join('../data',experiment,'phaseContrast',fname)
    im = imread(imPath)
    outputs = predictor(im)['instances'].to("cpu")
    nCells = len(outputs)
    for cellNum in range(nCells):
        print('\t Cell {}/{}'.format(cellNum+1, nCells))
        mask = outputs[cellNum].pred_masks.numpy()[0]
        mask = clear_border(mask)
        if np.sum(mask)>10:
            cells.append(cellPerims(experiment, imbase, splitNum, mask))

# %%
RGB = imread("../data/AG2021Split16/composite/composite_C5_1_2020y06m19d_21h33m_2.jpg")
pc = imread("../data/AG2021Split16/phaseContrast/phaseContrast_C5_1_2020y06m19d_21h33m_2.jpg")