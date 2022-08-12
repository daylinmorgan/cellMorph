# %% [markdown]
"""
Holds necessary structural information for storing cell information
"""
# %%
import numpy as np
import os
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.measure import find_contours

import cellMorphHelper
# %%
class cellPerims:
    """
    Assigns properties of cells from phase contrast imaging
    """    
    def __init__(self, experiment, imageBase, splitNum, mask):
        self.experiment = experiment
        self.imageBase = imageBase
        self.splitNum = splitNum
        fname = imageBase+'.png'
        self.phaseContrast = os.path.join('../data', experiment, 'phaseContrast','phaseContrast_'+fname)
        self.composite = os.path.join('../data', experiment, 'composite', 'composite_'+fname)
        self.mask = mask

        self.perimeter = self.findPerimeter()

        self.color = cellMorphHelper.findFluorescenceColor(self.composite, self.mask)

        self.perimAligned = ''
        self.perimInt = cellMorphHelper.interpolatePerimeter(self.perimeter)

    def imshow(self):
        RGB = imread(self.composite)
        mask = self.mask
        RGB[~np.dstack((mask,mask,mask))] = 0
        plt.figure()
        plt.imshow(RGB)
        plt.plot(self.perimeter[:,1], self.perimeter[:,0])
        plt.title(self.color)

    def findPerimeter(self):
        c = find_contours(self.mask)
        # assert len(c) == 1, "Error for {}".format(self.composite)
        return c[0]
    

# If something bad happened where you need to pickle a new object, fix it with this:
# for cell in cells:
#     cell.__class__ = eval(cell.__class__.__name__)`