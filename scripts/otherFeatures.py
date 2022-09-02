# %% [markdown]

"""
# Cell Feature Extraction

This notebook is a testbed for extracting features from cells using pyfeats
"""
# %%
# import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np
import datetime
import pickle

from cellMorph import cellPerims
# from cellMorphHelper import extractFeatures
import sys, importlib
# importlib.reload(sys.modules['cellMorphHelper'])
import cellMorphHelper

from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.io import imread
import cv2

import pyfeats
# %%
cells = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-D2.pickle',"rb"))
cells = cellMorphHelper.filterCells(cells, confluencyDate=datetime.datetime(2022, 4, 8, 16, 0), color='green', edge=True)
# %% Pyfeatures test
# Take subset of cells
maxCells = 400
random.seed(1234)
cellSample = np.array(cells)[random.sample(range(len(cells)), 5000)]
cellSample = [cell for cell in cellSample if cell.color != 'NaN']
allFeatures, allLabels = [], []
y = []
c = 1
# %%
cell = cellSample[0]
colors = []
allFeatures, allLabels = [], []

cellNum = 1
for cell in cellSample:
    f = rgb2gray(imread(cell.phaseContrast))
    mask = cell.mask
    a = np.where(mask==True)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    print(f'{cellNum}/{len(cellSample)}')
    f = f[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    # Stretches values to be in between 0 and 255
    # NOTE: Image must initial be bound between 0 and 1
    f = f*255 
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(int)
    # Take only mask
    image = np.multiply(f, mask)
    perimeter = cell.perimeter
    # extractedFeatures, extractedLabels = cellMorphHelper.extractFeatures(image, mask, perimeter)

    # allFeatures.append(extractedFeatures)
    # allLabels.append(extractedLabels)
    try :
        extractedFeatures, extractedLabels = cellMorphHelper.extractFeatures(image, mask, perimeter)

        allFeatures.append(extractedFeatures)
        allLabels.append(extractedLabels)
        colors.append(cell.color)
    except:
        print('Couldn\'t get features for cell')
    if cellNum%100 == 0:
        print('Saving features')
        pickle.dump([allFeatures, allLabels], open('../results/allFeaturesTJ2201.pickle', "wb"))
    cellNum += 1