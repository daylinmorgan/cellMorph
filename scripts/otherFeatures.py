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

from cellMorph import cellPerims
from skimage.color import rgb2gray
from skimage.measure import find_contours
from skimage.io import imread
import cv2

import pyfeats
# %%
experiment = 'TJ2201Split16'
cells=pickle.load(open('../results/{}CellPerims.pickle'.format(experiment),"rb"))

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
f = rgb2gray(imread(cell.phaseContrast))
mask = cell.mask
a = np.where(mask==True)
bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

f = f[bbox[0]:bbox[1], bbox[2]:bbox[3]]
image = f*255 # Stretches values
mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(int)