# %% [markdown]
"""
This script loads all images and saves their outlines.
"""
# %%
import sys, importlib
sys.path.append('../')
# importlib.reload(sys.modules['cellMorphHelper'])
import pickle
import os
import cv2
import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt
from cellMorphHelper import getSegmentModel, findFluorescenceColor, interpolatePerimeter, procrustes
from cellMorph import cellPerims

from skimage import data, measure
from skimage.segmentation import clear_border

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

# %%    
predictor = getSegmentModel('../../output/AG2021Split16')
# %% Find masks for experiment
experiment = 'TJ2201Split16'
print('Starting Experiment: {}'.format(experiment))
ims = os.listdir(os.path.join('../../data',experiment, 'phaseContrast'))
imbases = ['_'.join(im.split('.')[0].split('_')[1:-1]) for im in ims]
splitNums = [int(im.split('.')[0].split('_')[-1]) for im in ims]

cells = []

c = 1
# Go through each image
for imbase, splitNum in zip(imbases, splitNums):
    fname = 'phaseContrast_'+imbase+'_'+str(splitNum)+'.png'
    print('{}/{} Img: {}'.format(c, len(imbases), fname))
    imPath = os.path.join('../../data',experiment,'phaseContrast',fname)
    im = imread(imPath)
    outputs = predictor(im)['instances'].to("cpu")
    nCells = len(outputs)

    # Append information for each cell
    for cellNum in range(nCells):
        mask = outputs[cellNum].pred_masks.numpy()[0]
        mask = clear_border(mask)
        if np.sum(mask)>10:
            cells.append(cellPerims(experiment, imbase, splitNum, mask))
    c+=1

    # Save periodically
    if c % 100 == 0:
        print('\t Saving at ../results/{}CellPerims.pickle'.format(experiment))
        pickle.dump(cells, open('../../results/{}CellPerims.pickle'.format(experiment), "wb"))

pickle.dump(cells, open('../../results/{}CellPerims.pickle'.format(experiment), "wb"))
# %% Align green and red cells to reference cell so that they are all the same orientation
redCells, greenCells = [], []

for cell in cells:
    # Align all perimeters
    cell.perimInt = interpolatePerimeter(cell.perimeter)
    # Add to appropriate list
    if cell.color == 'red':
        redCells.append(cell)
    elif cell.color == 'green':
        greenCells.append(cell)

# Align green cells
greenCells[0].perimAligned = greenCells[0].perimInt - np.mean(greenCells[0].perimInt, axis=0)
referencePerim = greenCells[0].perimAligned

c = 1
for cell in greenCells[1:]:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim, scaling=False)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
# Align red cells
for cell in redCells:
    referencePerim = greenCells[0].perimAligned
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim, scaling=False)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
# Write altered cell perimeters
pickle.dump(cells, open('../../results/{}CellPerims.pickle'.format(experiment), "wb"))

# %%
