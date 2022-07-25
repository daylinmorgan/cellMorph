# %%
import sys, importlib
importlib.reload(sys.modules['cellMorphHelper'])
import pickle
import os
import cv2
import numpy as np
from skimage.io import imread

import matplotlib.pyplot as plt
from cellMorphHelper import getSegmentModel, findFluorescenceColor, interpolatePerimeter, procrustes

from skimage import data, measure
from skimage.segmentation import clear_border

from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

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
        # try:
        self.experiment = experiment
        self.imageBase = imageBase
        self.splitNum = splitNum
        fname = imageBase+'_'+str(splitNum)+'.jpg'
        self.phaseContrast = os.path.join('../data', experiment, 'phaseContrast','phaseContrast_'+fname)
        self.composite = os.path.join('../data', experiment, 'composite', 'composite_'+fname)
        self.mask = mask

        self.perimeter = self.findPerimeter()

        self.color = findFluorescenceColor(self.composite, self.mask)

    def imshow(self):
        RGB = imread(self.composite)
        mask = self.mask
        RGB[~np.dstack((mask,mask,mask))] = 0
        plt.figure()
        plt.imshow(RGB)
        plt.plot(self.perimeter[:,1], self.perimeter[:,0])
        plt.title(self.color)
        
predictor = getSegmentModel('../output/AG2021Split16')
# %%
experiment = 'AG2021Split16'
ims = os.listdir(os.path.join('../data',experiment, 'phaseContrast'))
imbases = ['_'.join(im.split('.')[0].split('_')[1:-1]) for im in ims]
splitNums = [int(im.split('.')[0].split('_')[-1]) for im in ims]

cells = []

# nIms = 2
# imbases = imbases[0:nIms]
# splitNums = splitNums[0:nIms]
c = 1
for imbase, splitNum in zip(imbases, splitNums):
    fname = 'phaseContrast_'+imbase+'_'+str(splitNum)+'.jpg'
    print('{}/{} Img: {}'.format(c, len(imbases), fname))
    imPath = os.path.join('../data',experiment,'phaseContrast',fname)
    im = imread(imPath)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                #    metadata=cell_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    outputs = predictor(im)['instances'].to("cpu")
    nCells = len(outputs)
    for cellNum in range(nCells):
        print('\t Cell {}/{}'.format(cellNum+1, nCells))
        mask = outputs[cellNum].pred_masks.numpy()[0]
        mask = clear_border(mask)
        if np.sum(mask)>10:
            cells.append(cellPerims(experiment, imbase, splitNum, mask))
    c+=1
pickle.dump(cells, open('../data/results/AG2021Split16CellPerims.pickle', "wb"))
# %% 
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
plt.figure()
greenCells[0].perimAligned = greenCells[0].perimInt - np.mean(greenCells[0].perimInt, axis=0)
referencePerim = greenCells[0].perimAligned

c = 1
for cell in greenCells[1:]:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
# Align red cells
for cell in redCells:
    referencePerim = greenCells[0].perimAligned
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
# %% Format for further analysis
cellPerims = []
cellColors = []

for cell in cells:
    if cell.color == 'NaN':
        continue
    cellPerims.append(cell.perimAligned.ravel())
    cellColors.append(cell.color)

cellPerims = pd.DataFrame(cellPerims)
cellPerims['color'] = cellColors
cellPerims.to_csv('../data/results/cellPerims.csv')