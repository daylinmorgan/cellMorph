# %%
import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pickle

from cellMorphHelper import procrustes

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
        fname = imageBase+'_'+str(splitNum)+'.png'
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
# %%
experiment = 'TJ2201Split16'
cells=pickle.load(open('../data/results/{}CellPerims.pickle'.format(experiment),"rb"))
# %%
colors = perims['color']
perims.drop('color', inplace=True, axis=1)
# %%
fit = umap.UMAP()
u = fit.fit_transform(perims)
# %%
plt.scatter(u[:,0], u[:,1], s=1, c=colors)