# %%
# %%
import cellMorphHelper

import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage.io import imread
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16/TJ2201Split16-E2.pickle',"rb"))
# %% Get dates to pick an easy one
dates = np.array([cell.date for cell in esamNeg])
uniqueDates = list(np.unique(dates))
uniqueDates.sort()

plt.imshow(imread(esamNeg[52553].composite))
# %% Visualize proper order of images
cell = esamNeg[52553]
pcName = f'composite_{cell.imageBase}.png'
plt.imshow(imread('../data/TJ2201/composite/'+pcName))
# %% Gather all cells
