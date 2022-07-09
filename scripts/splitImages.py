# %% [markdown]
"""
# Splits images into sub-images
Hopefully this will help Mask-RCNN not get overwhelmed with a large image with too many marked objects
"""
# %%
import cv2
import skimage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
# %% Test loading image
fname = '../data/AG2021/PhaseContrast/PhaseContrast_C5_1_2020y06m19d_03h33m.jpg'
im = cv2.imread(fname)
plt.imshow(im)
# %%
nIms = 16
div = int(np.sqrt(nIms))
M = im.shape[0]//div
N = im.shape[1]//div

tiles = []
for y in range(0,im.shape[1],N):
    for x in range(0,im.shape[0],M):
        tiles.append(im[x:x+M,y:y+N])

# %% View split images
nIms = len(tiles)
nRowCol = int(np.sqrt(nIms))

c = 1
nIm = 1
for col in range(1, nRowCol+1):
    for row in range(1, nRowCol+1):
        plt.subplot(row,col,nIm)
        plt.imshow(tiles[nIm-1])
        nIm += 1