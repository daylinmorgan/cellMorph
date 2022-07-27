# %%
# import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pickle
import random
import numpy as np

≈

from skimage.color import rgb2gray
from skimage.measure import find_contours
import cv2

import pyfeats
# %%
experiment = 'TJ2201Split16'
cells=pickle.load(open('../data/results/{}CellPerims.pickle'.format(experiment),"rb"))

# %% Pyfeatures test
# Take subset of cells
maxCells = 400
cellSample = np.array(cells)[random.sample(range(len(cells)), 5000)]
allFeatures, allLabels = [], []
y = []
c = 1

# Find textural features for each cell
for cell in cellSample:
    if cell.color == 'NaN':
        continue
    print('\t{}'.format(cell.color))
    y.append(cell.color)
    print('Cell {}/{}'.format(c, len(cells)))
    f = rgb2gray(imread(cell.phaseContrast))
    mask = cell.mask
    a = np.where(mask==True)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])

    f = f[bbox[0]:bbox[1], bbox[2]:bbox[3]]
    image = f*255
    mask = mask[bbox[0]:bbox[1], bbox[2]:bbox[3]].astype(int)

    perimeter = cell.perimeter

    # Write features and their meaning to list
    features, labels = [], []

    featuresFOS, labelsFOS = pyfeats.fos(image, mask)
    features+= list(featuresFOS)
    labels += labelsFOS

    features_mean, features_range, labels_mean, labels_range \
        =   pyfeats.glcm_features(image, ignore_zeros=True)
    features += list(features_mean)
    features += list(features_range)
    labels += labels_mean
    labels += labels_range

    featuresGLDS, labelsGLDS =   pyfeats.glds_features(image, mask, Dx=[0,1,1,1], Dy=[1,1,0,-1])
    features += list(featuresGLDS)
    labels += labelsGLDS

    featuresNGTDM, labelsNGTDM =  pyfeats.ngtdm_features(image, mask, d=1)
    features += list(featuresNGTDM)
    labels += labelsNGTDM

    featuresSFM, labelsSFM =    pyfeats.sfm_features(image, mask, Lr=4, Lc=4)
    features += list(featuresSFM)
    labels += labelsSFM

    featuresLTE, labelsLTE =    pyfeats.lte_measures(image, mask, l=7)
    features += list(featuresLTE)
    labels += labelsLTE

    featuresFDTA, labelsFDTA =   pyfeats.fdta(image, mask, s=3)
    features += list(featuresFDTA)
    labels += labelsFDTA

    featuresGLRLM, labelsGLRLM =  pyfeats.glrlm_features(image, mask, Ng=256)
    features += list(featuresGLRLM)
    labels += labelsGLRLM

    featuresFPS, labelsFPS =    pyfeats.fps(image, mask)
    features += list(featuresFPS)
    labels += labelsFPS

    featuresShape, labelsShape = pyfeats.shape_parameters(image, mask, perimeter, pixels_per_mm2=1)
    features += list(featuresShape)
    labels += labelsShape

    featuresHOS, labelsHOS =     pyfeats.hos_features(image, th=[135,140])
    features += list(featuresHOS)
    labels += labelsHOS

    featuresLBP, labelsLBP =     pyfeats.lbp_features(image, image, P=[8,16,24], R=[1,2,3])
    features += list(featuresLBP)
    labels += labelsLBP

    featuresGLSZM, labelsGLSZM =   pyfeats.glszm_features(image, mask)
    features += list(featuresGLSZM)
    labels += labelsGLSZM

    allFeatures.append(features)
    allLabels.append(labels)
    c += 1
    if c > maxCells:
        break
# %%
X = pd.DataFrame(allFeatures)
X['colors'] = y
X.to_csv('../data/{}textureFeatures.csv'.format(experiment), index=0)
# %%
