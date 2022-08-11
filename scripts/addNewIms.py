#!/usr/bin/env python3
"""
This is a script that will automatically add new images to the correct folder

"""
# %%
import unittest
import os

from cellMorphHelper import getImageBase

def getImageBase(imName):
    """
    Gets the "base" information of an image. 
    Files should be named as follows:
    imageType_Well_WellSection_Date.extension

    The image base has no extension or image type

    Inputs:
    imName: String of image name

    Outputs:
    imageBase: The core information about the image's information in the incucyte
    """

    imageBase = '_'.join(imName.split('.')[0].split('_')[1:])

    return imageBase
# %% Validate images
newFolder = '../data/TJ2201New'
compositeFolder = os.path.join(newFolder, 'composite')
phaseContrastFolder = os.path.join(newFolder, 'phaseContrast')
compositeFiles = os.listdir(compositeFolder)
phaseContrastFiles = os.listdir(phaseContrastFolder)

assert len(compositeFiles) == len(phaseContrastFiles), 'Different # of files'

compositeBases, pcBases = [], []
for composite, pc in zip(compositeFiles, phaseContrastFiles):
    compositeBase = getImageBase(composite)
    pcBase = getImageBase(pc)

    compositeBases.append(compositeBase)
    pcBases.append(pcBase)
for pc in pcBases:
    assert pc in compositeBases, 'Phase contrast image not in composite'
    compositeBases.remove(pc)

assert len(compositeBases) == 0, 'Non-matching composite and phase contrast images'
# %%
