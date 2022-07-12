# %%
import os
import shutil
import cv2
import numpy as np
# %%
def makeNewExperimentDirectory(experimentName):
    """
    Properly populates a new blank experiment data directory
    Inputs:
    - experimentName: Name of new experiment
    Outputs:
    None
    """
    assert os.path.isdir('../data')

    # Make base directory for experiment
    dataFolder = os.path.join('../data', experimentName)

    os.makedirs(dataFolder, exist_ok=True)

    # Make remaining sub-folders
    composite = os.path.join('../data', experimentName, 'composite')
    labelTrain = os.path.join('../data', experimentName, 'label', 'train')
    labelVal = os.path.join('../data', experimentName, 'label', 'val')
    mask = os.path.join('../data', experimentName, 'mask')
    phaseContrast = os.path.join('../data', experimentName, 'phaseContrast')

    newFolders = [composite, labelTrain, labelVal, mask, phaseContrast]

    for folder in newFolders:
        os.makedirs(folder, exist_ok=True)

def imSplit(im, nIms=16):
    """
    Splits images into given number of tiles
    Inputs:
    im: Image to be split
    nIms: Number of images (must be a perfect square)
    """
    div = int(np.sqrt(nIms))

    nRow = im.shape[0]
    nCol = im.shape[1]

    M = nRow//div
    N = nCol//div
    tiles = []
    for y in range(0,im.shape[1],N): # Column
        for x in range(0,im.shape[0],M): # Row
            tiles.append(im[x:x+M,y:y+N])
    return tiles

def splitExpIms(experiment, nIms=16):
    """
    Splits up Incucyte experiment into given number of chunks. Does this for both masks
    and phase contrast images
    Inputs:
    experiment: Experiment name located under ../data/
    nIms: Number of tiles for each image
    Outputs:
    New and populated directory
    """

    # Split masks
    dataDir = os.path.join('../data',experiment)
    splitDir = os.path.join('../data',experiment+'Split'+str(nIms))

    # Make new directory
    if os.path.isdir(splitDir):
        shutil.rmtree(splitDir)
    
    makeNewExperimentDirectory(splitDir)

    print('Splitting masks')    
    # Split and save masks
    maskNames = os.listdir(os.path.join(dataDir, 'mask'))

    for maskName in maskNames:
        # Read and split mask
        mask = cv2.imread(os.path.join(dataDir, 'mask', maskName), cv2.IMREAD_UNCHANGED)
        maskSplit = imSplit(mask, nIms)

        # For each mask append a number, then save it
        for num, mask in enumerate(maskSplit):
            newMaskName =  '.'.join([maskName.split('.')[0]+'_'+str(num+1), maskName.split('.')[1]])
            newMaskPath = os.path.join(splitDir, 'mask', newMaskName)
            cv2.imwrite(newMaskPath, mask)

    print('Splitting images')
    # Split up phase contrast images
    imNames = os.listdir(os.path.join(dataDir, 'phaseContrast'))

    for imName in imNames:
        # Read and split mask
        im = cv2.imread(os.path.join(dataDir, 'phaseContrast', imName))
        tiles = imSplit(im, nIms)

        # For each mask append a number, then save it
        for num, im in enumerate(tiles):
            newImName =  '.'.join([imName.split('.')[0]+'_'+str(num+1), imName.split('.')[1]])
            newImPath = os.path.join(splitDir, 'phaseContrast', newImName)
            cv2.imwrite(newImPath, im)

    # Copy over labels
    originalTrain = os.path.join(dataDir, 'label', 'train')
    originalVal = os.path.join(dataDir,'label', 'val')

    newTrain = os.path.join(splitDir, 'label', 'train')
    newVal =   os.path.join(splitDir,'label', 'val')

    shutil.rmtree(newTrain)
    shutil.rmtree(newVal)
    
    shutil.copytree(originalTrain, newTrain)
    shutil.copytree(originalVal, newVal)
# %%
experiment = 'AG2021'
nIms = 16
splitExpIms('AG2021', nIms=16)
# %%
