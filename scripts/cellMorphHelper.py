# %%
import os
import shutil
import cv2
import numpy as np
import pandas as pd
import pickle

import torch
import detectron2

# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import pandas as pd
import os, json, cv2, random
import matplotlib.pyplot as plt
# %matplotlib inline

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.color import rgb2hsv
from skimage.io import imread
from scipy.spatial import ConvexHull

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

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

def imSplit(im, nIms: int=16):
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

    print('Splitting composite')
    # Split up composite images
    imNames = os.listdir(os.path.join(dataDir, 'composite'))

    for imName in imNames:
        # Read and split mask
        im = cv2.imread(os.path.join(dataDir, 'composite', imName))
        tiles = imSplit(im, nIms)

        # For each mask append a number, then save it
        for num, im in enumerate(tiles):
            newImName =  '.'.join([imName.split('.')[0]+'_'+str(num+1), imName.split('.')[1]])
            newImPath = os.path.join(splitDir, 'composite', newImName)
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

def convertLabels(labelDir):
    """
    Converts labels from their .csv representation to a pickled dictionary
    Inputs:
    labelDir: Directory where labels are stored as train and val
    Outputs:
    A nested pickled dictionary for each image containing
    information about each cell's identity
    """
    labels = {}
    # Walk through directory and add each image's information
    for root, dirs, files in os.walk(labelDir):
        print(root)
        for imageLabel in files:
            if imageLabel.endswith('.csv'):
                labelDf = pd.read_csv(os.path.join(root,imageLabel))
                imBase = '_'.join(imageLabel.split('.')[0].split('_')[1:])

                maskLabels = labelDf['maskLabel']
                groups = labelDf['fluorescence']

                # Each image also has a dictionary which is accessed by the mask label
                labels[imBase] = {}
                for maskLabel, group in zip(maskLabels, groups):
                    labels[imBase][maskLabel] = group
    saveName = os.path.join(labelDir, 'labels.pkl')
    pickle.dump(labels, open(saveName, "wb"))

def getSegmentModel(modelPath: str):
    """
    Gets a segmentation model that can be used to output masks
    Inputs:
    modelPath: Folder with model. Final model must be named model_final.pth
    Outputs:
    Mask-RCNN model
    """
    cfg = get_cfg()
    if not torch.cuda.is_available():
        print('CUDA not available, resorting to CPU')
        cfg.MODEL.DEVICE='cpu'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("cellMorph_Train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.OUTPUT_DIR = modelPath
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    predictor = DefaultPredictor(cfg)

    return predictor
def segmentGreen(RGB):
    """
    Finds green pixels from Incucyte data
    Input: RGB image
    Output: # of green pixels and mask of green pixels
    """
    # def segment
    I = rgb2hsv(RGB)

    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.129
    channel1Max = 0.845

    # Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.309
    channel2Max = 1.000

    # Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.761
    channel3Max = 1.000

    # Create mask based on chosen histogram thresholds
    sliderBW =  np.array(I[:,:,0] >= channel1Min ) & np.array(I[:,:,0] <= channel1Max) & \
                np.array(I[:,:,1] >= channel2Min ) & np.array(I[:,:,1] <= channel2Max) & \
                np.array(I[:,:,2] >= channel3Min ) & np.array(I[:,:,2] <= channel3Max)
    BW = sliderBW

    # Initialize output masked image based on input image.
    maskedRGBImage = RGB.copy()

    # Set background pixels where BW is false to zero.
    maskedRGBImage[~np.dstack((BW, BW, BW))] = 0

    nGreen = np.sum(BW)
    return([nGreen, BW])

def segmentRed(RGB):
    """
    Finds red pixels from Incucyte data
    Input: RGB image
    Output: # of red pixels and mask of green pixels
    """
    # Convert RGB image to chosen color space
    I = rgb2hsv(RGB)

    # Define thresholds for channel 1 based on histogram settings
    channel1Min = 0.724
    channel1Max = 0.185

    # Define thresholds for channel 2 based on histogram settings
    channel2Min = 0.277
    channel2Max = 1.000

    # Define thresholds for channel 3 based on histogram settings
    channel3Min = 0.638
    channel3Max = 1.000

    # Create mask based on chosen histogram thresholds
    sliderBW =  np.array(I[:,:,0] >= channel1Min )  | np.array(I[:,:,0] <= channel1Max)  & \
                np.array(I[:,:,1] >= channel2Min ) &  np.array(I[:,:,1] <= channel2Max) & \
                np.array(I[:,:,2] >= channel3Min ) &  np.array(I[:,:,2] <= channel3Max)
    BW = sliderBW

    # Initialize output masked image based on input image.
    maskedRGBImage = RGB.copy()

    # Set background pixels where BW is false to zero.

    maskedRGBImage[~np.dstack((BW, BW, BW))] = 0

    nRed = np.sum(BW)
    return([nRed, BW])
# %%
RGB = imread('/stor/work/Brock/Tyler/cellMorph/data/AG2021Split16/composite/composite_C5_1_2020y06m19d_00h33m_1.jpg')
