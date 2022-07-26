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

from scipy.interpolate import interp1d
from scipy.spatial import ConvexHull

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.color import rgb2hsv
from skimage.io import imread

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode

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
    maskDir = os.path.join(dataDir, 'mask')
    if os.path.isdir(maskDir):
        maskNames = os.listdir(maskDir)

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
    pcDir = os.path.join(dataDir, 'phaseContrast')
    if os.path.isdir(pcDir):
        imNames = os.listdir(pcDir)

        for imName in imNames:
            # Read and split mask
            im = cv2.imread(os.path.join(dataDir, 'phaseContrast', imName))
            print('\t {}'.format(imName))
            tiles = imSplit(im, nIms)

            # For each mask append a number, then save it
            for num, im in enumerate(tiles):
                newImName =  '.'.join([imName.split('.')[0]+'_'+str(num+1), imName.split('.')[1]])
                newImPath = os.path.join(splitDir, 'phaseContrast', newImName)
                cv2.imwrite(newImPath, im)

    print('Splitting composite')
    # Split up composite images
    compositeDir = os.path.join(dataDir, 'composite')
    if os.path.isdir(compositeDir):
        imNames = os.listdir(compositeDir)

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

    if os.path.isdir(originalTrain):
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
    return nGreen, BW

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
    return nRed, BW

def findFluorescenceColor(RGBLocation, mask):
    """
    Finds the fluorescence of a cell
    Input: RGB image location
    Output: Color
    """
    RGB = imread(RGBLocation)
    RGB[~np.dstack((mask,mask,mask))] = 0
    nGreen, BW = segmentGreen(RGB)
    nRed, BW = segmentRed(RGB)
    if nGreen>=(nRed+100):
        return "green"
    elif nRed>=(nGreen+100):
        return "red"
    else:
        return "NaN"

def interpolatePerimeter(perim: np.array, nPts: int=150):
    """
    Interpolates a 2D curve to a given number of points. 
    Adapted from: https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
    Inputs:
    perim: 2D numpy array of dimension nptsx2
    nPts: Number of interpolated points
    Outputs:
    perimInt: Interpolated perimeter
    """
    distance = np.cumsum( np.sqrt(np.sum( np.diff(perim, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    alpha = np.linspace(0, 1, nPts)

    interpolator =  interp1d(distance, perim, kind='cubic', axis=0)
    perimInt = interpolator(alpha)
    
    return perimInt

def viewPredictorResult(predictor, imPath: str):
    """
    Plots an image of cells with masks overlaid.
    Inputs:
    predictor: A predictor trained with detectron2
    imPath: The path of the image to load
    Outputs:
    None
    """
    im = imread(imPath)
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                #    metadata=cell_metadata, 
                   scale=1, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.imshow(out.get_image()[:,:,::-1])

def procrustes(X, Y, scaling=True, reflection='best'):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
        d, Z, [tform] = procrustes(X, Y)
    Inputs:
    ------------
    X, Y
        matrices of target and input coordinates. they must have equal
        numbers of  points (rows), but Y may have fewer dimensions
        (columns) than X.
    scaling
        if False, the scaling component of the transformation is forced
        to 1
    reflection
        if 'best' (default), the transformation solution may or may not
        include a reflection component, depending on which fits the data
        best. setting reflection to True or False forces a solution with
        reflection or no reflection respectively.
    Outputs
    ------------
    d
        the residual sum of squared errors, normalized according to a
        measure of the scale of X, ((X - X.mean(0))**2).sum()
    Z
        the matrix of transformed Y-values
    tform
        a dict specifying the rotation, translation and scaling that
        maps X --> Y
    """

    n,m = X.shape
    ny,my = Y.shape

    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 /= normX
    Y0 /= normY

    if my < m:
        Y0 = np.concatenate((Y0, np.zeros(n, m-my)),0)

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    if reflection != 'best':

        # does the current solution use a reflection?
        have_reflection = np.linalg.det(T) < 0

        # if that's not what was specified, force another reflection
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V, U.T)

    traceTA = s.sum()

    if scaling:

        # optimum scaling of Y
        b = traceTA * normX / normY

        # standarised distance between X and b*Y*T + c
        d = 1 - traceTA**2

        # transformed coords
        Z = normX*traceTA*np.dot(Y0, T) + muX

    else:
        b = 1
        d = 1 + ssY/ssX - 2 * traceTA * normY / normX
        Z = normY*np.dot(Y0, T) + muX

    # transformation matrix
    if my < m:
        T = T[:my,:]
    c = muX - b*np.dot(muY, T)

    #transformation values
    tform = {'rotation':T, 'scale':b, 'translation':c}

    return d, Z, tform
# %%
# RGB = imread('/stor/work/Brock/Tyler/cellMorph/data/AG2021Split16/composite/composite_C5_1_2020y06m19d_00h33m_1.jpg')
# %%
