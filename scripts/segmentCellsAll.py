# %% [markdown]
"""
# Cell Segmentation With Mask-RCNN
This is to be run using fully-validated masks from cellpose
"""
# %%
import cellMorphHelper

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
import os, json, cv2, random, pickle
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm

# Import image processing
from skimage import measure
from skimage import img_as_float
from skimage.io import imread
from skimage.morphology import binary_dilation
from skimage.segmentation import clear_border
from scipy.spatial import ConvexHull

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode
# %% Getting/reading *.npy
experiment = 'TJ2201'
imgType = 'phaseContrast'

segDir = os.path.join('../data',experiment,'cellPoseOutput')
segFiles = os.listdir(segDir)
idx = 0

datasetDicts = []
for segFile in tqdm(segFiles[0:10]):
    # Load in cellpose output
    segFull = os.path.join(segDir, segFile)
    seg = np.load(segFull, allow_pickle=True)
    seg = seg.item()

    # Find information for each image
    imgBase = cellMorphHelper.getImageBase(seg['filename'].split('/')[-1])
    imgFile = f'{imgType}_{imgBase}.png'
    imgPath = os.path.join('../data', experiment, imgType, imgFile)
    assert os.path.isfile(imgPath)
    record = {}
    record['file_name'] = imgPath
    record['image_id'] = idx
    record['height'] = seg['img'].shape[0]
    record['width'] = seg['img'].shape[1]

    outlines = seg['outlines']
    cellNums = range(1, np.max(outlines))

    # Find information for each cell
    cells = []
    for cellNum in cellNums:
        outline = np.where(outlines==cellNum)
        px = outline[1]
        py = outline[0]
        poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
        poly = [p for x in poly for p in x]

        cell = {
            "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
            "bbox_mode": BoxMode.XYXY_ABS,
            "segmentation": [poly],
            "category_id": 0,
        }
        cells.append(cell)
    record["annotations"] = cells
datasetDicts.append(record)
idx+=1
# %% Working with split images
experiment = 'TJ2201'

segDir = os.path.join('../data',experiment,'cellPoseOutput')
segFiles = os.listdir(segDir)
idx = 0

datasetDicts = []
for segFile in tqdm(segFiles[0:1]):

    # Load in cellpose output
    segFull = os.path.join(segDir, segFile)
    seg = np.load(segFull, allow_pickle=True)
    seg = seg.item()

    splitOutlines = cellMorphHelper.imSplit(seg['outlines'])
    nSplits = len(splitOutlines)

    splitDir = f'{experiment}Split{nSplits}'

    for splitIm in range(1, len(splitOutlines)):
# %% 
datasetDicts = pickle.load(open('../output/TJ2201Split16_datasetDict.pickle',"rb"))

# %%
print(len(datasetDicts))
datasetDicts = [img for img in datasetDicts if dateCorrect(img)]
print(len(datasetDicts))
# %%
DatasetCatalog.register("cellMorph_" + "train", lambda x: datasetDicts)
MetadataCatalog.get("cellMorph_" + "train").set(thing_classes=["cell"])
cell_metadata = MetadataCatalog.get("cellMorph_train")
# %% Visualization of data loader
# datasetDicts = getCellDicts(inputs[0], inputs[1])
for d in random.sample(datasetDicts, 1):
    print(d["file_name"])
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cell_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
# %%
from detectron2.engine import DefaultTrainer

cfg = get_cfg()
if not torch.cuda.is_available():
    print('CUDA not available, resorting to CPU')
    cfg.MODEL.DEVICE='cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("cellMorph_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (cell). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
cfg.OUTPUT_DIR = '../output/AG2021Split16'
# %%
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()



# %%
