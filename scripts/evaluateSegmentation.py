# %%
import json
import os
import pickle
import random

import cellMorphHelper
import cv2
import detectron2
import matplotlib.pyplot as plt

# import some common libraries
import numpy as np
import pandas as pd
import torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from scipy.spatial import ConvexHull

# Import image processing
from skimage import img_as_float, measure

import cellMorph

# %matplotlib inline


# %%
experiment = "TJ2201Split16"
cells = pickle.load(open("../results/{}CellPerims.pickle".format(experiment), "rb"))
predictor = cellMorphHelper.getSegmentModel("../output/AG2021Split16")
# %%
# Get dates
dates = []
for cell in cells:
    date = "_".join(cell.imageBase.split("_")[2:])
    date = cellMorphHelper.convertDate(date)
    dates.append(date)

mostConfluent = np.where(np.array(dates) == max(dates))[0]
# %%
cellNum = mostConfluent[15]
cell = cells[cellNum]
cellMorphHelper.viewPredictorResult(predictor, cell.phaseContrast)
# %%
from detectron2.utils.visualizer import ColorMode

dataset_dicts = getCellDicts(inputs[0], inputs[1])
# %%
expDir = "../data/AG2021Split16"
stage = "val"

inputs = [expDir, stage]
random.seed(123)
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(
        im
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        im[:, :, ::-1],
        metadata=cell_metadata,
        scale=1,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.imshow(im)
    plt.subplot(122)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.show()
# %% Save training image that is predicted
expDir = "../data/AG2021Split16"
stage = "train"

inputs = [expDir, stage]

dataset_dicts = getCellDicts(inputs[0], inputs[1])
# %%
random.seed(1234)
for d in random.sample(dataset_dicts, 4):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=cell_metadata, scale=0.5)
    out = visualizer.draw_dataset_dict(d)

    plt.figure(figsize=(20, 20))
    plt.subplot(121)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.title("Input Training Image")
    outputs = predictor(
        img
    )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
        img[:, :, ::-1],
        metadata=cell_metadata,
        scale=1,
        instance_mode=ColorMode.IMAGE_BW,  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    plt.subplot(122)
    plt.imshow(out.get_image()[:, :, ::-1])
    plt.title("Phase Contrast Predicted Image")
    plt.savefig("../data/initMaskRCNNRes.png")


# %%
