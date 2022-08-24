# cellMorph
This is a repository to archive code to classify cells using their morphology. This respository can be used as a source to:

1. Train an instance segmentation algorithm using small numbers of masks. 
2. Predict cell identity (transcriptomic state, lineage, etc.) from information. 

This is meant to be a comprehensive set of scripts and functions which can train on an increasing complexity of features:
- [ ] Morphology
- [x] Textural Information
- [x] Perimeter shapes
- [ ] Convolutional Neural Networks
- [ ] VQ-VAE2 Features

This is built on [detectron2](https://github.com/facebookresearch/detectron2). Only code is included, no data or trained model outputs. 

If you want to run any of the analysis, you must build `detectron2` from source using the resources above. 

Currently, the project is structured like so:

```
├── data
├── output
├── results
└── scripts
```
Where:
* `data` is the image data for each experiment
* `output` is the model output from the segmentation model
* `results` are downstream analysis results
* `scripts` are the analysis scripts

Files in the `data` folder are organized like so:

```
├── data
│   ├── experiment
│   │   ├── composite
│   │   ├── label
│   │   │   ├── train
│   │   │   └── val
│   │   ├── mask
│   │   └── phaseContrast
```

Where:
* `composite` is for validation
* `labels` contains the acceptable cells and their labels
* `masks` are the mask outputs
* `phaseContrast` is the test images

Files are named like so:

`Directory_Base Information`

Where `Directory` is the immediate directory, and `Base Information` is the output from the Incucyte (`Well_Im#_Date`). 

## Envisioned Workflow
A primary objective of this project is take code from Incucyte to analysis as soon as possible. When starting a new experiment on a trained model, data should be downloaded from the Incucyte and uploaded to the computing server. After this, the outlines can be gathered by running:

```
python imgOutline.py <experiment>
```

This should should split the images, then log the information into a list stored in .pickle format. To ease memory concerns, these are then split by well in `results`. 
