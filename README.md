# cellMorph
This is a repository to archive code to classify cells using their morphology. 

This is built on [detectron2](https://github.com/facebookresearch/detectron2). Only code is included, no data or trained model outputs. 

If you want to run this code, you must build `detectron2` from source. 

Files in the `data` folder are organized like so:

```
.
├── data
│   ├── experiment
│   │   ├── composite
│   │   ├── labels
│   │   │   ├── train
│   │   │   └── val
│   │   ├── masks
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
