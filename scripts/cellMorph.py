# %%
import os
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
# %%
experimentName = 'AG2021Split'
makeNewExperimentDirectory(experimentName)