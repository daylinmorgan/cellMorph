# %% [markdown]
"""
Holds necessary structural information for storing cell information
"""
import datetime
import os

import matplotlib.pyplot as plt

# %%
import numpy as np
from scipy.interpolate import interp1d
from skimage.color import rgb2hsv
from skimage.io import imread
from skimage.measure import find_contours


# %%
def interpolatePerimeter(perim: np.array, nPts: int = 150):
    """
    Interpolates a 2D curve to a given number of points.
    Adapted from: https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
    Inputs:
    perim: 2D numpy array of dimension nptsx2
    nPts: Number of interpolated points
    Outputs:
    perimInt: Interpolated perimeter
    """
    distance = np.cumsum(np.sqrt(np.sum(np.diff(perim, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, nPts)

    interpolator = interp1d(distance, perim, kind="cubic", axis=0)
    perimInt = interpolator(alpha)

    return perimInt


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
    sliderBW = (
        np.array(I[:, :, 0] >= channel1Min)
        & np.array(I[:, :, 0] <= channel1Max)
        & np.array(I[:, :, 1] >= channel2Min)
        & np.array(I[:, :, 1] <= channel2Max)
        & np.array(I[:, :, 2] >= channel3Min)
        & np.array(I[:, :, 2] <= channel3Max)
    )
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
    sliderBW = np.array(I[:, :, 0] >= channel1Min) | np.array(
        I[:, :, 0] <= channel1Max
    ) & np.array(I[:, :, 1] >= channel2Min) & np.array(
        I[:, :, 1] <= channel2Max
    ) & np.array(
        I[:, :, 2] >= channel3Min
    ) & np.array(
        I[:, :, 2] <= channel3Max
    )
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
    RGB[~np.dstack((mask, mask, mask))] = 0
    nGreen, BW = segmentGreen(RGB)
    nRed, BW = segmentRed(RGB)
    if nGreen >= (nRed + 100):
        return "green"
    elif nRed >= (nGreen + 100):
        return "red"
    else:
        return "NaN"


def convertDate(date):
    """
    Returns a python datetime format of the Incucyte date format
    NOTE: This is very hardcoded and relies on a specific format.

    Input example: 2022y04m11d_00h00m
    Output example: 2022-04-11 00:00:00
    """
    year = int(date[0:4])
    month = int(date[5:7])
    day = int(date[8:10])
    hour = int(date[12:14])
    minute = int(date[15:17])

    date = datetime.datetime(year, month, day, hour, minute)

    return date


# %%
class cellPerims:
    """
    Assigns properties of cells from phase contrast imaging
    """

    def __init__(self, experiment, imageBase, splitNum, mask):
        self.experiment = experiment
        self.imageBase = imageBase
        self.splitNum = splitNum
        fname = imageBase + "_" + str(splitNum) + ".png"
        self.phaseContrast = os.path.join(
            "../data", experiment, "phaseContrast", "phaseContrast_" + fname
        )
        self.composite = os.path.join(
            "../data", experiment, "composite", "composite_" + fname
        )
        self.mask = mask

        self.perimeter = self.findPerimeter()

        self.color = findFluorescenceColor(self.composite, self.mask)

        self.perimAligned = ""

        self.perimInt = interpolatePerimeter(self.perimeter)

        self.well = imageBase.split("_")[0]
        date = imageBase.split("_")[2:]
        assert "y" in date[0], "No year in date, check to make sure it is a split image"
        date = "_".join(date)
        self.date = convertDate(date)

    def imshow(self):
        RGB = imread(self.composite)
        mask = self.mask
        # RGB[~np.dstack((mask,mask,mask))] = 0
        plt.figure()
        plt.imshow(RGB)
        plt.plot(self.perimeter[:, 1], self.perimeter[:, 0], c="red")
        plt.title(self.color)

    def findPerimeter(self):
        c = find_contours(self.mask)
        # assert len(c) == 1, "Error for {}".format(self.composite)
        return c[0]


# If something bad happened where you need to pickle a new object, fix it with this:
# for cell in cells:
#     cell.__class__ = eval(cell.__class__.__name__)`
