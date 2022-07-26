# %%
import os
from skimage.io import imread, imsave
# %%
pcDir = '../data/TJ2201Split16/phaseContrast'
compositeDir = '../data/TJ2201Split16/composite'

pcIms = os.listdir(pcDir)
for imPath in pcIms[0:3]:
    im = imread(os.path.join(pcDir, imPath))
    imPathNew = imPath.split('.')
    imPathNew[-1] = '.jpg'
    imPathNew = '.'.join(imPathNew)
    print('{} --> {}'.format(imPath, impathNew))
    # imsave(imPathNew, im)