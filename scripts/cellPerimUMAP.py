# %%
import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pickle

import numpy as np
from scipy.interpolate import interp1d
# %%
class cellPerims:
    """
    Assigns properties of cells from phase contrast imaging
    """
    def findPerimeter(self):
        c = measure.find_contours(self.mask)
        # assert len(c) == 1, "Error for {}".format(self.composite)
        return c[0]
    
    
    def __init__(self, experiment, imageBase, splitNum, mask):
        # try:
        self.experiment = experiment
        self.imageBase = imageBase
        self.splitNum = splitNum
        fname = imageBase+'_'+str(splitNum)+'.png'
        self.phaseContrast = os.path.join('../data', experiment, 'phaseContrast','phaseContrast_'+fname)
        self.composite = os.path.join('../data', experiment, 'composite', 'composite_'+fname)
        self.mask = mask

        self.perimeter = self.findPerimeter()

        self.color = findFluorescenceColor(self.composite, self.mask)

        self.perimAligned = ''
        self.perimInt = ''

    def imshow(self):
        RGB = imread(self.composite)
        mask = self.mask
        RGB[~np.dstack((mask,mask,mask))] = 0
        plt.figure()
        plt.imshow(RGB)
        plt.plot(self.perimeter[:,1], self.perimeter[:,0])
        plt.title(self.color)

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
# %%
experiment = 'TJ2201Split16'
cells=pickle.load(open('../results/{}CellPerims.pickle'.format(experiment),"rb"))
# %%
# %% 
redCells, greenCells = [], []

for cell in cells:
    # Align all perimeters
    cell.perimInt = interpolatePerimeter(cell.perimeter)
    # Add to appropriate list
    if cell.color == 'red':
        redCells.append(cell)
    elif cell.color == 'green':
        greenCells.append(cell)

# Align green cells
greenCells[0].perimAligned = greenCells[0].perimInt - np.mean(greenCells[0].perimInt, axis=0)
referencePerim = greenCells[0].perimAligned

c = 1
for cell in greenCells[1:]:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)
# Align red cells
for cell in redCells:
    referencePerim = greenCells[0].perimAligned
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)

cellPerims = []
cellColors = []

for cell in cells:
    if cell.color == 'NaN':
        continue
    cellPerims.append(cell.perimAligned.ravel())
    cellColors.append(cell.color)

cellPerims = pd.DataFrame(cellPerims)

pickle.dump(cells, open('../results/{}CellPerims.pickle'.format(experiment), "wb"))
# cellPerims['color'] = cellColors
# %%
# colors = perims['color']
# perims.drop('color', inplace=True, axis=1)
# %%
cellPerims2 = cellPerims.copy()
r = random.sample(range(cellPerims2.shape[0]), 10000)
# %%
fit = umap.UMAP()
u = fit.fit_transform(cellPerims2)
# %%
plt.scatter(u[:,0], u[:,1], s=1, c=cellColors2)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('ESAM +/- Perimeter UMAP')
# %%
# pickle.dump(cell, open('../results/test.pickle', "wb"))

# cell=pickle.load(open('../results/test.pickle',"rb"))

for cell in cells:
    cell.__class__ = eval(cell.__class__.__name__)

# %%
experiment = 'TJ2201Split16'
df = pd.read_csv('../data/{}textureFeatures.csv'.format(experiment), index_col=0).reset_index()
df = df.dropna()
y = df['colors']
df = df.drop(labels='colors', axis=1)
X = np.array(df)

# %%
fit = umap.UMAP()
u = fit.fit_transform(X)
# %%
plt.scatter(u[:,0], u[:,1], s=2, c=y)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('ESAM +/- Texture UMAP')
# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
clf = LogisticRegression(solver="liblinear", random_state=1234, C=1e-6,max_iter=1e7).fit(X_train, y_train)
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 0])