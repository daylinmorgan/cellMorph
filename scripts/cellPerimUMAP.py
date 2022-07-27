# %%
import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
import pickle

from cellMorphHelper import cellPerims
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