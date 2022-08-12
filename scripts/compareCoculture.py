# %%
import pickle
import random
import pandas as pd
from cellMorphHelper import procrustes
import numpy as np

import matplotlib.pyplot as plt
import umap
# %%
esamNeg=pickle.load(open('../results/TJ2201Split16ESAMNeg.pickle',"rb"))
coculture=pickle.load(open('../results/TJ2201Split16CellPerims.pickle',"rb"))

# %%
# Subset some cells
random.seed(1234)
coculturePerm = random.sample(range(len(coculture)), len(coculture))
esamNegPerm = random.sample(range(len(esamNeg)), len(esamNeg))

nDesired = 5000

cocultureSub = [coculture[x] for x in coculturePerm if coculture[x].color=='red']
esamNegSub = [esamNeg[x] for x in esamNegPerm if esamNeg[x].color=='red']

cocultureSub = cocultureSub[0:nDesired]
esamNegSub = esamNegSub[0:nDesired]
origPerim = cocultureSub[0].perimAligned.copy()

# %% Align perimeters to each other
referencePerim = esamNegSub[0].perimAligned
c = 1
for cell in cocultureSub:
    currentPerim = cell.perimInt
    
    refPerim2, currentPerim2, disparity = procrustes(referencePerim, currentPerim, scaling=False)

    cell.perimAligned = currentPerim2 - np.mean(currentPerim2, axis=0)

# %% Build dataframe
labels = ['coculture' for x in range(len(cocultureSub))]+['monoculture' for x in range(len(esamNegSub))]

y = [1 if x=='coculture' else 0 for x in labels]

allCells = cocultureSub+esamNegSub
X = []
for cell in allCells:
    X.append(cell.perimAligned.ravel())

X = pd.DataFrame(X)

# %% UMAP
fit = umap.UMAP()
u = fit.fit_transform(X)
# %%
plt.scatter(u[:,0], u[:,1], s=2, c=y)
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.title('ESAM Coculture UMAP')

# %% Quick and dirty logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)
clf = LogisticRegression(solver="liblinear", random_state=1234, C=1e-6,max_iter=1e7).fit(X_train, y_train)
roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])