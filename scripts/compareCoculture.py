# %%
import pickle
import random
import pandas as pd
from cellMorphHelper import procrustes
import numpy as np
# %%
esamNeg=pickle.load(open('../results/TJ2201Split16ESAMNeg.pickle',"rb"))
coculture=pickle.load(open('../results/TJ2201Split16CellPerims.pickle',"rb"))

# %%
# Subset some cells
random.seed(1234)
coculturePerm = random.sample(range(len(coculture)), len(coculture))
esamNegPerm = random.sample(range(len(esamNeg)), len(esamNeg))

nDesired = 2000

cocultureSub = [coculture[x] for x in coculturePerm if coculture[x].color=='red']
esamNegSub = [esamNeg[x] for x in esamNegPerm if esamNeg[x].color=='red']

cocultureSub = cocultureSub[0:nDesired]
esamNegSub = esamNegSub[0:nDesired]

# %% Align perimeters to each other
referencePerim = esamNegSub[0].perimAligned

for cell in cocultureSub:


# %% Build dataframe
labels = ['coculture' for x in range(len(cocoltureSub))]+['monoculture' for x in range(len(esamNeg))]

coculturePerims = [cell.p]