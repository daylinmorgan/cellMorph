# %%
import pickle
import skimage
import random
# %%
esamNeg=pickle.load(open('../results/TJ2201Split16ESAMNeg.pickle',"rb"))
coculture=pickle.load(open('../results/TJ2201Split16CellPerims.pickle',"rb"))

# %%
# Subset some cells
random.seed(1234)
coculturePerm = random.sample(range(len(coculture)), len(coculture))
esamNegPerm = random.sample(range(len(esamNeg)), len(esamNeg))

