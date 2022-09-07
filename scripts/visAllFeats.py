# %%
import pickle
import umap
import matplotlib.pyplot as plt
import pandas as pd
# %%
featureRes=pickle.load(open('../results/allFeaturesTJ2201.pickle',"rb"))
# %%
features = pd.DataFrame(featureRes[0])
# %%
features = features.dropna(axis='columns')
# %%
fit = umap.UMAP()
u = fit.fit_transform(features.values)
# %%
plt.scatter(u[:,0], u[:,1], c = featureRes[2])