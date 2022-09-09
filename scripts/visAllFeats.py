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
u = fit.fit_transform(normalized_df.values)
# %%
plt.scatter(u[:,0], u[:,1], c = featureRes[2])
# %%
df = features.copy()
normalized_df=(df-df.min())/(df.max()-df.min())
normalized_df = normalized_df.dropna(axis='columns')
# %%
fit = umap.UMAP()
u = fit.fit_transform(normalized_df.values)
# %%
plt.scatter(u[:,0], u[:,1], c = featureRes[2], s= 0.5)
# %%