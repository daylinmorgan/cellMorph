# %%
import umap
import pandas as pd
import numpy
import matplotlib.pyplot as plt
# %%
perims = pd.read_csv("../data/results/cellPerims.csv", index_col=0)
# %%
colors = perims['color']
perims.drop('color', inplace=True, axis=1)
# %%
fit = umap.UMAP()
u = fit.fit_transform(perims)
# %%
plt.scatter(u[:,0], u[:,1], s=1, c=colors)