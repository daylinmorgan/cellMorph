# %%
import pickle
import cellMorph
import cellMorphHelper
# %%
esamNeg = pickle.load(open('../results/TJ2201Split16ESAMNeg.pickle',"rb"))
AG2217 = pickle.load(open('../results/AG2217-ESAM-TGFB/AG2217-ESAM-TGFB.pickle',"rb"))

# %%
datesAG2217 = []
for cell in AG2217:
    pass