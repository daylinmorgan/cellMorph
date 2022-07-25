# %%
import sys, importlib
importlib.reload(sys.modules['cellMorphHelper'])
from cellMorphHelper import splitExpIms
# %%
experiment = 'TJ2201'
# %%
splitExpIms(experiment)