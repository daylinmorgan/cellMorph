# %% [markdown]
# # Guess the Phenotype
# ## Run all cells until prompted

import datetime
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
from skimage.color import label2rgb
from skimage.io import imread
from skimage.measure import label

# %%
import cellMorph


# %%
def clearEdgeCells(cell):
    mask = cell.mask
    maskDilate = binary_dilation(mask)
    maskFinal = clear_border(maskDilate)
    if np.sum(maskFinal) == 0:
        return 0
    else:
        return 1


# %%
print("Loading cell data, hold on...")
esamNeg = pickle.load(open("../results/TJ2201Split16/TJ2201Split16-E2.pickle", "rb"))
esamPos = pickle.load(open("../results/TJ2201Split16/TJ2201Split16-D2.pickle", "rb"))
# Filter dates
unFiltNum = len(esamNeg + esamPos)
esamNeg = [cell for cell in esamNeg if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
esamPos = [cell for cell in esamPos if cell.date < datetime.datetime(2022, 4, 8, 16, 0)]
diff = unFiltNum - len(esamNeg + esamPos)
print(f"Filtered {diff} cells by date")
# Filter color
unFiltNum = len(esamNeg + esamPos)
esamNeg = [cell for cell in esamNeg if cell.color == "red"]
esamPos = [cell for cell in esamPos if cell.color == "green"]
diff = unFiltNum - len(esamNeg + esamPos)
print(f"Filtered {diff} cells by color")
# Filter borders
unFiltNum = len(esamNeg + esamPos)
esamNeg = [cell for cell in esamNeg if clearEdgeCells(cell) == 1]
esamPos = [cell for cell in esamPos if clearEdgeCells(cell) == 1]
diff = unFiltNum - len(esamNeg + esamPos)
print(f"Filtered {diff} cells from edges")
cells = esamNeg + esamPos
# %%
imEsamNeg = [f"{cell.imageBase}_{cell.splitNum}" for cell in (esamNeg)]
imEsamPos = [f"{cell.imageBase}_{cell.splitNum}" for cell in (esamPos)]

allImageBases = np.array(imEsamNeg + imEsamPos)
# %%
def labelImage(cells, cell, imageBases):
    # Find where other cells have been identified
    imageBase = cell.imageBase
    splitNum = cell.splitNum
    cellIdx = np.where(imageBases == f"{imageBase}_{splitNum}")[0]
    # Collect their perimeters
    perims = []
    for idx in cellIdx:
        perims.append(cells[idx].perimeter)

    # Plot image
    composite = imread(cells[cellIdx[0]].composite)

    plt.imshow(composite)
    for perim in perims:
        plt.plot(perim[:, 1], perim[:, 0])
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )


# labelImage(esamNeg, esamNeg[200], allImageBases)

# %%
def imValidate(cell, cells, imageBases, guess):
    """
    Makes an informative plot which shows the:
    1. Isolated cell with fluorescence
    2. Phase contrast with overlaid mask
    3. Entire composite image with outline
    4. Entire phase-contrast image with outline
    """
    if cell.color == "NaN":
        return "Unidentified fluorescence color"

    composite = imread(cell.composite)
    phaseContrast = imread(cell.phaseContrast)
    mask = cell.mask
    perimeter = cell.perimeter

    plt.subplot(2, 2, 1)
    compositeIsolate = composite.copy()
    compositeIsolate[~np.dstack((mask, mask, mask))] = 0
    plt.imshow(compositeIsolate)
    plt.plot(perimeter[:, 1], perimeter[:, 0])
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    plt.subplot(2, 2, 2)
    label_image = label(mask)
    overlay = label2rgb(label_image, image=phaseContrast, bg_label=0)
    plt.imshow(overlay)
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    plt.subplot(2, 2, 3)
    plt.imshow(composite)
    plt.plot(perimeter[:, 1], perimeter[:, 0])
    plt.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        labelleft=False,
        labelbottom=False,
    )

    plt.subplot(2, 2, 4)
    labelImage(cells, cell, imageBases)

    phenoDict = {"red": "ESAM (-)", "green": "ESAM (+)", "quit": "quit"}
    phenotype = phenoDict[cell.color]
    plt.suptitle(
        f"Identified as {phenotype} ({cell.color})\n You guessed {phenoDict[guess]}"
    )


# imValidate(cells[802], cells, allImageBases)


def askQuestion(cell):
    display.clear_output(wait=True)
    plt.figure()
    plt.imshow(imread(cell.phaseContrast))
    plt.plot(cell.perimeter[:, 1], cell.perimeter[:, 0])
    plt.title("Is this red (ESAM (-))  or green (ESAM (+))?")
    plt.show()
    while True:
        answer = input("Type in red or green, type quit to quit")
        answer = answer.lower()
        if answer in ["red", "green", "quit"]:
            break
    return answer


# %% [markdown]
# # Guess the Phenotype Game
# ### This next cell will ask you to guess the phenotype, simply type in and record your answers

# %%
# %matplotlib inline
from IPython import display

print(
    """
 _____ _     _____ ____  ____                           
/  __// \ /\/  __// ___\/ ___\                          
| |  _| | |||  \  |    \|    \                          
| |_//| \_/||  /_ \___ |\___ |                          
\____\\____/\____\\____/\____/                          
                                                        
 _____  _     ____  _____                               
/__ __\/ \ /|/  _ \/__ __\                              
  / \  | |_||| / \|  / \                                
  | |  | | ||| |-||  | |                                
  \_/  \_/ \|\_/ \|  \_/                                
                                                        
 ____  _     _____ _      ____  _____ ___  _ ____  _____
/  __\/ \ /|/  __// \  /|/  _ \/__ __\\  \///  __\/  __/
|  \/|| |_|||  \  | |\ ||| / \|  / \   \  / |  \/||  \  
|  __/| | |||  /_ | | \||| \_/|  | |   / /  |  __/|  /_ 
\_/   \_/ \|\____\\_/  \|\____/  \_/  /_/   \_/   \____\  
"""
)

nIms = 5
# Grab some random esam negative and esam positive cells
# np.random.seed(1234)
negPerm = np.random.permutation(range(len(esamNeg)))
posPerm = np.random.permutation(range(len(esamPos)))

cellsGuess = []
c = 0
for nPos in range(nIms):
    cellsGuess.append(esamPos[posPerm[nPos]])

for nNeg in range(nIms):
    cellsGuess.append(esamNeg[negPerm[nNeg]])

# Play the game
random.shuffle(cellsGuess)
correctColors = [cell.color for cell in cellsGuess]

playersAnswers = []
for cell in cellsGuess:
    answer = askQuestion(cell)
    playersAnswers.append(answer)
    if answer == "quit":
        break

# %% [markdown]
# # Ready to see how you did?
# ### The next cell will show you how many answers you got right. It will also show you the cell in isolation, with a light masking layer, and in context with the othe

# %%
correctAnswers = 0
for i in range(len(playersAnswers)):
    #     print(f'You guessed {playersAnswers[i]}')
    plt.figure()
    imValidate(cellsGuess[i], cells, allImageBases, playersAnswers[i])
    if playersAnswers[i] == cellsGuess[i].color:
        correctAnswers += 1
print(f"You got {correctAnswers} correct out of {i}")


# %%
