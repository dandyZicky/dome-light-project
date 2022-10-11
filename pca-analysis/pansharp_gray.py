
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import decomposition

from utils import img_load

FOLDER_DATASET_NAME = "/dataset/images/"

PATH = str(sys.path[0] + FOLDER_DATASET_NAME)
FILE = str(PATH + "IMG_0894.JPG")

N_COMPONENTS = 126

PLOTITNG = 1



#%%
uniband = img_load.load_image_file(FILE, 'L')

print(uniband.shape)



#%%
if PLOTITNG:
    plt.imshow(uniband, cmap='gray')


#%%
pca = decomposition.PCA().fit(uniband)


#%%
eigen_img = pca.components_[:N_COMPONENTS]
if PLOTITNG:
    plt.bar(list(range(1, len(pca.explained_variance_ratio_)+1)), pca.explained_variance_ratio_)

print(f"Variance ratio : {sum(pca.explained_variance_ratio_)}")

#%%
print(eigen_img.shape)
if PLOTITNG:
    plt.imshow(eigen_img.reshape(uniband.shape), cmap='gray')


# %%
