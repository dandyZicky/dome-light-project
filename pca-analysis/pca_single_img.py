#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import decomposition
import cv2

from utils import img_load

FOLDER_DATASET_NAME = "/dataset/images/"

PATH = str(sys.path[0] + FOLDER_DATASET_NAME)
FILE = str(PATH + "IMG_0894.JPG")

N_COMPONENTS = 1000

PLOTITNG = 1

#%%
img = img_load.load_image_file(FILE)

uniband = img_load.load_image_file(FILE, 'L')

print(img.shape)
#%%

red, green, blue = cv2.split(img)

if PLOTITNG:
    figure1, axis1 = plt.subplots(1, 3)
    axis1[0].imshow(red) 
    axis1[1].imshow(green)
    axis1[2].imshow(blue)

scaled_red = red
scaled_green = green
scaled_blue = blue

#%%
pca_r = decomposition.PCA(n_components=N_COMPONENTS)
pca_r.fit(scaled_red)
trans_pca_red = pca_r.transform(scaled_red)

pca_g = decomposition.PCA(n_components=N_COMPONENTS)
pca_g.fit(scaled_green)
trans_pca_green = pca_r.transform(scaled_green)

pca_b = decomposition.PCA(n_components=N_COMPONENTS)
pca_b.fit(scaled_blue)
trans_pca_blue = pca_r.transform(scaled_blue)
#%%

if PLOTITNG:
    figure2, axis2 = plt.subplots(1, 3, layout='constrained')
    figure2.suptitle('Eigen & Variance')
    axis2[0].bar(list(range(1, N_COMPONENTS+1)), pca_r.explained_variance_ratio_)
    axis2[1].bar(list(range(1, N_COMPONENTS+1)), pca_g.explained_variance_ratio_)
    axis2[2].bar(list(range(1, N_COMPONENTS+1)), pca_b.explained_variance_ratio_)

print(f"Blue Channel : {sum(pca_b.explained_variance_ratio_)}")
print(f"Green Channel: {sum(pca_g.explained_variance_ratio_)}")
print(f"Red Channel  : {sum(pca_r.explained_variance_ratio_)}")
print(len(trans_pca_blue))

r_arr = pca_r.inverse_transform(trans_pca_red)
g_arr = pca_g.inverse_transform(trans_pca_green)
b_arr = pca_b.inverse_transform(trans_pca_blue)

img_reconstructed = cv2.cvtColor(cv2.merge((b_arr, g_arr, r_arr)).astype(np.uint8), cv2.COLOR_BGR2RGB)
if PLOTITNG:
    figure3, axis3 = plt.subplots(1, 2)
    figure3.suptitle('Before & After Decomposition')
    axis3[0].imshow(img)
    axis3[1].imshow(img_reconstructed)


if PLOTITNG: plt.show()
# print(res)
# %%
