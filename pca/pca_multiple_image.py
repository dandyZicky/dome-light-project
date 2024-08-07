
#%%
from cv2 import split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from sklearn import decomposition

from utils import img_load

FOLDER_DATASET_NAME = "/dataset/images/"

PATH = str(sys.path[0] + FOLDER_DATASET_NAME)
FILE = []

for i in range(1, 8):
    FILE.append(str(f'{PATH}turki{i}.jpg'))

N_COMPONENTS = 100

PLOTITNG = 1



#%%
r_band = [0 for _ in range(len(FILE))] 
g_band = [0 for _ in range(len(FILE))]
b_band = [0 for _ in range(len(FILE))]
for idx, img in enumerate(FILE):
    temp = img_load.load_image_file(img)
    r_band[idx], g_band[idx], b_band[idx] = split(temp)

read_img_ex_list = np.array(r_band[0].flatten())
for i in range(1, 7):
    arr = r_band[i]
    read_img_ex_list = np.vstack((read_img_ex_list, arr.flatten()))

for i in range(7):
    arr = g_band[i]
    read_img_ex_list = np.vstack((read_img_ex_list, arr.flatten()))

for i in range(7):
    arr = b_band[i]
    read_img_ex_list = np.vstack((read_img_ex_list, arr.flatten()))


#%%
# figure1, axis1 = plt.subplots(1, 7)
# for val, img in enumerate(uniband):
#     axis1[val].imshow(img, cmap='gray')

matrix_img = read_img_ex_list

#%%
pca = decomposition.PCA().fit(matrix_img)


#%%
eigen_img = pca.components_
if PLOTITNG:
    plt.bar(list(range(1, len(pca.explained_variance_ratio_)+1)), pca.explained_variance_ratio_)

print(f"Sum of explained variance ratio : {sum(pca.explained_variance_ratio_)}")

#%%
if PLOTITNG:
    figure2, axis2 = plt.subplots(2, 2, figsize=(20, 20))
    for i in range(4):
        axis2[i//2][i%2].imshow(eigen_img[-(i+2)].reshape(r_band[0].shape), cmap='gray')


# %%
for i in range(21):
    plt.imsave(f'pca{i}.png', eigen_img[i].reshape(r_band[0].shape), cmap='gray')
# %%
plt.show()