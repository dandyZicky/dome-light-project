
#%%
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
uniband = []
for img in FILE:
    uniband.append(img_load.load_image_file(img, 'L'))

read_img_ex_list = np.array(uniband[0].flatten())
for i in range(1, 7):
    arr = uniband[i]
    read_img_ex_list = np.vstack((read_img_ex_list, arr.flatten()))


#%%
figure1, axis1 = plt.subplots(1, 7)
for val, img in enumerate(uniband):
    axis1[val].imshow(img, cmap='gray')

matrix_img = read_img_ex_list

#%%
pca = decomposition.PCA().fit(matrix_img)


#%%
eigen_img = pca.components_
if PLOTITNG:
    plt.bar(list(range(1, len(pca.explained_variance_ratio_)+1)), pca.explained_variance_ratio_)

print(f"Variance ratio : {sum(pca.explained_variance_ratio_)}")

#%%
if PLOTITNG:
    figure2, axis2 = plt.subplots(1, 7, figsize=(20, 20))
    for i in range(7):
        axis2[i].imshow(eigen_img[i].reshape(uniband[0].shape), cmap='gray')


# %%
for i in range(7):
    plt.imsave(f'pca{i}.jpg', eigen_img[i].reshape(uniband[0].shape), cmap='gray')
# %%
