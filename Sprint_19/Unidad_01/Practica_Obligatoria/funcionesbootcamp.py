import os
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
import numpy as np
import cv2


def read_data(directorio, reshape_dim = (32,32)):
    ''' Esta función lee archivos de imagen y las redimensiona según la tupla del argumento.
        Sirve para crear subsets (X,y)
        arg: 
        - directorio: (path)
        - reshape_dime: (tupla, 2)
        Devuelve arrays de X e y'''
    
    X = []
    y = []
    for folder in os.listdir(directorio):
        if os.path.isdir('/'.join([directorio, folder])):
            for file in os.listdir('/'.join([directorio, folder])):

                image = imread('/'.join([directorio, folder, file]))
                image = cv2.resize(image, reshape_dim) # Redimensionamos las imágenes a 32x32

                X.append(image)
                y.append(folder)

    return np.array(X),np.array(y)


def show_images_batch(paisajes, names = [], n_cols = 5, size_scale = 2):
    '''Esta función pinta imágenes de datasets
        args:
        - paisajes(imagen): X_train[indice]
        - names: lista con los nombres de las categorias
        -n_cols: n columnas de graficos'''
    
    
    n_rows = ((len(paisajes) - 1) // n_cols + 1)
    plt.figure(figsize=(n_cols * size_scale, n_rows * 1.1*size_scale))
    for index, paisaje in enumerate(paisajes):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(paisaje, cmap = "Greys")
        plt.axis("off")
        if len(names):
            plt.title(names[index])