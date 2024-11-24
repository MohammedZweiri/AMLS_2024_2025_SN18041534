#========Import required libraries==================
import cv2 as cv
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#==========this libraries for the models =============
# from tensorflow import keras
# from tensorflow.keras.utils.np_utils import normalize, to_categorical
# from tensorflow.keras import optimizers
# from keras.preprocessing.image import ImageDataGenerator
# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
# from sklearn.metrics import ConfusionMatrixDisplay
# from tensorflow.keras.layers import BatchNormalization


#============== Splitting the data into training and validation =========
from sklearn.model_selection import train_test_split

 
#========== This libraries for getting the result of accurcy and confusion matrix of the model =======
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, jaccard_score

#======================= Avoiding warnings messages===============
import warnings
warnings.filterwarnings('ignore')

#============= This library used for saved the model =================
import pickle

import medmnist
from medmnist import INFO, Evaluator

#dataset = BloodMNIST(split='train', download=True, root='../Datasets/BloodMNIST')

def dataset_download(dataset_name):

    print(f"downloading the MedMNIST dataset. Source information: MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE} ")

    download = True
    
    dataset_info = INFO[dataset_name]

    dataset_class = getattr(medmnist, dataset_info('python_class'))

    print(dataset_info['description'])

    train_dataset = dataset_class(split='train', download=download)
    validation_dataset = dataset_class(split='val', download=download)
    test_dataset = dataset_class(split='test', download=download)

    print("Training Dataset")
    print(train_dataset)

    if train_dataset.imgs.ndim == 3:
        print("Adding channel to images...")
        train_dataset.imgs = np.expand_dims(train_dataset.imgs, axis=1)

    print("Validation Dataset")
    print(validation_dataset)

    if validation_dataset.imgs.ndim == 3:
        print("Adding channel to images...")
        validation_dataset.imgs = np.expand_dims(validation_dataset.imgs, axis=1)

    print("Testing Dataset")
    print(test_dataset)

    if test_dataset.imgs.ndim == 3:
        print("Adding channel to images...")
        test_dataset.imgs = np.expand_dims(test_dataset.imgs, axis=1)

    print("Shapes of images")
    print(f"Training shape: {str(train_dataset.imgs.shape)}")
    print(f"Validation shape: {str(validation_dataset.imgs.shape)}")
    print(f"Testing shape: {str(test_dataset.imgs.shape)}")

    return train_dataset, validation_dataset, test_dataset

