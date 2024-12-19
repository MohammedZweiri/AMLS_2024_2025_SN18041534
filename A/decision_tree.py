"""Accomplishing Task A via Decision Tree.

    This module acquires BreastMNIST data from medmnist library, then it uses the decision tree model to accuractly predict if the images are tumor or not.

    """

#========Import required libraries==================
import cv2 as cv
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import time

 
#========== This libraries for getting the result of accurcy and confusion matrix of the model =======
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier

import medmnist
from medmnist import INFO



def dataset_download(dataset_name):

    """Download dataset.

    This function downloads the BloodMNIST dataset from the medmnist library

    Args:
            dataset_name(str): The dataset name to be downloaded

    Returns:
            Training, validation and test datasets.

    """

    try:

        # Setting the correct parameters to download the dataset
        print(f"downloading the MedMNIST dataset. Source information: MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE} ")
        download = True
        dataset_info = INFO[dataset_name]
        dataset_class = getattr(medmnist, dataset_info['python_class'])

        # Performing data split
        train_dataset = dataset_class(split='train', download=download)
        validation_dataset = dataset_class(split='val', download=download)
        test_dataset = dataset_class(split='test', download=download)        

        # Adding channels to the images
        if train_dataset.imgs.ndim == 3:
            print("Adding channel to images...")
            train_dataset.imgs = np.expand_dims(train_dataset.imgs, axis=-1)

        if validation_dataset.imgs.ndim == 3:
            print("Adding channel to images...")
            validation_dataset.imgs = np.expand_dims(validation_dataset.imgs, axis=-1)
        
        if test_dataset.imgs.ndim == 3:
            print("Adding channel to images...")
            test_dataset.imgs = np.expand_dims(test_dataset.imgs, axis=-1)

        # Outputting the shapes of the datasets.
        print("Shapes of images")
        print(f"Training shape: {str(train_dataset.imgs.shape)}")
        print(f"Validation shape: {str(validation_dataset.imgs.shape)}")
        print(f"Testing shape: {str(test_dataset.imgs.shape)}")

        return train_dataset, validation_dataset, test_dataset
    
    except Exception as e:
        print(f"Downloading breastMNIST dataset failed. Error: {e}")



def preprocess_check(train_dataset, validation_dataset, test_dataset):
    """Pre-process check.

    This function checks if the datasets has no missing images.

    Args:
            training, validation and test datasets.

    Returns:
            N/A

    """

    if len(train_dataset.imgs) != 546 :
        print("Missing images detected in the training dataset")
        print(f"Found {len(train_dataset.imgs)}, should be 11959.")
    else:
        print("SUCCESS: No missing images in training dataset.")

    if len(validation_dataset.imgs) != 78:
        print("Missing images detected in the validation dataset")
        print(f"Found {len(validation_dataset.imgs)}, should be 1712.")
    else:
        print("SUCCESS: No missing images in validation dataset.")

    if len(test_dataset.imgs) != 156:
        print("Missing images detected in the test dataset")
        print(f"Found {len(test_dataset.imgs)}, should be 3421.")
    else:
        print("SUCCESS: No missing images in test dataset.")



def normalize_dataset(train_dataset, validation_dataset, test_dataset):
    """Normalize dataset.

    This function performs data transform via normalization.

    Args:
            training, validation and test datasets.

    Returns:
            normalized training, validation and test datasets.

    """

    try:

        # Performing data transformation via normalization
        train_dataset.imgs = train_dataset.imgs/255.0
        validation_dataset.imgs = validation_dataset.imgs/255.0
        test_dataset.imgs = test_dataset.imgs/255.0

        return train_dataset, validation_dataset, test_dataset
    
    except Exception as e:
        print(f"Data normalization failed. Error: {e}")



def visualise_subset(train_dataset):
    """Visualise Subset.

    This function visualises a subset of the training dataset images data.

    Args:
            training datasets.

    Returns:
            Images saved in figures folder.

    """

    try:

        plt.figure(figsize=(12,10))
        for x in range(9):
            value = 330+1+x
            plt.subplot(value)
            plt.imshow(train_dataset.imgs[x], cmap=plt.get_cmap('gray'))
            plt.title(f"Class {train_dataset.labels[x][0]}")

        plt.savefig("figures/subset_images.jpeg")
        plt.close()

    except Exception as e:
        print(f"Visualising subset failed. Error: {e}")



def descision_tree_training(x_train, x_val, x_test, y_train, y_val, y_test):
    """Visualise Subset.

    This function visualises a subset of the training dataset images data.

    Args:
            training datasets.

    Returns:
            Images saved in figures folder.

    """

    try:

        class_labels = ['Malignant', 'Benign']
        # Set the following as empty lists.
        accuracy_test_scores = []
        maximum_tree_depth = []
        maximum_No_features = []
        minimum_sample_number = []

        # Iterate through sample number, maxmimum features and maxmimum tree depth and output the one with highest accuracy
        for sample_number in range(2,31):
            for maximum_features in range(1,30):
                for maximum_depth in range(1,30):
                   
                    
                    classifier = DecisionTreeClassifier(min_samples_split=sample_number,max_features=maximum_features, max_depth=maximum_depth, criterion='entropy', random_state=0)
                    classifier.fit(x_train, y_train)
                    predict_y = classifier.predict(x_val)
                    accuracy_test = accuracy_score(y_val, predict_y)

                    # Store the values of hyper-parameter along side the accuracy into the list
                    accuracy_test_scores.append(accuracy_test)
                    maximum_No_features.append(maximum_features)
                    maximum_tree_depth.append(maximum_depth)
                    minimum_sample_number.append(sample_number)

        
        # Extract the highest accuracy from a list and output the hyper-parameters asscoiated with it
        highest_accuracy = max(accuracy_test_scores)
        highest_accuracy_index = accuracy_test_scores.index(highest_accuracy)
        maximum_depth = maximum_tree_depth[highest_accuracy_index]
        maximum_features = maximum_No_features[highest_accuracy_index]
        sample_number = minimum_sample_number[highest_accuracy_index]
        print(f"The highest accuracy is {highest_accuracy}, which was achieved via {maximum_depth} tree depth and {maximum_features} features with minimum sample split of {sample_number}")

        # Using those hyper-parameters, test the model on a test dataset
        classifier = DecisionTreeClassifier(min_samples_split=sample_number,max_features=maximum_features, max_depth=maximum_depth, criterion='entropy', random_state=0)
        classifier.fit(x_train, y_train)
        predict_y = classifier.predict(x_test)
        accuracy_test = accuracy_score(y_test, predict_y)
        print(f"The test accuracy is {accuracy_test}")

        # Evaluate the model
        y_test = np.asarray(y_test)
        y_train = np.asarray(y_train)
        y_true = np.argmax(y_test)
        evaluate_model(y_test, predict_y, class_labels)

        return accuracy_test_scores, maximum_tree_depth, maximum_No_features, minimum_sample_number, sample_number, maximum_depth, maximum_features, highest_accuracy
    
    except Exception as e:
        print(f"Decision tree model training failed. Error: {e}")



def evaluate_model(true_labels, predicted_labels, label_names):
    """Evaluate the decision tree model.

    This function evaluates the decision tree model and produces classification report and confusion matrix

    Args:
            true_labels
            predicted_labels
            label_names

    """

    try:

        # Calculates accuracry, precision, recall and f1 scores.
        print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
        print(f"Precision: {precision_score(true_labels, predicted_labels, average='weighted')}")
        print(f"Recall: {recall_score(true_labels, predicted_labels, average='weighted')}")
        print(f"F1 Score: {f1_score(true_labels, predicted_labels, average='weighted')}")

        # Performs classification report
        print("Classification report : ")
        print(classification_report(true_labels, predicted_labels, target_names=label_names))

        # Generates confusion matrix
        matrix = confusion_matrix(true_labels, predicted_labels)
        ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues)
        if label_names is not None:
            tick_marks = np.arange(len(label_names))
            plt.title("Decision Tree Confusion Matrix Display")
            plt.xticks(tick_marks, label_names)
            plt.yticks(tick_marks, label_names)
            plt.savefig('figures/Confusion_Matrix_decision_tree.png')
        plt.close()

    except Exception as e:
        print(f"Evaluating the decision tree model has failed. Error: {e}")