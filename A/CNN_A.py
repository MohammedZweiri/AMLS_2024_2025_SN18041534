"""Accomplishing Task A via Convolutional Neural Networks.

    This module acquires BreastMNIST data from medmnist library, then it uses the CNN model accuractly predict if the images are tumor or not.

    """


#========Import required libraries==================
import cv2 as cv
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


#==========this libraries for the models =============
from tensorflow import keras
from keras.utils import plot_model
import tensorflow as tf
import torch
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
 
#========== This libraries for getting the result of accurcy and confusion matrix of the model =======
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from keras import Input
from sklearn.utils import class_weight
from src import utils


def preprocess_check(train_dataset, validation_dataset, test_dataset):
    """Pre-process check.

    This function checks if the datasets has no missing images.

    Args:
            training, validation and test datasets.

    Returns:
            N/A

    """

    if len(train_dataset.imgs) != 546:
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



def evaluate_model(true_labels, predicted_labels, predicted_probs, label_names):
    """Evaluate the CNN model.

    This function evaluates the CNN model and produces classification report and confusion matrix

    Args:
            true_labels
            predict_probs
            label_names

    """
    try:

        true_labels_1D = true_labels[:, 0]
        predicted_labels_1D = predicted_labels[:, 0]
        predicted_probs_1D = predicted_probs[:, 0]
        # Calculates accuracry, precision, recall and f1 scores.
        print(f"Accuracy: {accuracy_score(true_labels_1D, predicted_labels_1D)}")
        print(f"Precision: {precision_score(true_labels_1D, predicted_labels_1D)}")
        print(f"Recall: {recall_score(true_labels_1D, predicted_labels_1D)}")
        print(f"F1 Score: {f1_score(true_labels_1D, predicted_labels_1D)}")

        # Performs classification report
        print("Classification report : ")
        print(classification_report(true_labels_1D, predicted_labels_1D, target_names=label_names))

        # Generates confusion matrix
        matrix = confusion_matrix(true_labels, predicted_labels_1D)
        ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues)
        if label_names is not None:
            tick_marks = np.arange(len(label_names))
            plt.title("CNN Confusion Matrix Display")
            plt.xticks(tick_marks, label_names)
            plt.yticks(tick_marks, label_names)
            plt.savefig('A/figures/Confusion_Matrix_CNN.png')

    except Exception as e:
        print(f"Evaluating the model has failed. Error: {e}")



def class_imbalance_handling(train_dataset):
    """Handling class imbalance

    This function performs classes weights, which is useful for the model to "pay more attention" to samples from an under-represented class.

    Args:
            training dataset
    
    Returns:

            Class Weights

    """
    
    try:

        # Computing class weights
        breast_class_weights = class_weight.compute_class_weight('balanced',
                                                                classes = np.unique(train_dataset.labels[:,0]),
                                                                y = train_dataset.labels[:, 0])
        # Link each weight to it's corresponding class.
        weights = {0 : breast_class_weights[0], 
                1 : breast_class_weights[1]}

        print(f"Class weights for imbalance {weights}")
        return weights
    
    except Exception as e:
        print(f"Class imbalance handling has failed. Error: {e}")


def CNN_model_training(train_dataset, validation_dataset, test_dataset):
    """CNN model training

    This function traings the CNN models and tests it on the test dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:

        # Class labels
        class_labels = ['Malignant','Benign']

        # Convert to numpy array
        train_imgs = np.array(train_dataset.imgs)
        train_labels = np.array(train_dataset.labels)
        val_imgs = np.array(validation_dataset.imgs)
        val_labels = np.array(validation_dataset.labels)
 
        # CNN model
        model = Sequential()
        model.add(Input(shape=(28,28,1)))
        model.add(Conv2D(32, (3,3), padding='same', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(1,1), padding='same'))
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))


        # Output the model summary
        print(model.summary())

        # Plot the CNN model
        plot_model(model, 
                to_file='A/figures/CNN_Model_taskA_add.png', 
                show_shapes=True,
                    show_dtype=False,
                    show_layer_names=False,
                    rankdir="TB",
                    expand_nested=False,
                    dpi=200,
                    show_layer_activations=True,
                    show_trainable=False)

        # Compile the CNN model
        model.compile(loss='binary_crossentropy',
                optimizer=Adam(learning_rate=0.0005),
                metrics=['accuracy'])
        
        # Handle the class imbalance.
        weights = class_imbalance_handling(train_dataset)

        # Fit the CNN model
        history = model.fit(train_imgs, train_labels, 
                            batch_size=32,
                            epochs=30,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                            validation_data=(val_imgs, val_labels),
                            shuffle=True,
                            class_weight=weights)
        
        utils.save_model("A",model, "CNN_model_taskA_add")

        # Evaluate the model
        test_predict_prob = model.predict(test_dataset.imgs, verbose=0)
        test_predict_labels = np.where(test_predict_prob > 0.5, 1, 0)
        evaluate_model(test_dataset.labels, test_predict_labels, test_predict_prob, class_labels)
        utils.plot_accuray_loss("A",history)

    except Exception as e:
        print(f"Training and saving the CNN model failed. Error: {e}")


def CNN_model_testing(test_dataset):
    """CNN model testing

    This function loads the final CNN model and tests it on the test dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:

        # Class labels
        class_labels = ['Malignant','Benign']

        # load the CNN model
        model = utils.load_model("A", "CNN_model_taskA_final")

        # Output the model summary
        print(model.summary())

        # Evaluate the model
        test_predict_prob = model.predict(test_dataset.imgs, verbose=0)
        test_predict_labels = np.where(test_predict_prob > 0.5, 1, 0)
        evaluate_model(test_dataset.labels, test_predict_labels, test_predict_prob, class_labels)
        

    except Exception as e:
        print(f"Loading and testing the CNN model failed. Error: {e}")