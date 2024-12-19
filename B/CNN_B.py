"""Accomplishing Task B via Convolutional Neural Networks.

    This module acquires BlooddMNIST data from medmnist library, then it uses the CNN model to accuractly predict the 8 different
    classes of the blood diseases.

    """

import cv2 as cv
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

#==========this libraries for the models =============
from tensorflow import keras
from keras.utils import to_categorical, plot_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from PIL import ImageFont

 
#========== This libraries for getting the result of accurcy and confusion matrix of the model =======
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path
from keras.models import model_from_json
from sklearn.utils import class_weight
import visualkeras
import medmnist
from medmnist import INFO, Evaluator

class_labels = ['basophil',
                'eosinophil',
                'erythroblast',
                'immature granulocytes',
                'lymphocyte',
                'monocyte',
                'neutrophil',
                'platelet']

def dataset_download(dataset_name):
    """Download dataset.

    This function downloads the BloodMNIST dataset from the medmnist library

    Args:
            dataset_name(str): The dataset name to be downloaded

    Returns:
            Training, validation and test datasets.

    """


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
        train_dataset.imgs = np.expand_dims(train_dataset.imgs, axis=1)
    

    if validation_dataset.imgs.ndim == 3:
        print("Adding channel to images...")
        validation_dataset.imgs = np.expand_dims(validation_dataset.imgs, axis=1)    

    if test_dataset.imgs.ndim == 3:
        print("Adding channel to images...")
        test_dataset.imgs = np.expand_dims(test_dataset.imgs, axis=1)

    # Outputting the shapes of the datasets
    print("Shapes of images")
    print(f"Training shape: {str(train_dataset.imgs.shape)}")
    print(f"Validation shape: {str(validation_dataset.imgs.shape)}")
    print(f"Testing shape: {str(test_dataset.imgs.shape)}")

    return train_dataset, validation_dataset, test_dataset



def preprocess_check(train_dataset, validation_dataset, test_dataset):
    """Pre-process check.

    This function checks if the datasets has no missing images.

    Args:
            training, validation and test datasets.

    Returns:
            N/A

    """

    if len(train_dataset.imgs) != 11959:
        print("Missing images detected in the training dataset")
        print(f"Found {len(train_dataset.imgs)}, should be 11959.")
    else:
        print("SUCCESS: No missing images in training dataset.")

    if len(validation_dataset.imgs) != 1712:
        print("Missing images detected in the validation dataset")
        print(f"Found {len(validation_dataset.imgs)}, should be 1712.")
    else:
        print("SUCCESS: No missing images in validation dataset.")

    if len(test_dataset.imgs) != 3421:
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

    # Performing data transformation via normalization
    train_dataset.imgs = train_dataset.imgs/255.0
    validation_dataset.imgs = validation_dataset.imgs/255.0
    test_dataset.imgs = test_dataset.imgs/255.0

    return train_dataset, validation_dataset, test_dataset


def save_model(model, model_name):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    # Convert the model structure into json
    model_structure = model.to_json()

    # Creates a json file and writes the json model structure
    file_path = Path(f"model/{model_name}.json")
    file_path.write_text(model_structure)

    # Saves the weights as .h5 file
    model.save_weights(f"{model_name}.weights.h5")


def load_model(model_name):
    """Save CNN model.

    This function loads the saved CNN model and weights to be used later on.

    Args:
            model_name(str)
            
    Returns:
            CNN model

    """

    # Locate the model structure file
    file_path = Path(f"model/{model_name}.json")

    # Read the json file and extract the CNN model
    model_structure = file_path.read_text()
    model = model_from_json(model_structure)

    # Load the CNN weights
    model.load_weights(f"{model_name}.weights.h5")

    return model


def evaluate_model(true_labels, predicted_labels, predict_probs, label_names):
    """Evaluate the CNN model.

    This function evaluates the CNN model and produces classification report and confusion matrix

    Args:
            true_labels
            predicted_labels
            predict_probs
            label_names

    """

    if(true_labels.ndim==2):
        true_labels = true_labels[:,0]
    if(predicted_labels.ndim==2):
        predicted_labels=predicted_labels[:,0]
    if(predict_probs.ndim==2):
        predict_probs=predict_probs[:,0]

    # Calculates accuracry, precision, recall and f1 scores.
    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
    print(f"Precision: {precision_score(true_labels, predicted_labels, average='micro')}")
    print(f"Recall: {recall_score(true_labels, predicted_labels, average='micro')}")
    print(f"F1 Score: {f1_score(true_labels, predicted_labels, average='micro')}")

    # Performs classification report
    print("Classification report : ")
    print(classification_report(true_labels, predicted_labels, target_names=label_names))

    # Generates confusion matrix
    matrix = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(10, 7), dpi=200)
    ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix for CNN")
    plt.savefig('figures/Confusion_Matrix_test1.png', bbox_inches = 'tight')


def plot_accuray_loss(model_history):
    """Plot accuracy loss graphs for the CNN model.

    This function plots the CNN model's accuracy and loss against epoch into a fig file.

    Args:
            model history

    """

    # Create the subplots variables.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,6), dpi=160)

    # Plot the accuracy subplot
    accuracy = model_history.history['accuracy']
    validation_accuracy = model_history.history['val_accuracy']
    epochs = range(1, len(accuracy)+1)
    ax1.plot(epochs, accuracy, label="Training Accuracy")
    ax1.plot(epochs, validation_accuracy, label="Validation Accuracy")
    ax1.set_title('Training and validation accuracy')
    ax1.set_xlabel('Number of Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid()

    # Plot the loss subplot
    loss = model_history.history['loss']
    val_loss = model_history.history['val_loss']
    ax2.plot(epochs, loss, label="Training loss")
    ax2.plot(epochs, val_loss, label="Validation loss")
    ax2.set_title('Training and validation loss')
    ax2.set_xlabel('Number of Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid()

    # Save the subplots file.
    fig.savefig('figures/CNN_accuracy_loss_test1.png')

def class_imbalance_handling(train_dataset):
    """Handling class imbalance

    This function performs classes weights, which is useful for the model to "pay more attention" to samples from an under-represented class.

    Args:
            training dataset
    
    Returns:

            Class Weights

    """

    # Computing class weights
    blood_class_weights = class_weight.compute_class_weight('balanced',
                                                         classes = np.unique(train_dataset.labels[:,0]),
                                                         y = train_dataset.labels[:, 0])

    # Link each weight to it's corresponding class.
    weights = {0 : blood_class_weights[0], 
            1 : blood_class_weights[1], 
            2 : blood_class_weights[2], 
            3 : blood_class_weights[3], 
            4 : blood_class_weights[4], 
            5 : blood_class_weights[5], 
            6 : blood_class_weights[6], 
            7 : blood_class_weights[7] }

    print(f"Class weights for imbalance {weights}")
    return weights

def CNN_model(train_dataset, validation_dataset, test_dataset):
    """CNN model testing

    This function loads the final CNN model and tests it on the test dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    # Categorise the labels into 8 classes
    train_labels = to_categorical(train_dataset.labels, num_classes=8)
    val_labels = to_categorical(validation_dataset.labels, num_classes=8)

    # model = Sequential()

    # model.add(Conv2D(32, (3,3), padding='same', input_shape=(28,28,3), activation="relu"))
    # model.add(Conv2D(32, (3,3), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(64, (3,3), padding='same', activation="relu"))
    # model.add(Conv2D(64, (3,3), activation="relu"))
    # model.add(MaxPooling2D(pool_size=(2,2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(512, activation="relu"))
    # model.add(Dropout(0.5))
    # model.add(Dense(8, activation="softmax"))

    # Load the CNN model
    model = load_model("CNN_model_taskB_final")

    # Output the model summary
    print(model.summary())

    # Plot the CNN model
    plot_model(model, 
               to_file='figures/CNN_Model_testB.png', 
               show_shapes=True,
                show_dtype=False,
                show_layer_names=False,
                rankdir="TB",
                expand_nested=False,
                dpi=200,
                show_layer_activations=True,
                show_trainable=False)

    # Compile the CNN model
    model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
    
    # Handle the class imbalance.
    weights = class_imbalance_handling(train_dataset)

    # Fit the CNN model
    history = model.fit(train_dataset.imgs, train_labels, 
              epochs=40,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
              validation_data=(validation_dataset.imgs, val_labels),
              batch_size=32,
            shuffle=True,
            class_weight=weights)
    
    #save_model(model, "CNN_model_taskB_final")

    # Evaluate the model
    test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
    test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
    evaluate_model(test_dataset.labels, test_predict_labels, test_dataset_prob, class_labels)
    plot_accuray_loss(history)