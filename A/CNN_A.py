"""Accomplishing Task A via Convolutional Neural Networks.

    This module acquires BreastMNIST data from medmnist library, then it uses the CNN model accuractly predict if the images are tumor or not.

    """


#========Import required libraries==================
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
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense, BatchNormalization, RandomFlip, RandomZoom, RandomRotation
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from PIL import ImageFont

 
#========== This libraries for getting the result of accurcy and confusion matrix of the model =======
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path
from sklearn.utils import shuffle
from keras.models import model_from_json
from sklearn.utils import class_weight
import visualkeras
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

        # Outputting the shapes of the datasets
        print("Shapes of images")
        print(f"Training shape: {str(train_dataset.imgs.shape)}")
        print(f"Validation shape: {str(validation_dataset.imgs.shape)}")
        print(f"Testing shape: {str(test_dataset.imgs.shape)}")

        return train_dataset, validation_dataset, test_dataset
    
    except Exception as e:
        print(f"Downloading BreastMNIST dataset failed. Error: {e}")

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


def save_model(model, model_name):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    try:

        # Convert the model structure into json
        model_structure = model.to_json()

        # Creates a json file and writes the json model structure
        file_path = Path(f"model/{model_name}.json")
        file_path.write_text(model_structure)

        # Saves the weights as .h5 file
        model.save_weights(f"{model_name}.weights.h5")

    except Exception as e:
        print(f"Saving the CNN model failed. Error: {e}")


def load_model(model_name):
    """Save CNN model.

    This function loads the saved CNN model and weights to be used later on.

    Args:
            model_name(str)
            
    Returns:
            CNN model

    """

    try:
        
        # Locate the model structure file
        file_path = Path(f"model/{model_name}.json")

        # Read the json file and extract the CNN model
        model_structure = file_path.read_text()
        model = model_from_json(model_structure)

        # Load the CNN weights
        model.load_weights(f"{model_name}.weights.h5")

        return model
    
    except Exception as e:
        print(f"Loading the CNN model failed. Error: {e}")


def evaluate_model(true_labels, predict_probs, label_names):
    """Evaluate the CNN model.

    This function evaluates the CNN model and produces classification report and confusion matrix

    Args:
            true_labels
            predict_probs
            label_names

    """
    try:

        # Calculates accuracry, precision, recall and f1 scores.
        print(f"Accuracy: {accuracy_score(true_labels, predict_probs.round())}")
        print(f"Precision: {precision_score(true_labels, predict_probs.round(), average='weighted')}")
        print(f"Recall: {recall_score(true_labels, predict_probs.round(), average='weighted')}")
        print(f"F1 Score: {f1_score(true_labels, predict_probs.round(), average='weighted')}")

        # Performs classification report
        print("Classification report : ")
        print(classification_report(true_labels, predict_probs.round(), target_names=label_names))

        # Generates confusion matrix
        matrix = confusion_matrix(true_labels, predict_probs.round())
        ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues)
        if label_names is not None:
            tick_marks = np.arange(len(label_names))
            plt.xticks(tick_marks, label_names)
            plt.yticks(tick_marks, label_names)
            plt.savefig('figures/Confusion_Matrix_test1.png')

    except Exception as e:
        print(f"Evaluating the model has failed. Error: {e}")



def plot_accuray_loss(model_history):
    """Plot accuracy loss graphs for the CNN model.

    This function plots the CNN model's accuracy and loss against epoch into a fig file.

    Args:
            model history

    """

    try:

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
    
    except Exception as e:
        print(f"Plotting accuracy and loss has failed. Error: {e}")


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

def CNN_model(train_dataset, validation_dataset, test_dataset):
    """CNN model testing

    This function loads the final CNN model and tests it on the test dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:

        # Class labels
        class_labels = ['Benign','Malignant']

        # Convert to numpy array
        train_imgs = np.array(train_dataset.imgs)
        train_labels = np.array(train_dataset.labels)
        val_imgs = np.array(validation_dataset.imgs)
        val_labels = np.array(validation_dataset.labels)

        data_augmentation = Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.2),
            RandomZoom(0.2),
        ])

        model = Sequential()
        model.add(data_augmentation)
        model.add(Conv2D(128, (6,6), padding='same', input_shape=(28,28,1), activation="relu"))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(96, (6,6), kernel_initializer='he_uniform', activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))

        model.build(input_shape=(None, 28, 28, 1))

        # Output the model summary
        print(model.summary())

        # Plot the CNN model
        plot_model(model, 
                to_file='figures/CNN_Model_test2.png', 
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
                optimizer=Adam(learning_rate=0.0001),
                metrics=['accuracy'])
        
        # Handle the class imbalance.
        weights = class_imbalance_handling(train_dataset)

        # Fit the CNN model
        history = model.fit(train_imgs, train_labels,
                epochs=100,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)],
                validation_data=(val_imgs, val_labels),
                batch_size=64,
                shuffle=True,
                class_weight=weights)
        
        save_model(model, "CNN_model_taskB")

        # Evaluate the model
        test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
        #test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
        evaluate_model(test_dataset.labels, test_dataset_prob, class_labels)
        plot_accuray_loss(history)

    except Exception as e:
        print(f"Running the CNN model failed. Error: {e}")