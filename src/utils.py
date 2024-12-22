""" Provide python functions which can be used by multiple python files


"""

import medmnist
from medmnist import INFO
import numpy as np
from pathlib import Path
from keras.models import model_from_json
import matplotlib.pyplot as plt
import os


def create_directory(directory):
    """Create directory under the current path

    Args:
        directory: String formatted directory name
    
    """
    try:

        # Get the current path
        current_path = os.getcwd()

        # Merge the current path with the desired one
        path = os.path.join(current_path, directory)

        # If the directory exists, do nothing. Otherwise, create it
        if os.path.isdir(path):
            return
        else:
            os.mkdir(path)

    except Exception as e:
        print(f"Creating directory failed. Error: {e}")



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
        print(f"Downloading {dataset_name} dataset failed. Error: {e}")



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



def save_model(task_name,model, model_name):
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
        file_path = Path(f"{task_name}/model/{model_name}.json")
        file_path.write_text(model_structure)

        # Saves the weights as .h5 file
        model.save_weights(f"{task_name}/{model_name}.weights.h5")

    except Exception as e:
        print(f"Saving the CNN model failed. Error: {e}")



def load_model(task_name, model_name):
    """Save CNN model.

    This function loads the saved CNN model and weights to be used later on.

    Args:
            model_name(str)
            
    Returns:
            CNN model

    """

    try:
        
        # Locate the model structure file
        file_path = Path(f"{task_name}/model/{model_name}.json")

        # Read the json file and extract the CNN model
        model_structure = file_path.read_text()
        model = model_from_json(model_structure)

        # Load the CNN weights
        model.load_weights(f"{task_name}/{model_name}.weights.h5")

        return model
    
    except Exception as e:
        print(f"Loading the CNN model failed. Error: {e}")



def plot_accuray_loss(task_name, model_history):
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
        fig.savefig(f'{task_name}/figures/CNN_accuracy_loss_test1.png')
    
    except Exception as e:
        print(f"Plotting accuracy and loss has failed. Error: {e}")


def visualise_subset(task_name, train_dataset):
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

        plt.savefig(f"{task_name}/figures/subset_images.jpeg")
        plt.close()

    except Exception as e:
        print(f"Visualising subset failed. Error: {e}")