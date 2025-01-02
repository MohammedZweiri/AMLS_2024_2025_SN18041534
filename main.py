""" main.py is a centralised module where the process for both tasks initiates


"""
from A import CNN_A, decision_tree
from B import CNN_B
from src import utils
import argparse
import numpy as np


def Task_A_DT():
    """ Runs the decision tree training model for the breastMNIST dataset


"""

    print("Task A via decision tree has started...")
    # Download the dataset    
    dataset_name = "breastmnist"
    train_dataset, validation_dataset, test_dataset = utils.dataset_download(dataset_name)

    # Perform preprocess check
    decision_tree.preprocess_check(train_dataset, validation_dataset, test_dataset)

    # Transform data using normalization
    train_dataset, validation_dataset, test_dataset = utils.normalize_dataset(train_dataset, validation_dataset, test_dataset)

    # visualise a subset of the dataset
    utils.visualise_subset("A",train_dataset)

    # Reshape
    x_train = np.array(train_dataset.imgs).reshape(546, -1)
    x_val = np.array(validation_dataset.imgs).reshape(78, -1)
    x_test = np.array(test_dataset.imgs).reshape(156, -1)

    # Convert the labels into numpy array
    y_train = np.array(train_dataset.labels)
    y_val = np.array(validation_dataset.labels)
    y_test = np.array(test_dataset.labels)

    # Perform the decision tree training
    decision_tree.descision_tree_training(x_train, x_val, x_test, y_train, y_val, y_test)

    print("Task A via decision tree has finished...")


def Task_A_CNN():

    """ Runs the CNN model for breastMNIST dataset


    """ 
    print("Task A via CNN is starting...")

    # Download the dataset
    dataset_name = "breastmnist"
    train_dataset, validation_dataset, test_dataset = utils.dataset_download(dataset_name)

    # Perform preprocess check
    CNN_A.preprocess_check(train_dataset, validation_dataset, test_dataset)

    # Transform data using normalization
    train_dataset, validation_dataset, test_dataset = utils.normalize_dataset(train_dataset, validation_dataset, test_dataset)

    # visualise a subset of the dataset
    utils.visualise_subset("A",train_dataset)

    # Run the CNN model
    CNN_A.CNN_model_testing(test_dataset)

    print("Task A via CNN has finished...")


def Task_B_CNN():
    """ Runs the CNN model for bloodMNIST dataset


    """

    print("Task B via CNN is starting...")

    # Download the dataset
    dataset_name = "bloodmnist"
    train_dataset, validation_dataset, test_dataset = utils.dataset_download(dataset_name)

    # Perform preprocess check
    CNN_B.preprocess_check(train_dataset, validation_dataset, test_dataset)

    # Transform data using normalization
    train_dataset, validation_dataset, test_dataset = utils.normalize_dataset(train_dataset, validation_dataset, test_dataset)

    # visualise a subset of the dataset
    utils.visualise_subset("B", train_dataset)

    # Run the CNN model
    CNN_B.CNN_model_testing(test_dataset)

    print("Task B via CNN has finished...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='all',
                        help ='select the task')
    args = parser.parse_args()

    # Creating the figures and models folders under each task folder
    utils.create_directory("./A/figures")
    utils.create_directory("./A/model")
    utils.create_directory("./B/figures")
    utils.create_directory("./B/model")

    # Create Datasets folder
    utils.create_directory("Datasets")

    
    # Run the required functions depending on user's input
    
    if args.task == 'task_a':
        Task_A_DT()
        Task_A_CNN()

    elif args.task == 'task_b':
        Task_B_CNN()

    elif args.task == 'all':
        Task_A_DT()
        Task_A_CNN()
        Task_B_CNN()

    






