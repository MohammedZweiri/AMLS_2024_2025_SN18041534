from A import CNN_A, decision_tree
from B import CNN_B
from src import utils
import argparse
import numpy as np


def Task_A_DT():

    # Download the dataset    
    dataset_name = "breastmnist"
    train_dataset, validation_dataset, test_dataset = utils.dataset_download(dataset_name)

    # Perform preprocess check
    decision_tree.preprocess_check(train_dataset, validation_dataset, test_dataset)

    # Transform data using normalization
    train_dataset, validation_dataset, test_dataset = utils.normalize_dataset(train_dataset, validation_dataset, test_dataset)

    # visualise a subset of the dataset
    utils.visualise_subset(train_dataset)

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

    return None

def Task_A_CNN():

    # Download the dataset
    dataset_name = "breastmnist"
    train_dataset, validation_dataset, test_dataset = utils.dataset_download(dataset_name)

    # Perform preprocess check
    CNN_A.preprocess_check(train_dataset, validation_dataset, test_dataset)

    # Transform data using normalization
    train_dataset, validation_dataset, test_dataset = utils.normalize_dataset(train_dataset, validation_dataset, test_dataset)

    # visualise a subset of the dataset
    utils.visualise_subset(train_dataset)

    # Run the CNN model
    CNN_A.CNN_model(train_dataset, validation_dataset, test_dataset)


def Task_B_CNN():

    # Download the dataset
    dataset_name = "bloodmnist"
    train_dataset, validation_dataset, test_dataset = utils.dataset_download(dataset_name)

    # Perform preprocess check
    CNN_B.preprocess_check(train_dataset, validation_dataset, test_dataset)

    # Transform data using normalization
    train_dataset, validation_dataset, test_dataset = utils.normalize_dataset(train_dataset, validation_dataset, test_dataset)

    # visualise a subset of the dataset
    utils.visualise_subset(train_dataset)

    # Run the CNN model
    CNN_B.CNN_model(train_dataset, validation_dataset, test_dataset)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', default='all',
                        help ='select the task')
    args = parser.parse_args()

    if args.t == 'task_a':
        Task_A_DT()
        Task_A_CNN()

    elif args.t == 'task_b':
        Task_B_CNN()

    elif args.t == 'all':
        Task_A_DT()
        Task_A_CNN()
        Task_B_CNN()

    






