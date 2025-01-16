# ELEC0134 AMLS assingment 2024/2025

There two tasks for this assignment:
1. Task A which is performing the binary classification on the BreastMNIST dataset.
2. Task B which is performing the multi-class classification on the BloodMNIST dataset.

## What are the folders and files
1. Main folder contains main.py file, which runs other instances of python scripts.
2. Folder `A` contains two python files for Task A:
   - decision_tree.py, which runs the decision tree model on BreastMNIST.
   - CNN_A.py, which runs the CNN model on the BreastMNIST.
3. Folder `B` contains two python files for Task B:
   - CNN_B.py, which runs the CNN model on the BloodMNIST.
4. Both folders `A` & `B` contain `model` folder which contains CNN model files needed when running the scripts.
5. `src` folder contain `utils.py` script. This file is called by all ML model scripts in this assignment to perform a centralized tasks, such as: 
   - Downloading BreastMNIST and BloodMNIST dataset.
   - Split data into train, validation and testing datasets.
   - Perform pre-process check.
   - Normalize the dataset.
   - output a subset of a dataset.
   - plot accuracy and loss graphs.
   - Save CNN model as json and .weight.h5 file
   - Load CNN model.
   - Creating directories.

## Important note before the procedure.
1. `main.py` has two arguments set.
   - `task`, which the user can define what tasks they wish to run. You can run either A using `-t task_a` argument, or B using `-t task_b` or both by providing no input. If no input was provided, then the default is set to `all`, which runs both tasks consecutively.
   - `decision`, which the user can define how to run the models. You can either run on the training, validation and test dataset (training the model from scratch) using `-d train` or test the model using test dataset by adding no input. The default is set to `test`.
2. `Datasets` folder does not exist in the repository. However, running the `main.py` script runs through creating that folder and then adding data into it shortly.

  
## Procedures

1. You should be able to see a `requirements.txt` file, which contains all the libraries needed for this assignment. To install all the libraries from it, simply use `pip install -r requirements.txt` from the command line.

2. Once installed, you can start running the tasks. There multiple ways to do it.
    - If you want to run all the models for all the tasks as running models on the test datasets only, then run `python main.py`
    - If you want to run all the models for all the tasks as performing the entire training and validation process , then run `python main.py -d train`
    - If you want to run the entirety of task A as running models on the test datasets, then run `python main.py -t task_a`
    - If you want to run the entirety of task A as running models on the training and validation datasets, then run `python main.py -t task_a -d train`
    - If you want to run the entirety of task B as running models on the test datasets, then run `python main.py -t task_b`
    - If you want to run the entirety of task B as running models on the training and validation datasets, then run `python main.py -t task_b -d train`