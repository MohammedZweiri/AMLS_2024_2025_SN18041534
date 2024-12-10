#========Import required libraries==================
import cv2 as cv
import os 
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import time

#==========this libraries for the models =============
from sklearn.ensemble import RandomForestClassifier
from PIL import ImageFont

 
#========== This libraries for getting the result of accurcy and confusion matrix of the model =======
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, ConfusionMatrixDisplay
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.tree import plot_tree

import medmnist
from medmnist import INFO, Evaluator

class_labels = []

def dataset_download(dataset_name):

    print(f"downloading the MedMNIST dataset. Source information: MedMNIST v{medmnist.__version__} @ {medmnist.HOMEPAGE} ")

    download = True
    
    dataset_info = INFO[dataset_name]

    #print(dataset_info)
    dataset_class = getattr(medmnist, dataset_info['python_class'])


    train_dataset = dataset_class(split='train', download=download)
    validation_dataset = dataset_class(split='val', download=download)
    test_dataset = dataset_class(split='test', download=download)
    print("Training Dataset")
    

    if train_dataset.imgs.ndim == 3:
        #print("Adding channel to images...")
        train_dataset.imgs = np.expand_dims(train_dataset.imgs, axis=-1)

    print("Validation Dataset")
    

    if validation_dataset.imgs.ndim == 3:
        #print("Adding channel to images...")
        validation_dataset.imgs = np.expand_dims(validation_dataset.imgs, axis=-1)

    print("Testing Dataset")
    

    if test_dataset.imgs.ndim == 3:
        #print("Adding channel to images...")
        test_dataset.imgs = np.expand_dims(test_dataset.imgs, axis=-1)

    print("Shapes of images")
    print(f"Training shape: {str(train_dataset.imgs.shape)}")
    print(f"Validation shape: {str(validation_dataset.imgs.shape)}")
    print(f"Testing shape: {str(test_dataset.imgs.shape)}")

    return train_dataset, validation_dataset, test_dataset

def preprocess_check(train_dataset, validation_dataset, test_dataset):
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
    train_dataset.imgs = train_dataset.imgs/255.0
    validation_dataset.imgs = validation_dataset.imgs/255.0
    test_dataset.imgs = test_dataset.imgs/255.0

    return train_dataset, validation_dataset, test_dataset

def visualise_subset(train_dataset):
    fig = plt.figure(figsize=(12,10))
    for x in range(9):
        value = 330+1+x
        plt.subplot(value)
        plt.imshow(train_dataset.imgs[x], cmap=plt.get_cmap('gray'))
        plt.title(f"Class {train_dataset.labels[x][0]}")
    
    plt.savefig("figures/subset_images.jpeg")
    plt.close()

def descision_tree_training(x_train, x_val, x_test, y_train, y_val, y_test):

    accuracy_test_scores = []
    maximum_tree_depth = []
    maximum_No_features = []
    minimum_sample_number = []

    for sample_number in range(2,31):
        for maximum_features in range(1,30):
            for maximum_depth in range(1,30):
                print("############# ############")
                print(f"The current iteration has the minimum sample split of {sample_number}, maximum features of {maximum_features} and tree depth of {maximum_depth}")
                
                classifier = DecisionTreeClassifier(min_samples_split=sample_number,max_features=maximum_features, max_depth=maximum_depth, criterion='entropy', random_state=0)
                classifier.fit(x_train, y_train)
                predict_y = classifier.predict(x_val)
                accuracy_test = accuracy_score(y_val, predict_y)

                print(f"training accuracy score {accuracy_test}")
                print("\n")
                
                accuracy_test_scores.append(accuracy_test)
                maximum_No_features.append(maximum_features)
                maximum_tree_depth.append(maximum_depth)
                minimum_sample_number.append(sample_number)
                #time.sleep(0.5)

    
    highest_accuracy = max(accuracy_test_scores)
    highest_accuracy_index = accuracy_test_scores.index(highest_accuracy)
    maximum_depth = maximum_tree_depth[highest_accuracy_index]
    maximum_features = maximum_No_features[highest_accuracy_index]
    sample_number = minimum_sample_number[highest_accuracy_index]

    # plot_accuray(maximum_No_features, maximum_tree_depth, accuracy_test_scores)
    print(f"The highest accuracy is {highest_accuracy}, which was achieved via {maximum_depth} tree depth and {maximum_features} features with minimum sample split of {sample_number}")

    classifier = DecisionTreeClassifier(min_samples_split=sample_number,max_features=maximum_features, max_depth=maximum_depth, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    predict_y = classifier.predict(x_test)
    accuracy_test = accuracy_score(y_test, predict_y)
    print(f"The test accuracy is {accuracy_test}")

    feature_importances = classifier.feature_importances_ 
    important_features = [(index, importance) for index, importance in enumerate(feature_importances) if importance > 0] 
    # Print the important features 
    print("Important features and their importance scores:") 
    feature_names = []
    for feature_index, importance in important_features: 
        
        print(f"Feature {feature_index}: {importance:.4f}") # To visualize the decision paths and features used in splits 
        feature_names.append(f"Feature {feature_index}")

    tree_rules = export_text(classifier) 
    #print(tree_rules)

    plt.figure()
    plt.bar(important_features,feature_importances)
    plt.xticks(rotation=45)
    plt.ylabel('feature importance')
    plt.savefig("decision_tree_features_importance.png")
   




    return accuracy_test_scores, maximum_tree_depth, maximum_No_features, minimum_sample_number, sample_number, maximum_depth, maximum_features, highest_accuracy

def random_forrest():


    return None

def evaluate_model(true_labels, predicted_labels, predict_probs, label_names):
    if(true_labels.ndim==2):
        true_labels = true_labels[:,0]
    if(predicted_labels.ndim==2):
        predicted_labels=predicted_labels[:,0]
    if(predict_probs.ndim==2):
        predict_probs=predict_probs[:,0]

    print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
    print(f"Precision: {precision_score(true_labels, predicted_labels, average='micro')}")
    print(f"Recall: {recall_score(true_labels, predicted_labels, average='micro')}")
    print(f"F1 Score: {f1_score(true_labels, predicted_labels, average='micro')}")

    print("Classification report : ")
    print(classification_report(true_labels, predicted_labels, target_names=label_names))

    matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix : ")

    ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues)
    if label_names is not None:
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)
        plt.savefig('figures/Confusion_Matrix_test1.png')

def visualise_tree(tree_to_print):
    plt.figure()
    fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (10,10), dpi=800)
    plot_tree(tree_to_print,
               feature_names = iris.feature_names,
               class_names=iris.target_names, 
               filled = True,
              rounded=True)
    plt.show()




if __name__=="__main__":


    dataset_name = "breastmnist"

    train_dataset, validation_dataset, test_dataset = dataset_download(dataset_name)

    preprocess_check(train_dataset, validation_dataset, test_dataset)

    train_dataset, validation_dataset, test_dataset = normalize_dataset(train_dataset, validation_dataset, test_dataset)

    visualise_subset(train_dataset)

    x_train = np.array(train_dataset.imgs).reshape(546, -1)
    x_val = np.array(validation_dataset.imgs).reshape(78, -1)
    x_test = np.array(test_dataset.imgs).reshape(156, -1)
    y_train = np.array(train_dataset.labels)
    y_val = np.array(validation_dataset.labels)
    y_test = np.array(test_dataset.labels)
    descision_tree_training(x_train, x_val, x_test, y_train, y_val, y_test)
    
    # save_model(model, "CNN_model_task1")

    # test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
    # test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
    # evaluate_model(test_dataset.labels, test_predict_labels, test_dataset_prob, class_labels)

    # plot_accuray_loss(history)