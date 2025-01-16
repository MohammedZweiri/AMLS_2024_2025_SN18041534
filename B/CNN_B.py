"""Accomplishing Task B via Convolutional Neural Networks.

    This module acquires BlooddMNIST data from medmnist library, then it uses the CNN model to accuractly predict the 8 different
    classes of the blood diseases.

    """

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical, plot_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras import Input
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
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



def evaluate_model(true_labels, predicted_labels, predict_probs, label_names):
    """Evaluate the CNN model.

    This function evaluates the CNN model and produces classification report and confusion matrix

    Args:
            true_labels
            predicted_labels
            predict_probs
            label_names

    """

    try:

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
        plt.savefig('B/figures/Confusion_Matrix_CNN.png', bbox_inches = 'tight')

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
    
    except Exception as e:
        print(f"Class imbalance handling has failed. Error: {e}")

def CNN_model_training(train_dataset, validation_dataset, test_dataset):
    """CNN model training

    This function trains the CNN model and tests it on the dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:
            
        # Class labels
        class_labels = ['basophil',
                'eosinophil',
                'erythroblast',
                'immature granulocytes',
                'lymphocyte',
                'monocyte',
                'neutrophil',
                'platelet']
        

        # Categorise the labels into 8 classes
        train_labels = to_categorical(train_dataset.labels, num_classes=8)
        val_labels = to_categorical(validation_dataset.labels, num_classes=8)

        # CNN model
        model = Sequential()
        model.add(Input(shape=(28,28,3)))
        model.add(Conv2D(32, (3,3), padding='same', activation="relu"))
        model.add(Conv2D(32, (3,3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3,3), padding='same', activation="relu"))
        model.add(Conv2D(64, (3,3), activation="relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(8, activation="softmax"))

        # Output the model summary
        print(model.summary())

        # Plot the CNN model
        plot_model(model, 
                to_file='B/figures/CNN_Model_testB_add.png', 
                show_shapes=True,
                show_layer_activations=True)

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
        
        # save the CNN model
        utils.save_model("B",model, "CNN_model_taskB_final_add")

        # Evaluate the model
        test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
        test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
        evaluate_model(test_dataset.labels, test_predict_labels, test_dataset_prob, class_labels)
        utils.plot_accuray_loss("B",history)

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
        class_labels = ['basophil',
                'eosinophil',
                'erythroblast',
                'immature granulocytes',
                'lymphocyte',
                'monocyte',
                'neutrophil',
                'platelet']

        # Load the CNN model
        model = utils.load_model("B","CNN_model_taskB_final")

        # Output the model summary
        print(model.summary())

        # Evaluate the model
        test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
        test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
        evaluate_model(test_dataset.labels, test_predict_labels, test_dataset_prob, class_labels)

    except Exception as e:
        print(f"Loading and testing the CNN model failed. Error: {e}")