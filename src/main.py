import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, concatenate, GlobalMaxPooling1D, Dense, LSTM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder

import os
from pathlib import Path

class ModelTraining:
    def __init__(self, data_path, label_path):
        self.data_path = data_path
        self.label_path = label_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.enc = None
        self.nb_classes = None
        self.input_shape = None

    def load_data(self):
        values = pd.read_pickle(self.data_path)
        labels = pd.read_pickle(self.label_path)

        self.labels = labels.values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(values, self.labels, test_size=0.2, random_state=42)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_train shape:", self.y_train.shape)
        print("y_test shape:", self.y_test.shape)

        self.nb_classes = len(np.unique(np.concatenate((self.y_train, self.y_test), axis=0)))
        self.input_shape = self.X_train.shape[1:]

    def preprocess_labels(self):
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((self.y_train, self.y_test), axis=0).reshape(-1, 1))
        self.y_train = enc.transform(self.y_train.reshape(-1, 1)).toarray()
        self.y_test = enc.transform(self.y_test.reshape(-1, 1)).toarray()

    def build_model(self):
        input_layer = Input(shape=self.input_shape)

        conv_branch1 = Conv1D(filters=128, kernel_size=3, activation='relu')(input_layer)

        # Convolutional branch
        conv_branch = Conv1D(filters=64, kernel_size=3, activation='relu')(conv_branch1)
        conv_branch = MaxPooling1D(pool_size=2)(conv_branch)

        # First recurrent branch
        rnn_branch1 = LSTM(units=64, return_sequences=True)(conv_branch1)

        # Second recurrent branch
        rnn_branch2 = LSTM(units=64, return_sequences=True)(input_layer)

        # Apply global max pooling to the convolutional branch
        conv_branch = GlobalMaxPooling1D()(conv_branch)

        # Apply global max pooling to the recurrent branches
        rnn_branch1 = GlobalMaxPooling1D()(rnn_branch1)
        rnn_branch2 = GlobalMaxPooling1D()(rnn_branch2)

        # Concatenate the outputs of all branches
        concatenated = concatenate([conv_branch, rnn_branch1, rnn_branch2])

        # Classification layers
        dense_layer = Dense(units=128, activation='relu')(concatenated)
        output_layer = Dense(units=self.nb_classes, activation='softmax')(dense_layer)

        # Create the model
        self.model = Model(inputs=input_layer, outputs=output_layer)

        # Compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train_model(self, batch_size=32, nb_epochs=500):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        hist = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=nb_epochs,
                              validation_data=(self.X_test, self.y_test), callbacks=[early_stopping])

    def save_model(self, model_path):
        self.model.save(model_path)

    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def load_test_data(self, test_data_path, test_label_path):
        test_values = pd.read_pickle(test_data_path)
        test_labels = pd.read_pickle(test_label_path)
        test_labels.to_excel("test_label_300_10.xlsx")

        test_values = test_values.values

        test_values = test_values.reshape((test_values.shape[0], test_values.shape[1], 1))

        self.predictions = self.model.predict(test_values)

    def process_predictions(self, threshold=0.0001):
        prediction_list = []

        # Iterate over each row of predictions
        for row in self.predictions:
            # Sort the probabilities in descending order and get the indices
            sorted_indices = np.argsort(row)[::-1]

            # Check if the highest prediction is above the threshold
            if row[sorted_indices[0]] >= threshold:
                prediction1 = sorted_indices[0]
            else:
                prediction1 = 3

            # Check if the second highest prediction is above the threshold
            if row[sorted_indices[1]] >= threshold:
                prediction2 = sorted_indices[1]
            else:
                prediction2 = 3

            # Add the predictions to the list
            prediction_list.append([prediction1, prediction2])

        prediction_list = np.array(prediction_list)
        labels = ['Bias', 'Drift', 'Gain', 'NoFault', 'Outliers', 'Precisiondegradation']
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        predicted_labels_1 = label_encoder.inverse_transform(prediction_list[:, 0])
        predicted_labels_2 = label_encoder.inverse_transform(prediction_list[:, 1])
        df_predictions = pd.DataFrame({'Prediction': predicted_labels_1, 'Prediction2': predicted_labels_2})

        # Display the resulting DataFrame
        print(df_predictions)
        df_predictions.to_excel('predictions4.xlsx')


# Usage example:
data_path = os.path.join(parent_folder_path, 'train_data_values_300_10')
label_path = os.path.join(parent_folder_path, 'train_labels_300_10')

test_data_path = 'test_data_values_300_10'
test_label_path = 'test_labels_300_10'

model_path = 'model_parallel_azad.hdf5'

# Create an instance of the ModelTraining class
model_trainer = ModelTraining(data_path, label_path)

# Load and preprocess the data
model_trainer.load_data()
model_trainer.preprocess_labels()

# Build and train the model
model_trainer.build_model()
model_trainer.train_model()

# Save the trained model
model_trainer.save_model(model_path)

# Load the saved model
model_trainer.load_model(model_path)

# Load test data and make predictions
model_trainer.load_test_data(test_data_path, test_label_path)
model_trainer.process_predictions()
