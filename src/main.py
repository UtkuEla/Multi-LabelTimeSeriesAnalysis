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
from datetime import datetime
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
        self.values = pd.read_pickle(self.data_path)
        self.labels = pd.read_pickle(self.label_path)

        self.values = self.values.values
        self.labels = self.labels.values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.values, self.labels, test_size=0.2, random_state=42)

        self.X_train = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        self.X_test = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        print("X_train shape:", self.X_train.shape)
        print("X_test shape:", self.X_test.shape)
        print("y_train shape:", self.y_train.shape)
        print("y_test shape:", self.y_test.shape)

        self.nb_classes = len(np.unique(np.concatenate((self.y_train, self.y_test), axis=0)))
        self.input_shape = self.X_train.shape[1:]

        print("Number of Classes:", self.nb_classes)
        print("Input Shape:", self.input_shape)

    def preprocess_labels(self):
        enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
        enc.fit(np.concatenate((self.y_train, self.y_test), axis=0).reshape(-1, 1))
        self.y_train = enc.transform(self.y_train.reshape(-1, 1)).toarray()
        self.y_test = enc.transform(self.y_test.reshape(-1, 1)).toarray()

    def import_model(self,modelName):
        from models import Models, ParallelModels
        models = ParallelModels(self.input_shape, self.nb_classes)
        model_function = getattr(models,modelName)
        self.model = model_function()
        print(self.model.summary())

    def train_model(self, batch_size=32, nb_epochs=500):
        
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

        hist = self.model.fit(self.X_train, self.y_train, batch_size=batch_size, epochs=nb_epochs,
                              validation_data=(self.X_test, self.y_test), callbacks=[early_stopping])

    def save_model(self,model_path, modelName):
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%H-%M")
        model_filename = model_path + '/' + modelName + '_' + f"_{timestamp}.h5"
        self.model.save(model_filename)
        print(f"Model saved as {model_filename}")

    def load_trained_model(self, model_path, modelName):
        path = os.path.join(model_path, modelName)
        self.model = tf.keras.models.load_model(path)


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

            sorted_indices = np.argsort(row)[::-1]

            if row[sorted_indices[0]] >= threshold:
                prediction1 = sorted_indices[0]
            else:
                prediction1 = 3

            if row[sorted_indices[1]] >= threshold:
                prediction2 = sorted_indices[1]
            else:
                prediction2 = 3

            prediction_list.append([prediction1, prediction2])

        prediction_list = np.array(prediction_list)
        labels = ['Bias', 'Drift', 'Gain', 'NoFault', 'Outliers', 'Precisiondegradation']
        label_encoder = LabelEncoder()
        label_encoder.fit(labels)
        predicted_labels_1 = label_encoder.inverse_transform(prediction_list[:, 0])
        predicted_labels_2 = label_encoder.inverse_transform(prediction_list[:, 1])
        df_predictions = pd.DataFrame({'Prediction': predicted_labels_1, 'Prediction2': predicted_labels_2})
        
        now = datetime.now()
        timestamp = now.strftime("%d-%m-%H-%M")
        prediction_name = model_path + modelName + f"_{timestamp}" + "_predictions.xlsx"
        print(df_predictions)
        df_predictions.to_excel(prediction_name)


modelName = 'P_CNN_RNN_1'

current_folder_path = os.path.dirname(os.path.abspath(__file__))
parent_folder_path = os.path.dirname(current_folder_path)
data_save_path = os.path.join(parent_folder_path, 'data')
model_path = os.path.join(parent_folder_path,'models')

train_data_path = os.path.join(parent_folder_path,'train_data', 'train_data_values_300_10')
train_label_path = os.path.join(parent_folder_path,'train_data' ,'train_labels_300_10')

test_data_path = os.path.join(parent_folder_path,'test_data', 'test_data_values_300_10')
test_label_path = os.path.join(parent_folder_path,'test_data' ,'test_labels_300_10')


# print("Current Folder Path:", current_folder_path)
# print("Parent Folder Path:", parent_folder_path)
# print("Data Save Path:", data_save_path)
# print("Model Path:", model_path)
# print("Test Data Path:", test_data_path)
# print("Test Label Path:", test_label_path)

# Create an instance of the ModelTraining class
model_trainer = ModelTraining(train_data_path, train_label_path)

# Load and preprocess the data
model_trainer.load_data()
model_trainer.preprocess_labels()

# Build and train the model
model_trainer.import_model(modelName)
model_trainer.train_model(32,5)

# Save the trained model
model_trainer.save_model(model_path,modelName)

# Load the saved model
# trained_modelName = ""
# model_trainer.load_trained_model(model_path, trained_modelName)

# Load test data and make predictions
model_trainer.load_test_data(test_data_path, test_label_path)
model_trainer.process_predictions()
