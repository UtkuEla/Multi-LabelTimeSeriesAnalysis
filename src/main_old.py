import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, concatenate, GlobalMaxPooling1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Rescaling
from tensorflow.keras import Model
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam

current_folder_path = os.getcwd()
parent_folder_path = os.path.dirname(current_folder_path)
data_save_path = os.path.join(parent_folder_path, 'data')

path = parent_folder_path + "\\train_data_values_300_10"
values= pd.read_pickle(path)
path = parent_folder_path + "\\train_labels_300_10"
labels= pd.read_pickle(path)

labels = labels.values

X_train, X_test, y_train, y_test = train_test_split(values,labels, test_size=0.2, random_state=42)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

input_shape = X_train.shape[1:]

enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, LSTM, Dense, GlobalMaxPooling1D, concatenate

input_layer = Input(shape=input_shape)

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
output_layer = Dense(units=nb_classes, activation='softmax')(dense_layer)

# Create the model
model1 = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping

batch_size = 32
nb_epochs = 500

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

# Compile the model
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with optimizations
hist = model1.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs,
                 validation_data=(X_test, y_test), callbacks=[early_stopping])

model1.save('model_parallel_azad.hdf5')

test_values = pd.read_pickle('test_data_values_300_10')


test_labels = pd.read_pickle('test_labels_300_10')
test_labels.to_excel("test_label_300_10.xlsx")

test_values = test_values.values

test_values = test_values.reshape((test_values.shape[0], test_values.shape[1], 1))

predictions = model1.predict(test_values)

prediction_list = []

threshold = 0.0001

# Iterate over each row of predictions
for row in predictions:
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

# Create the DataFrame from the list
#df = pd.DataFrame(prediction_list, columns=['prediction1', 'prediction2'])
prediction_list = np.array(prediction_list)
labels = ['Bias', 'Drift', 'Gain', 'NoFault', 'Outliers', 'Precisiondegradation']
label_encoder = LabelEncoder()
label_encoder.fit(labels)
predicted_labels_1 = label_encoder.inverse_transform(prediction_list[:,0])
predicted_labels_2 = label_encoder.inverse_transform(prediction_list[:,1])
df_predictions = pd.DataFrame({'Prediction': predicted_labels_1, 'Prediction2': predicted_labels_2})

# Display the resulting DataFrame
print(df_predictions)
df_predictions.to_excel('predictions4.xlsx')

