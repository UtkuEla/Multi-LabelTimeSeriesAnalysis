import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout, concatenate, GlobalMaxPooling1D
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import sklearn

class TimeSeriesModel:
    """
    This class inolves different model architectures

    """
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def Parallel_CNN_RNN(self):
        
        input_layer = Input(shape=self.input_shape)
        conv_layer1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
        conv_layer2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_layer1)
        conv_layer3 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_layer2)
        maxpool_layer = MaxPooling1D(pool_size=2)(conv_layer3)
        rnn_layer = LSTM(units=64, return_sequences=True)(maxpool_layer)
        cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(maxpool_layer)
        cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
        cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_layer)
        cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
        cnn_layer = Conv1D(filters=64, kernel_size=3, activation='relu')(cnn_layer)
        cnn_layer = MaxPooling1D(pool_size=2)(cnn_layer)
        cnn_layer = Flatten()(cnn_layer)
        combined_layer = concatenate([rnn_layer, cnn_layer])
        global_pooling_layer = GlobalMaxPooling1D()(combined_layer)
        dense_layer = Dense(units=128, activation='relu')(global_pooling_layer)
        dropout_layer = Dropout(0.5)(dense_layer)
        output_layer = Dense(units=self.num_classes, activation='softmax')(dropout_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def CNN_RNN_1(self):
        
        input_layer = Input(shape=self.input_shape)
        conv_layer1 = Conv1D(filters=64, kernel_size=5, activation='relu')(input_layer)
        conv_layer2 = Conv1D(filters=64, kernel_size=5, activation='relu')(conv_layer1)
        maxpool_layer = MaxPooling1D(pool_size=2)(conv_layer2)
        rnn_layer = LSTM(units=128, return_sequences=True)(maxpool_layer)
        rnn_layer = LSTM(units=128)(rnn_layer)
        dense_layer = Dense(units=256, activation='relu')(rnn_layer)
        output_layer = Dense(units=self.num_classes, activation='softmax')(dense_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        
        return model

    def CNN_RNN_2(self):
        
        input_layer = Input(shape=self.input_shape)
        conv_layer1 = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
        conv_layer2 = Conv1D(filters=32, kernel_size=3, activation='relu')(conv_layer1)
        maxpool_layer = MaxPooling1D(pool_size=2)(conv_layer2)
        rnn_layer = LSTM(units=64, return_sequences=True)(maxpool_layer)
        rnn_layer = LSTM(units=64)(rnn_layer)
        dense_layer = Dense(units=128, activation='relu')(rnn_layer)
        output_layer = Dense(units=self.num_classes, activation='softmax')(dense_layer)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        return model

    def EchoStateSimple(self):
        input_layer = Input(shape=self.input_shape)
        rnn_layer = SimpleRNN(units=self.num_units, activation='relu')(input_layer)
        dense_layer = Dense(units=self.num_units, activation='relu')(rnn_layer)
        output_layer = Dense(units=self.input_shape[1])(dense_layer)

        model = Model(inputs=input_layer, outputs=output_layer)

        return model

    def RNN_SLP_fromWeb(self):
        
    #     # Define timesteps and the number of features

    #     n_timesteps = 8

    #     n_features = 7

    #     # RNN + SLP Model

    #     # Define input layer

    #     recurrent_input = Input(shape=(n_timesteps,n_features),name=&amp;amp;amp;quot;TIMESERIES_INPUT&amp;amp;amp;quot;)

    #     static_input = Input(shape=(x_train_over_static.shape[1], ),name=&amp;amp;amp;quot;STATIC_INPUT&amp;amp;amp;quot;)

    #     # RNN Layers

    #     # layer - 1

    #     rec_layer_one = Bidirectional(LSTM(128, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01),return_sequences=True),name =&amp;amp;amp;quot;BIDIRECTIONAL_LAYER_1&amp;amp;amp;quot;)(recurrent_input)

    #     rec_layer_one = Dropout(0.1,name =&amp;amp;amp;quot;DROPOUT_LAYER_1&amp;amp;amp;quot;)(rec_layer_one)

    #     # layer - 2

    #     rec_layer_two = Bidirectional(LSTM(64, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)),name =&amp;amp;amp;quot;BIDIRECTIONAL_LAYER_2&amp;amp;amp;quot;)(rec_layer_one)

    #     rec_layer_two = Dropout(0.1,name =&amp;amp;amp;quot;DROPOUT_LAYER_2&amp;amp;amp;quot;)(rec_layer_two)

    #     # SLP Layers

    #     static_layer_one = Dense(64, kernel_regularizer=l2(0.001), activation='relu',name=&amp;amp;amp;quot;DENSE_LAYER_1&amp;amp;amp;quot;)(static_input)

    #     # Combine layers - RNN + SLP

    #     combined = Concatenate(axis= 1,name = &amp;amp;amp;quot;CONCATENATED_TIMESERIES_STATIC&amp;amp;amp;quot;)([rec_layer_two,static_layer_one])

    #     combined_dense_two = Dense(64, activation='relu',name=&amp;amp;amp;quot;DENSE_LAYER_2&amp;amp;amp;quot;)(combined)

    #     output = Dense(n_outputs,activation='sigmoid',name=&amp;amp;amp;quot;OUTPUT_LAYER&amp;amp;amp;quot;)(combined_dense_two)

    #     # Compile Model

    #     model = Model(inputs=[recurrent_input,static_input],outputs=[output])
        pass


class NewModels:
    def __init__(self):
        pass

