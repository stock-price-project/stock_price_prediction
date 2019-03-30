'''
This script is for training the time series neural network
'''

# Importing the Keras libraries 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


def training(X_train, y_train, no_of_features, epochs):
    # initialising the rnn
    model = Sequential()
    
    # adding the first lstm layer and some dropout regularization
    model.add(LSTM(units = X_train.shape[1], return_sequences = True, input_shape = (X_train.shape[1], no_of_features)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 100, return_sequences = True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(units = 40))
    model.add(Dropout(0.2))
    
    model.add(Dense(units = 1, activation='linear'))
    
    # compiling the rnn
    model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
    
    # fitting the rnn to the training set
    model.fit(X_train, y_train, epochs = epochs, batch_size = 32)
    
    return model

