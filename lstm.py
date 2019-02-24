# this file created on 14th feb, 2019
# Data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

# Importing the training set
df = pd.read_csv('./dataset/train.csv')
# selecting open and close columns as input feature
training_set = df.iloc[:, [1, 4]].values                # from 0 to 3421

# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timestamp and predicting 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i])     # from 0 to 3361 (open, close)
    y_train.append(training_set_scaled[i, 0])        # from 60 to 3421 (open)

# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)

###############################################################################
# Building the RNN
# Importing the Keras libraries 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

# initialising the rnn
model = Sequential()

# adding the first lstm layer and some dropout regularization
model.add(LSTM(units = X_train.shape[1], return_sequences = True, input_shape = (X_train.shape[1], 2)))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 40))
model.add(Dropout(0.2))

model.add(Dense(units = 1, activation='sigmoid'))

# compiling the rnn
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# fitting the rnn to the training set
model.fit(X_train, y_train, epochs = 50, batch_size = 32)

############################################################################### 

# serialize model to JSON
model_json = model.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model1.h5")
print("Saved model to disk")


# loading the model
json_file = open('model1.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model1.h5")
print("Loaded model from disk")

###############################################################################

# importing the testing file
df_test = pd.read_csv('./dataset/test.csv')
actual_open = df_test.iloc[:, [1, 4]].values

# feature scaling
sc_t = MinMaxScaler(feature_range = (0,1))
actual_open_scaled = sc_t.fit_transform(actual_open)

# creating test data
X_test = []
for i in range(60, len(actual_open_scaled)):
    X_test.append(actual_open_scaled[i-60: i])
    
# converting to numpy array
X_test = np.array(X_test)

# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 2))    

# performing prediction on test set
pred_test_scaled = loaded_model.predict(X_test)
# performing prediction on train set
pred_train_scaled = loaded_model.predict(X_train)

# rescaling for predictions ( test data )
scpred = MinMaxScaler(feature_range = (0,1))
scpred = scpred.fit(actual_open[:,0].reshape(-1,1))
pred_test = scpred.inverse_transform(pred_test_scaled)

# rescaling for predictions ( train data )
scpred1 = MinMaxScaler(feature_range = (0,1))
scpred1 = scpred1.fit(training_set[:,0].reshape(-1,1))
pred_train = scpred1.inverse_transform(pred_train_scaled)

# r2 score and mse score on test data
print(r2_score(actual_open[60:len(actual_open),0], pred_test))
print(mean_squared_error(actual_open[60:len(actual_open),0], pred_test))

# visualising the results for test results
plt.plot(actual_open[60:len(actual_open),0] , color='red', label='actual stock price')
plt.plot(pred_test, color='blue', label='predicted stock price')
plt.title('google stock price')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()

# r2 score and mse score on train data
print(r2_score(training_set[60:len(training_set),0], pred_train))
print(mean_squared_error(training_set[60:len(training_set),0], pred_train))

# visualising the results for train results
plt.plot(training_set[60:len(training_set),0], color='red', label='actual stock price')
plt.plot(pred_train, color='blue', label='predicted stock price')
plt.title('google stock price')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()
