# this file created on 14th feb, 2019
# Data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
df = pd.read_csv('train.csv')
training_set = df.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timestamp and 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)



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
model.add(LSTM(units = X_train.shape[1], return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 40))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

# compiling the rnn
model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# fitting the rnn to the training set
model.fit(X_train, y_train, epochs = 120, batch_size = 32)


###############################################################################
# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")


# loading the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

###############################################################################
# importing the testing file
test_data = pd.read_csv('test.csv')
actual_open = test_data.iloc[:, 1:2].values

dataset_total = pd.concat((df['Open'], test_data['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(test_data) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

X_test = []
for i in range(60, len(actual_open)+60):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

pred_open = loaded_model.predict(X_test)
pred_open_train = loaded_model.predict(X_train)

pred_open = sc.inverse_transform(pred_open)
pred_open_train = sc.inverse_transform(pred_open_train)

from sklearn.metrics import r2_score, mean_squared_error

print(r2_score(actual_open, pred_open))
print(mean_squared_error(actual_open, pred_open))

# visualising the results
plt.plot(actual_open , color='red', label='actual stock price')
plt.plot(pred_open, color='blue', label='predicted stock price')
plt.title('google stock price')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()


y_train = sc.inverse_transform(y_train.reshape(-1,1))
z = np.append(pred_open_train, pred_open)
w = np.append(y_train, actual_open)

print(r2_score(w, z))
print(mean_squared_error(w, z))


# visualising the results
plt.plot(w, color='red', label='actual stock price')
plt.plot(z, color='blue', label='predicted stock price')
plt.title('google stock price')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()
