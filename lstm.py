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
avg_val = pd.DataFrame((df['High'] + df['Low'])/2, columns=['Avg.val'])
df = pd.concat([df, avg_val], axis=1)

# selecting Open,Close,Volume and Avg.val columns as input feature
training_set = df.iloc[:, [1, 4,6,7]].values                # from 0 to 3421

# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and predicting 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i])     # from 0 to 3361 (open, close)
    y_train.append(training_set_scaled[i, [0,1]])        # from 60 to 3421 (open)

# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)


# Building the RNN
# Importing the Keras libraries 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

# initialising the rnn

model1 = Sequential()
model2 = Sequential()
model = [model1, model2]
 
for i in range(y_train.shape[1]):
# adding the first lstm layer and some dropout regularization
    model[i].add(LSTM(units = X_train.shape[1], return_sequences = True, input_shape = (X_train.shape[1], 4)))
    model[i].add(Dropout(0.2))

    model[i].add(LSTM(units = 100, return_sequences = True))
    model[i].add(Dropout(0.2))

    model[i].add(LSTM(units = 100, return_sequences = True))
    model[i].add(Dropout(0.2))

    model[i].add(LSTM(units = 40))
    model[i].add(Dropout(0.2))

#model.add(Flatten())
    model[i].add(Dense(units = 1, activation='linear'))

# compiling the rnn
    model[i].compile(optimizer = 'rmsprop', loss = 'mean_squared_error')

# fitting the rnn to the training set
    model[i].fit(X_train, y_train[:,i], epochs = 50, batch_size = 32)

# serialize model to JSON
    model_json = model[i].to_json()
    json_pathname = "./model/model" + str(i) + ".json"
    with open(json_pathname, "w") as json_file:
        json_file.write(model_json)
# serialize weights to HDF5
    h5_pathname = "./model/model" + str(i) + ".h5"
    model[i].save_weights(h5_pathname)
    print("Saved model to disk")




# importing the testing file
df_test = pd.read_csv('./dataset/test.csv')
avg_val_test = pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])
df_test = pd.concat([df_test, avg_val_test], axis=1)
test_set = df_test.iloc[:, [1, 4,6,7]].values

# feature scaling
sc_t = MinMaxScaler(feature_range = (0,1))
test_set_scaled = sc_t.fit_transform(test_set)

# creating test data
X_test = []
y_test = [] 
for i in range(60, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-60: i])
    y_test.append(test_set_scaled[i, [0,1]])
# converting to numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)
# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

loaded_model1 = Sequential()
loaded_model2 = Sequential()
loaded_model = [loaded_model1, loaded_model2]
y_test_pred = []
y_train_pred = []
y_test_pred_rescaled = []
y_train_pred_rescaled = []

for i in range(y_test.shape[1]):
    # loading the model'
    json_pathname = "./model/model" + str(i) + ".json"
    json_file = open(json_pathname, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model[i] = model_from_json(loaded_model_json)
    # load weights into new model
    h5_pathname = "./model/model" + str(i) + ".h5"
    loaded_model[i].load_weights(h5_pathname)
    print("Loaded model from disk")
    
    # performing prediction on test set
    y_test_pred.append(loaded_model[i].predict(X_test))
    # performing prediction on train set
    y_train_pred.append(loaded_model[i].predict(X_train))
    
    # rescaling for predictions ( test data )
    scpred = MinMaxScaler(feature_range = (0,1))
    scpred = scpred.fit(test_set[:,i].reshape(-1,1))
    y_test_pred_rescaled.append(scpred.inverse_transform(y_test_pred[i]))
    
    # rescaling for predictions ( train data )
    scpred1 = MinMaxScaler(feature_range = (0,1))
    scpred1 = scpred1.fit(training_set[:,i].reshape(-1,1))
    y_train_pred_rescaled.append(scpred1.inverse_transform(y_train_pred[i]))
    
    # r2 score and mse score on test data
    print(r2_score(test_set[60:len(test_set),i], y_test_pred_rescaled[i]))
    print(mean_squared_error(test_set[60:len(test_set),i], y_test_pred_rescaled[i]))
    
    # visualising the results for test results
    plt.plot(test_set[60:len(test_set),i] , color='green', label='actual stock price')
    plt.plot(y_test_pred_rescaled[i], color='blue', label='predicted stock price')
    plt.title('google stock price')
    plt.xlabel('time')
    plt.ylabel('google stock price')
    plt.legend()
    plt.show()
    
    # r2 score and mse score on train data
    print(r2_score(training_set[60:len(training_set),i], y_train_pred_rescaled[i]))
    print(mean_squared_error(training_set[60:len(training_set),0], y_train_pred_rescaled[i]))
    
    # visualising the results for train results
    plt.plot(training_set[60:len(training_set),0], color='red', label='actual stock price')
    plt.plot(y_train_pred_rescaled[i], color='blue', label='predicted stock price')
    plt.title('google stock price')
    plt.xlabel('time')
    plt.ylabel('google stock price')
    plt.legend()
    plt.show()
    
# range to bid on stock
bid_range = y_test_pred_rescaled[1] - y_test_pred_rescaled[0]
plt.plot(bid_range, color='red', label='bid_range')
plt.hlines(0, 0, len(bid_range), colors='blue', linestyles='dashed')
plt.show()

plt.plot(y_test_pred_rescaled[0] , color='red', label='predicted opening price')
plt.plot(y_test_pred_rescaled[1], color='blue', label='predicted closing price')
plt.title('google stock price')
plt.xlabel('time')
plt.ylabel('google stock price')
plt.legend()
plt.show()

    