# Prediction using Multiple attribute [(open, close, volume, avg) ----> close]
# Data preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
import csv
from math import sqrt
# Importing the Keras libraries 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json


# Importing the training set
df = pd.read_csv('./dataset/train.csv')

# Adding average feature in the dataframe
avg_val = pd.DataFrame((df['High'] + df['Low'])/2, columns=['Avg.val'])
df = pd.concat([df, avg_val], axis=1)

# selecting Open, Close, Volume and Avg.val columns as input feature
training_set = df.iloc[:, [1, 4, 6, 7]].values                # from 0 to 3421

# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# creating a data structure with 60 timesteps and predicting 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i])     
    y_train.append(training_set_scaled[i, 1])   

# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)

###############################################################################  
# initializing a sequential model
model = Sequential()

# adding the first lstm layer and some dropout regularization
model.add(LSTM(units = X_train.shape[1], return_sequences = True, input_shape = (X_train.shape[1], 4)))
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
model.fit(X_train, y_train, epochs = 120, batch_size = 32)

# Saving the model
# serialize model to JSON
model_json = model.to_json()
with open("./model/prediction_multiple/model.json", "w") as json_file:
    json_file.write(model_json)
    
# serialize weights to HDF5
model.save_weights("./model/prediction_multiple/model.h5")
print("Saved model to disk")


###############################################################################

# importing the testing file
df_test = pd.read_csv('./dataset/test.csv')

# including the avg attribute in the test set
avg_val_test = pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])
df_test = pd.concat([df_test, avg_val_test], axis=1)
testing_set = df_test.iloc[:, [1, 4, 6, 7]].values

# feature scaling
sc_t = MinMaxScaler(feature_range = (0,1))
testing_set_scaled = sc_t.fit_transform(testing_set)

# creating X_test, y_test
X_test = []
y_test = [] 
for i in range(60, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-60: i])
    y_test.append(testing_set_scaled[i, [0,1]])
    
# converting to numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)

# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))


# loading the model
json_file = open('./model/prediction_multiple/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./model/prediction_multiple/model.h5")
print("Loaded model from disk")

# performing prediction on test set
pred_test_scaled = loaded_model.predict(X_test)
# performing prediction on train set
pred_train_scaled = loaded_model.predict(X_train)

# rescaling for predictions ( test data )
scpred = MinMaxScaler(feature_range = (0,1))
scpred = scpred.fit(testing_set[:,1].reshape(-1,1))
pred_test = scpred.inverse_transform(pred_test_scaled)

# rescaling for predictions ( train data )
scpred1 = MinMaxScaler(feature_range = (0,1))
scpred1 = scpred1.fit(training_set[:,1].reshape(-1,1))
pred_train = scpred1.inverse_transform(pred_train_scaled)

# r2 score and mse score on test data
print(r2_score(testing_set[60:len(testing_set),1], pred_test))
print(mean_squared_error(testing_set[60:len(testing_set),1], pred_test))

# visualising the results for test results
plt.plot(testing_set[60:len(testing_set),1] , color='red', label='actual close price')
plt.plot(pred_test, color='blue', label='predicted close price')
plt.title('google stock price')
plt.xlabel('days')
plt.ylabel('stock price value')
plt.legend()
plt.show()

# r2 score and mse score on train data
print(r2_score(training_set[60:len(training_set),1], pred_train))
print(mean_squared_error(training_set[60:len(training_set),1], pred_train))

# visualising the results for train results
plt.plot(training_set[60:len(training_set),1], color='red', label='actual close price')
plt.plot(pred_train, color='blue', label='predicted close price')
plt.title('google stock price')
plt.xlabel('days')
plt.ylabel('google stock price')
plt.legend()
plt.show()

#writing output to a csv file

actual_price_df = pd.DataFrame(testing_set[60:len(testing_set),1])
predict_price_df = pd.DataFrame(pred_test)
mse_list = []
rmse_list = []

for i in range(len(pred_test)):
    mse_list.append(mean_squared_error(np.array(testing_set[60:len(testing_set),1][i]).reshape(-1), pred_test[i]))

for j in range(len(pred_test)):
    rmse_list.append(sqrt(mean_squared_error(np.array(testing_set[60:len(testing_set),1][j]).reshape(-1), pred_test[j])))



mse_value = pd.DataFrame(mse_list)
rmse_value = pd.DataFrame(rmse_list)
combined_df = pd.concat([actual_price_df, predict_price_df, mse_value, rmse_value], axis = 1 )
combined_df.columns = ['Actual_Closing_Price', 'Predicted_Closing_Price', 'MSE_Value','RMSE_Value']
combined_df.to_csv('./result/prediction_multiple_result.csv', index = False)











