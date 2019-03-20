# comparing the two result {single attribute result vs multiple attribute result}

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from keras.models import model_from_json

# Importing the training set
df = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')


# Adding average feature in the train set
avg_val = pd.DataFrame((df['High'] + df['Low'])/2, columns=['Avg.val'])
df = pd.concat([df, avg_val], axis=1)
avg_val_test = pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])
df_test = pd.concat([df_test, avg_val_test], axis=1)


# selecting Open, Close, Volume and Avg.val columns as input feature
training_set = df.iloc[:, [1, 4, 6, 7]].values               
testing_set = df_test.iloc[:, [1, 4, 6, 7]].values


# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)
sc_t = MinMaxScaler(feature_range = (0,1))
testing_set_scaled = sc_t.fit_transform(testing_set)


# creating a data structure with 60 timesteps and predicting 1 output
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i])     
    y_train.append(training_set_scaled[i, 1])  
    
X_test = []
y_test = [] 
for i in range(60, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-60: i])
    y_test.append(testing_set_scaled[i, [0,1]])
    
    
# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)


# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))


# loading the model
json_file = open('./model/prediction_single/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_sin = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_sin.load_weights("./model/prediction_single/model.h5")
print("Loaded model_sin from disk")

json_file = open('./model/prediction_multiple/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model_mul = model_from_json(loaded_model_json)
# load weights into new model
loaded_model_mul.load_weights("./model/prediction_multiple/model.h5")
print("Loaded model_mul from disk")



# performing prediction on test set
pred_test_scaled_sin = loaded_model_sin.predict(X_test[:,:,0].reshape(X_test.shape[0], X_test.shape[1], 1))
pred_test_scaled_mul = loaded_model_mul.predict(X_test)
# performing prediction on train set
pred_train_scaled_sin = loaded_model_sin.predict(X_train[:,:,0].reshape(X_train.shape[0], X_train.shape[1], 1))
pred_train_scaled_mul = loaded_model_mul.predict(X_train)


# rescaling for predictions ( test data )
scpred = MinMaxScaler(feature_range = (0,1))
scpred = scpred.fit(testing_set[:,1].reshape(-1,1))
pred_test_sin = scpred.inverse_transform(pred_test_scaled_sin)
pred_test_mul = scpred.inverse_transform(pred_test_scaled_mul)

# rescaling for predictions ( train data )
scpred1 = MinMaxScaler(feature_range = (0,1))
scpred1 = scpred1.fit(training_set[:,1].reshape(-1,1))
pred_train_sin = scpred1.inverse_transform(pred_train_scaled_sin)
pred_train_mul = scpred1.inverse_transform(pred_train_scaled_mul)


# r2 score and mse score on test data
print("Test Score")
print("R2 Score (single) : ", r2_score(testing_set[60:len(testing_set),1], pred_test_sin))
print("R2 Score (multiple): ", r2_score(testing_set[60:len(testing_set),1], pred_test_mul))
print("MSE (single): ", mean_squared_error(testing_set[60:len(testing_set),1], pred_test_sin))
print("MSE (multiple): ", mean_squared_error(testing_set[60:len(testing_set),1], pred_test_mul))


# visualising the results for test results
plt.plot(testing_set[60:len(testing_set),1] , color='red', label='actual close price')
plt.plot(pred_test_sin, color='blue', label='predicted close price (single)')
plt.plot(pred_test_mul, color='green', label='predicted close price (multiple)')
plt.title('google stock price')
plt.xlabel('days')
plt.ylabel('stock price value')
plt.legend()
plt.show()

# r2 score and mse score on train data
print("Train Score")
print("R2 Score (single) : ", r2_score(training_set[60:len(training_set),1], pred_train_sin))
print("R2 Score (multiple): ", r2_score(training_set[60:len(training_set),1], pred_train_mul))
print("MSE (single): ", mean_squared_error(training_set[60:len(training_set),1], pred_train_sin))
print("MSE (multiple): ", mean_squared_error(training_set[60:len(training_set),1], pred_train_mul))


# visualising the results for train results
plt.plot(training_set[60:len(training_set),1], color='red', label='actual close price')
plt.plot(pred_train_sin, color='blue', label='predicted close price (single)')
plt.plot(pred_train_mul, color='green', label='predicted close price (mulitple)')
plt.title('google stock price')
plt.xlabel('days')
plt.ylabel('google stock price')
plt.legend()
plt.show()
