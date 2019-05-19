'''
this script is for incremental training of neural network for predicting the 
stock price.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from utils import train
from utils import plot
from math import sqrt


# loading training data
df_train = pd.read_csv('./dataset/train.csv')

'''
we have used open and close data of the stock data here, open price is used of
the input data trend and close price will be our predicted output
'''
training_set = df_train.iloc[:, [1, 4]].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

'''
creating a data structure with 60 timestamp and predicting 1 output, later it is
reshaped, resulting in 3D tensor
'''

timestep = 60
train_predict = []
train_close = []
for i in range(0, len(training_set_scaled)-60):
    X_train = []
    y_train = []

    X_train.append(training_set_scaled[i: timestep+i, 0])     # from 0 to 3361 (open)
    y_train.append(training_set_scaled[timestep+i, 1])                 # from 60 to 3421 (close)

    # converting to numpy array
    X_train, y_train = np.array(X_train), np.array(y_train)
    # Reshaping to create 3D tensor
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    epochs = 5
    no_of_features = 1
    model = train.training(X_train, y_train, no_of_features, epochs)

    # prediction using train data
    pred_train_scaled = model.predict(X_train)

    # rescaling for predictions ( train data )
    scpred1 = MinMaxScaler(feature_range = (0,1))
    scpred1 = scpred1.fit(training_set[timestep+i, 1].reshape(-1,1))
    train_predict.append(scpred1.inverse_transform(pred_train_scaled))

    train_close.append(training_set[timestep+i, 1])

###############################################################################

train_predict = np.array(train_predict).reshape(-1)
train_close = np.array(train_close)

print('R2 Score : ', r2_score(train_close, train_predict))
print('MSE Score : ', mean_squared_error(train_close, train_predict))
print('RMSE Score : ',sqrt(mean_squared_error(train_close, train_predict)))


plot.time_series_plot(train_close, train_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Neural Network (single attribute - train data)')


