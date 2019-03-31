'''
this script helps to predict opening and closing price of dataset using neural
network using multiple attributes. Input to the model is open, close, volume, 
and avg.. After finding out opening and closing price, we can produce the 
bidding range.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from utils import train
from utils import save_load
from utils import plot
from math import sqrt


# loading training data
df = pd.read_csv('./dataset/train.csv')

# Adding average feature in the dataframe
df = pd.concat([df, pd.DataFrame((df['High'] + df['Low'])/2, columns=['Avg.val'])], axis=1)

'''
we have used open, close, volume, avg. data of the stock data here, open price,
close price, volume and avg. is used as the input data trend and close price 
will be our predicted output
'''
training_set = df.iloc[:, [1, 4, 6, 7]].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

'''
creating a data structure with 60 timestamp and predicting 1 output, later it is
reshaped, resulting in 3D tensor
'''
X_train = []
y_train = []
for i in range(60, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-60: i])     
    y_train.append(training_set_scaled[i, [0,1]])

# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)


# importing the testing file
df_test = pd.read_csv('./dataset/test.csv')

# including the avg attribute in the test set
df_test = pd.concat([df_test, pd.DataFrame((df_test['High'] + df_test['Low'])/2, \
                                           columns=['Avg.val'])], axis=1)
test_set = df_test.iloc[:, [1, 4, 6, 7]].values
x1 = pd.DataFrame(training_set[len(training_set)-60:])
x2 = pd.DataFrame(test_set)
test_set = np.array(pd.concat([x1, x2]))

# feature scaling
sc_t = MinMaxScaler(feature_range = (0,1))
test_set_scaled = sc_t.fit_transform(test_set)

# creating X_test, y_test
X_test = [] 
for i in range(60, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-60: i])
    
# converting to numpy array
X_test = np.array(X_test)

# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

###############################################################################

'''
first loop for training model with open price
second loop for training model with close price
'''
epochs = 120
no_of_features = 4

for i in range(y_train.shape[1]):
    
    model = train.training(X_train, y_train[:, i], no_of_features, epochs)
    
    path_name = "./model/bid_model" 

    # Saving the model
    save_load.save_model(path_name + "/" + str(i), model)
    

###############################################################################

path_name = "./model/bid_model"
test_predict = []
test_predict_rescaled = []

'''
first loop for predicting open price
second loop for predicting close price
'''
for i in range(y_train.shape[1]):
    
    # loading the model
    model = save_load.load_model(path_name + "/" + str(i))

    # performing prediction on test set
    test_predict.append(model.predict(X_test))
    
    # rescaling for predictions ( test data )
    scpred = MinMaxScaler(feature_range = (0,1))
    scpred = scpred.fit(test_set[:,i].reshape(-1,1))
    test_predict_rescaled.append(scpred.inverse_transform(np.array(test_predict[i]).reshape(-1,1)))

# analysis using opening price   
test_actual = test_set[60:len(test_set),0]
print(r2_score(test_actual, test_predict_rescaled[0]))
print(mean_squared_error(test_actual, test_predict_rescaled[0]))
print(sqrt(mean_squared_error(test_actual, test_predict_rescaled[0])))
plot.time_series_plot(test_actual, test_predict_rescaled[0], 'red', 'blue', 'actual', \
                      'predicted', 'days', 'open price', 'Neural Network (multiple | opening)')    

# analysis using closing price 
test_actual = test_set[60:len(test_set),1]
print(r2_score(test_actual, test_predict_rescaled[1]))
print(mean_squared_error(test_actual, test_predict_rescaled[1]))
print(sqrt(mean_squared_error(test_actual, test_predict_rescaled[1])))
plot.time_series_plot(test_actual, test_predict_rescaled[1], 'red', 'blue', 'actual', \
                      'predicted', 'days', 'close price', 'Neural Network (multiple | closing)') 

# analysis for bidding
plot.bid_plot(test_predict_rescaled) 


###############################################################################

# saving the results in csv format
mse_open_list = []
mse_close_list = []
bid_range = []

for i in range(X_test.shape[0]):
    
    test_actual = test_set[60:len(test_set),0][i]
    mse_open = mean_squared_error(test_actual.reshape(-1), test_predict_rescaled[0][i])
    mse_open_list.append(mse_open)
    
    test_actual = test_set[60:len(test_set),1][i]
    mse_close = mean_squared_error(test_actual.reshape(-1), test_predict_rescaled[1][i])
    mse_close_list.append(mse_close)
    
    bid_value = test_predict_rescaled[1][i] - test_predict_rescaled[0][i]
    bid_range.append(str(round(float(test_predict_rescaled[0][i]), 3)) +" + " + str(round(float(bid_value), 3)))
    

actual_close_df = pd.DataFrame(test_set[60:len(test_set),1]).round(3)
actual_open_df = pd.DataFrame(test_set[60:len(test_set),0]).round(3)
predict_close_df = pd.DataFrame(test_predict_rescaled[1]).round(3)
predict_open_df = pd.DataFrame(test_predict_rescaled[0]).round(3)
bid_df = pd.DataFrame(bid_range)
mse_open_df = pd.DataFrame(mse_open_list).round(3)
mse_close_df = pd.DataFrame(mse_close_list).round(3)

combined_df = pd.concat([actual_open_df, predict_open_df, actual_close_df, \
                         predict_close_df, bid_df, mse_open_df, \
                         mse_close_df], axis = 1 )
combined_df.columns = ['actual_open','predict_open', 'actual_close', \
                       'predict_close', 'bid_value', 'open error', \
                       'close error']
combined_df.to_csv('./result/prediction_bid_result.csv', index = False)
print("results saved to csv file")

