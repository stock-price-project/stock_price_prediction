'''
this script helps to predict closing price of dataset using neural network using
multiple attribute. Input to the model is open, close, volume, and avg. and, the 
output of the model is the closing price of the stock.
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
    y_train.append(training_set_scaled[i, 1])   

# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 4)


# importing the testing file
df_test = pd.read_csv('./dataset/test.csv')

# including the avg attribute in the test set
df_test = pd.concat([df_test, pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])], axis=1)
testing_set = df_test.iloc[:, [1, 4, 6, 7]].values
x1 = pd.DataFrame(training_set[len(training_set)-60:])
x2 = pd.DataFrame(testing_set)
testing_set = np.array(pd.concat([x1, x2]))

# feature scaling
sc_t = MinMaxScaler(feature_range = (0,1))
testing_set_scaled = sc_t.fit_transform(testing_set)

X_test = []
for i in range(60, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-60: i])
    
# converting to numpy array
X_test = np.array(X_test)

# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))


###############################################################################  

epochs = 120
no_of_features = 4
model = train.training(X_train, y_train, no_of_features, epochs)


path_name = "./model/prediction_multiple"

# Saving the model
save_load.save_model(path_name, model)

###############################################################################


# loading the model
path_name = "./model/prediction_multiple"
model = save_load.load_model(path_name)

# prediction using train set
pred_train_scaled = model.predict(X_train)
# rescaling for predictions ( train data )
scpred1 = MinMaxScaler(feature_range = (0,1))
scpred1 = scpred1.fit(training_set[:,1].reshape(-1,1))
train_predict = scpred1.inverse_transform(pred_train_scaled)

train_close = training_set[60:len(training_set),1]
print('R2 Score : ', r2_score(train_close, train_predict))
print('MSE Score : ', mean_squared_error(train_close, train_predict))
print('RMSE Score : ', sqrt(mean_squared_error(train_close, train_predict)))

plot.time_series_plot(train_close, train_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Neural Network (multiple attributes - train data)')


# prediction using test set
pred_test_scaled = model.predict(X_test)
# rescaling for predictions ( test data )
scpred = MinMaxScaler(feature_range = (0,1))
scpred = scpred.fit(testing_set[:,1].reshape(-1,1))
test_predict = scpred.inverse_transform(pred_test_scaled)

test_close = testing_set[60:len(testing_set),1]
print('R2 Score : ', r2_score(test_close, test_predict))
print('MSE Score : ', mean_squared_error(test_close, test_predict))
print('RMSE Score : ', sqrt(mean_squared_error(test_close, test_predict)))

plot.time_series_plot(test_close, test_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Neural Network (multiple attributes - test data)')


# plotting error
error_list = []

for i in range(len(test_close)):
    error = test_close[i] - test_predict[i]
    error_list.append(error)
    
plot.error_plot(error_list, "error graph - closing price prediction", 'close error') 
    
###############################################################################


# saving the results in csv format
actual_price_df = pd.DataFrame(test_close).round(3)
predict_price_df = pd.DataFrame(test_predict).round(3)
error_df = pd.DataFrame(error_list).round(3)
combined_df = pd.concat([actual_price_df, predict_price_df, error_df], axis = 1 )
combined_df.columns = ['actual_close', 'predicted_close', 'error_value']
combined_df.to_csv('./result/prediction_multiple_result.csv', index = False)
print("results saved to csv file")


