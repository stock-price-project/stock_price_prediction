'''
this script helps to predict closing price of dataset using SVM regression.
Input to the model in the no. of days and the output of the model is the closing 
price of the stock.
'''

# importing libraries
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from utils import plot


# loading training data
df_train = pd.read_csv("./dataset/train.csv")

# input to the model
train_dates = []
for i in range(len(df_train)):
    train_dates.append(i)                         # from 0 to 3421
train_dates = np.array(train_dates).reshape(-1,1)

# output label to the model
train_close = np.array(df_train['Close']).reshape(-1,1)


# loading testing data
df_test = pd.read_csv("./dataset/test.csv")

# test input
test_dates = []
for i in range(len(df_train),len(df_train)+len(df_test)):
    test_dates.append(i)
test_dates = np.array(test_dates).reshape(-1,1)

# output label
test_close = np.array(df_test['Close']).reshape(-1,1)


##############################################################################

# initializing support vector regression model
svr_rbf = SVR(kernel = 'rbf', C=1e3 , gamma = 0.1)
svr_rbf.fit(train_dates, train_close)

##############################################################################


# prediction using training data
train_predict = svr_rbf.predict(train_dates)
train_predict = np.array(train_predict).reshape(-1,1)

print("R2 Score : ", r2_score(train_close, train_predict))
print("MSE Score : ", mean_squared_error(train_close, train_predict))
print("RMSE Score : ", sqrt(mean_squared_error(train_close, train_predict)))

plot.time_series_plot(train_close, train_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'SVM Model (train data)')

# prediction using testing data
test_predict = svr_rbf.predict(test_dates)
test_predict = np.array(test_predict).reshape(-1,1)

print("R2 Score : ", r2_score(test_close, test_predict))
print("MSE Score : ", mean_squared_error(test_close, test_predict))
print("RMSE Score : ", sqrt(mean_squared_error(test_close, test_predict)))

plot.time_series_plot(test_close, test_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'SVM Model (test data)')


# saving the results in csv format
mse_list = []
rmse_list = []

for i in range(len(test_close)):
    mse_error = mean_squared_error(np.array(test_close[i]).reshape(-1), test_predict[i])
    rmse_error = sqrt(mse_error)
    mse_list.append(mse_error)
    rmse_list.append(rmse_error)

actual_price_df = pd.DataFrame(test_close)
predict_price_df = pd.DataFrame(test_predict)
mse_value = pd.DataFrame(mse_list)
rmse_value = pd.DataFrame(rmse_list)
combined_df = pd.concat([actual_price_df, predict_price_df, mse_value, rmse_value], axis = 1 )
combined_df.columns = ['Actual_Closing_Price', 'Predicted_Closing_Price', 'MSE_Value','RMSE_Value']
combined_df.to_csv('./result/prediction_svm_result.csv', index = False)
print("results saved to csv file")

