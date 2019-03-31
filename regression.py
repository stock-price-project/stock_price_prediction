'''
this script helps to predict closing price of dataset using polynomial regression.
Input to the model in the no. of days and the output of the model is the closing 
price of the stock.
'''

# importing libraries
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
from utils import plot

# loading training data
df_train = pd.read_csv("./dataset/train.csv")

# input to the model
train_dates = []
for i in range(len(df_train)):
    train_dates.append(i)
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

###############################################################################

# initializing polynomial features
'''
you can find the best degree using a loop where you get the max. r2 score, we 
have got degree 12
'''
poly_feat = PolynomialFeatures(degree = 12)
X_poly = poly_feat.fit_transform(train_dates)

# polynomial model
poly_model = LinearRegression()
poly_model = poly_model.fit(X_poly,train_close)


###############################################################################

# prediction using training data
X_poly_train = poly_feat.fit_transform(train_dates)
train_predict = poly_model.predict(X_poly_train)
train_predict = np.array(train_predict).reshape(-1,1)

# printing scores
print("R2 Score : ", r2_score(train_close, train_predict))
print("MSE Score : ", mean_squared_error(train_close, train_predict))
print("RMSE Score : ", sqrt(mean_squared_error(train_close, train_predict)))  

plot.time_series_plot(train_close, train_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Regression Model (traing data)')

 
# prediction for testing data
X_poly_test = poly_feat.fit_transform(test_dates)
test_predict = poly_model.predict(X_poly_test)
test_predict = np.array(test_predict).reshape(-1,1)

# printing scores
print("R2 Score : ", r2_score(test_close, test_predict))
print("MSE Score : ", mean_squared_error(test_close, test_predict))
print("RMSE Score : ", sqrt(mean_squared_error(test_close, test_predict)))

plot.time_series_plot(test_close, test_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Regression Model (test data)')


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
combined_df.to_csv('./result/prediction_regression_result.csv', index = False)
print("results saved to csv file")
