import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt

# training dataframe
df_train = pd.read_csv("./dataset/train.csv")

# input to the model
train_dates = []
for i in range(len(df_train)):
    train_dates.append(i)                         # from 0 to 3421
train_dates = np.array(train_dates).reshape(-1,1)

# output label to the model
train_close = np.array(df_train['Close']).reshape(-1,1)

# polynomial features
poly_feat = PolynomialFeatures(degree = 12)
X_poly = poly_feat.fit_transform(train_dates)

# polynomial model
poly_model = LinearRegression()
poly_model = poly_model.fit(X_poly,train_close)


# testing dataframe
df_test = pd.read_csv("./dataset/test.csv")

# test input
test_dates = []
for i in range(len(df_train),len(df_train)+len(df_test)):
    test_dates.append(i)
test_dates = np.array(test_dates).reshape(-1,1)

# output label
test_close = np.array(df_test['Close']).reshape(-1,1)

# prediction for training data
X_poly_train = poly_feat.fit_transform(train_dates)
train_predict = poly_model.predict(X_poly_train)
train_predict = np.array(train_predict).reshape(-1,1)
   
plt.plot(train_dates, train_predict , color= 'blue', label ='RBF model')
plt.plot(train_dates, train_close, color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Support Vector Regresion')
plt.show()

print("R2 Score : ", r2_score(train_close, train_predict))
print("MSE Score : ", mean_squared_error(train_close, train_predict))
print("RMSE Score : ", sqrt(mean_squared_error(train_close, train_predict)))   
# prediction for training data
X_poly_test = poly_feat.fit_transform(test_dates)
test_predict = poly_model.predict(X_poly_test)
test_predict = np.array(test_predict).reshape(-1,1)
    
plt.plot(test_dates, test_predict, color= 'blue', label ='RBF model')
plt.plot(test_dates, test_close, color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Support Vector Regresion')
#plt.ylim((0,1400))
#plt.xlim((0,3550))
plt.show()

print("R2 Score : ", r2_score(test_close, test_predict))
print("MSE Score : ", mean_squared_error(test_close, test_predict))
print("RMSE Score : ", sqrt(mean_squared_error(test_close, test_predict)))

actual_price_df = pd.DataFrame(test_close)
predict_price_df = pd.DataFrame(test_predict)
mse_list = []
rmse_list = []

for i in range(len(test_predict)):
    mse_list.append(mean_squared_error(np.array(test_close[i]).reshape(-1), test_predict[i]))

for j in range(len(test_predict)):
    rmse_list.append(sqrt(mean_squared_error(np.array(test_close[j]).reshape(-1), test_predict[j])))



mse_value = pd.DataFrame(mse_list)
rmse_value = pd.DataFrame(rmse_list)
combined_df = pd.concat([actual_price_df, predict_price_df, mse_value, rmse_value], axis = 1 )
combined_df.columns = ['Actual_Closing_Price', 'Predicted_Closing_Price', 'MSE_Value','RMSE_Value']
combined_df.to_csv('./result/prediction_regression_result.csv', index = False)

