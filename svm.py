import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error


# training dataframe
df_train = pd.read_csv("./dataset/train.csv")

# input to the model
train_dates = []
for i in range(len(df_train)):
    train_dates.append(i)                         # from 0 to 3421
train_dates = np.array(train_dates).reshape(-1,1)

# output label to the model
train_close = np.array(df_train['Close']).reshape(-1,1)


##############################################################################

# initializing support vector regression model
svr_rbf = SVR(kernel = 'rbf', C=1e3 , gamma = 0.1)
svr_rbf.fit(train_dates, train_close)

##############################################################################

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
train_predict = svr_rbf.predict(train_dates)
train_predict = np.array(train_predict).reshape(-1,1)

plt.plot(train_dates, train_predict , color= 'blue', label ='RBF model')
plt.plot(train_dates, train_close, color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Support Vector Regresion')
plt.show()

print("R2 Score : ", r2_score(train_close, train_predict))
print("MSE Score : ", mean_squared_error(train_close, train_predict))


# prediction for training data
test_predict = svr_rbf.predict(test_dates)
test_predict = np.array(test_predict).reshape(-1,1)

plt.plot(test_dates, test_predict, color= 'blue', label ='RBF model')
plt.plot(test_dates, test_close, color='red')
plt.xlabel('Days')
plt.ylabel('Price')
plt.title('Support Vector Regresion')
plt.show()

print("R2 Score : ", r2_score(test_close, test_predict))
print("MSE Score : ", mean_squared_error(test_close, test_predict))
