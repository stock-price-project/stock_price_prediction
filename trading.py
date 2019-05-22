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
from utils import save_load
from utils import plot


# loading training data
df_train = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')
df = pd.concat([df_train, df_test], axis=0)

# Adding average feature in the dataframe
df = pd.concat([df, pd.DataFrame((df['High'] + df['Low'])/2, columns=['Avg.val'])], axis=1)

columns = [1, 4, 6, 7]
features_open = 3
features_close = 4
timestep = 80
input_open = [1, 2, 3]
input_close = [0, 1, 2, 3]
col_output_open = [0]
col_output_close = [1]

input_set = df.iloc[:, columns].values

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(input_set) 


df_train = pd.concat([df_train, pd.DataFrame((df_train['High'] + df_train['Low'])/2, columns=['Avg.val'])], axis=1)
training_set = df_train.iloc[:, columns].values

# Feature Scaling
training_set_scaled = sc.transform(training_set)

# including the avg attribute in the test set
df_test = pd.concat([df_test, pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])], axis=1)

testing_set = df_test.iloc[:, columns].values

x1 = pd.DataFrame(training_set[len(training_set)-timestep:])
x2 = pd.DataFrame(testing_set)
testing_set = np.array(pd.concat([x1, x2]))

# feature scaling
testing_set_scaled = sc.transform(testing_set)

X_test_open = []
y_test_open = []
for i in range(timestep, len(testing_set_scaled)):
    X_test_open.append(testing_set_scaled[i-timestep: i, input_open])
    y_test_open.append(testing_set_scaled[i, col_output_open])
    
X_test_close = []
y_test_close = []
for i in range(timestep, len(testing_set_scaled)):
    X_test_close.append(testing_set_scaled[i-timestep: i, input_close])
    y_test_close.append(testing_set_scaled[i, col_output_close])
    
# converting to numpy array
X_test_open, y_test_open = np.array(X_test_open), np.array(y_test_open)
X_test_close, y_test_close = np.array(X_test_close), np.array(y_test_close)

# creating 3D tensor
X_test_open = np.reshape(X_test_open, (X_test_open.shape[0], X_test_open.shape[1], features_open))
X_test_close = np.reshape(X_test_close, (X_test_close.shape[0], X_test_close.shape[1], features_close))

###############################################################################

sc_output = MinMaxScaler(feature_range = (0,1))
sc_output.fit(input_set[:, col_output_open]) 

# loading the model
path_name_open = "./model/trading_model/open"
model_open = save_load.load_model(path_name_open)

# prediction using train set
output_open_scaled = model_open.predict(X_test_open)

# rescaling for predictions ( train data )
output_open = sc_output.inverse_transform(output_open_scaled)
actual_open = sc_output.inverse_transform(y_test_open)

# analysis using opening price
print('R2 Score : ', r2_score(actual_open, output_open))
print('MSE Score : ', mean_squared_error(actual_open, output_open))
plot.time_series_plot(actual_open, output_open, 'red', 'blue', 'actual', \
                      'predicted', 'days', 'open price', 'Neural Network (trading)')    

# loading the model
path_name_close = "./model/trading_model/close"
model_close = save_load.load_model(path_name_close)

# prediction using train set
output_close_scaled = model_close.predict(X_test_close)

# rescaling for predictions ( train data )
output_close = sc_output.inverse_transform(output_close_scaled)
actual_close = sc_output.inverse_transform(y_test_close)

# analysis using closing price 
print('R2 Score : ', r2_score(actual_close, output_close))
print('MSE Score : ', mean_squared_error(actual_close, output_close))
plot.time_series_plot(actual_close, output_close, 'red', 'blue', 'actual', \
                      'predicted', 'days', 'close price', 'Neural Network (trading)')  


plot.time_series_plot(output_open, output_close, 'red', 'blue', 'actual', \
                      'predicted', 'days', 'price', 'Neural Network (trading)')  

###############################################################################

# saving the results in csv format
error_open = []
error_close = []
avg_error = []
trading_range = []

for i in range(len(output_close)):
    trading_range.append(str(output_close[i].round(2)) + " - " + str(output_open[i].round(2)))

for i in range(len(actual_open)):
    error = ((actual_open[i] - output_open[i])/actual_open[i])*100
    error_open.append(error)
    
for i in range(len(actual_close)):
    error = ((actual_close[i] - output_close[i])/actual_close[i])*100
    error_close.append(error)
    
list1 = [abs(x) for x in error_close]
list2 = [abs(x) for x in error_open]  

import operator
list3 = list(map(operator.add, list1, list2)) 
avg_error = [x/2 for x in list3]   

date = pd.DataFrame(df_test['Date'])
actual_close_df = pd.DataFrame(actual_close).round(3)
actual_open_df = pd.DataFrame(actual_open).round(3)
predict_close_df = pd.DataFrame(output_close).round(3)
predict_open_df = pd.DataFrame(output_open).round(3)
trade_df = pd.DataFrame(trading_range).round(3)
avg_error_df = pd.DataFrame(avg_error).round(3)

combined_df = pd.concat([date, actual_open_df, predict_open_df, actual_close_df, \
                         predict_close_df, trade_df, avg_error_df], axis = 1 )
combined_df.columns = ['date','actual_open','predict_open', 'actual_close', \
                       'predict_close', 'trading_range', 'avg. error']
combined_df.to_excel('./model/trading_model/result.xlsx', index = False)

