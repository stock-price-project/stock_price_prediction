'''
this script helps to predict closing price of dataset using neural network using
single attribute. Input to the model is the open price trend and the output of the 
model is the closing price of the stock.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from utils import train
from utils import save_load
from utils import plot

# loading training data
df_train = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')
df = pd.concat([df_train, df_test], axis=0)

columns = [1, 4]

input_set = df.iloc[:, columns].values

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(input_set) 

'''
we have used open and close data of the stock data here, open price is used of
the input data trend and close price will be our predicted output
'''

training_set = df_train.iloc[:, columns].values

# Feature Scaling
training_set_scaled = sc.transform(training_set)

'''
creating a data structure with 60 timestamp and predicting 1 output, later it is
reshaped, resulting in 3D tensor
'''

timestep = 60
no_of_feature = 1
input_col = [1]
output_col = [1]

X_train = []
y_train = []
for i in range(timestep, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timestep: i, input_col])
    y_train.append(training_set_scaled[i, output_col])        

# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], no_of_feature)

# importing the testing file
testing_set = df_test.iloc[:, columns].values

x1 = pd.DataFrame(training_set[len(training_set)-timestep:])
x2 = pd.DataFrame(testing_set)
testing_set = np.array(pd.concat([x1, x2]))

# feature scaling
testing_set_scaled = sc.transform(testing_set)

'''
creating a data structure with 60 timestamp and predicting 1 output, later it is
reshaped, resulting in 3D tensor
'''
X_test = []
y_test = []
for i in range(timestep, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-timestep: i, input_col])
    y_test.append(testing_set_scaled[i, output_col])
    
# converting to numpy array
X_test, y_test = np.array(X_test), np.array(y_test)

# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_of_feature))

###############################################################################

epochs = 120
model = train.training(X_train, y_train, no_of_feature, epochs)

path_name = "./model/single_attr_pred_open_from_open"

# Saving the model
save_load.save_model(path_name, model)

###############################################################################

# loading the model
path_name = "./model/single_attr_pred_close_from_close"
model = save_load.load_model(path_name)

sc_output = MinMaxScaler(feature_range = (0,1))
sc_output.fit(input_set[:,output_col]) 

# prediction using train data
pred_train_scaled = model.predict(X_train)

# rescaling for predictions ( train data )
train_predict = sc_output.inverse_transform(pred_train_scaled)
train_actual = sc_output.inverse_transform(y_train) 

print('R2 Score : ', r2_score(train_actual, train_predict))
print('MSE Score : ', mean_squared_error(train_actual, train_predict))

plot.time_series_plot(train_actual, train_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Neural Network (single attribute - train data)')


pred_test_scaled = model.predict(X_test)

# rescaling for predictions ( test data )
test_predict = sc_output.inverse_transform(pred_test_scaled)
#test = sc_output.inverse_transform(testing_set_scaled)

test_actual = sc_output.inverse_transform(y_test)

print('R2 Score : ', r2_score(test_actual, test_predict))
print('MSE Score : ', mean_squared_error(test_actual, test_predict))

plot.time_series_plot(test_actual, test_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Neural Network (single attribute - test data)')


# plotting error
error_list = []

for i in range(len(test_actual)):
    error = ((test_actual[i] - test_predict[i])/test_actual[i])*100
    error_list.append(error)
    
#plot.error_plot(error_list, "error graph - closing price prediction", 'close error') 

###############################################################################

# saving the results in excel format
date = pd.DataFrame(df_test['Date'])
actual_price_df = pd.DataFrame(test_actual).round(3)
predict_price_df = pd.DataFrame(test_predict).round(3)
error_df = pd.DataFrame(error_list).round(3)
combined_df = pd.concat([date, actual_price_df, predict_price_df, error_df], axis = 1 )
combined_df.columns = ['date','actual_close', 'predicted_close', 'error_percent']
combined_df.to_excel('./model/single_attr_pred_close_from_close/result.xlsx', index = False)

