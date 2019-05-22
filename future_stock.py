'''
this script helps to predict closing price of dataset using neural network using
single attribute. Input to the model is the open price trend and the output of the 
model is the closing price of the stock.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import save_load

# loading training data
df_train = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')
df = pd.concat([df_train, df_test], axis=0)

columns = [1]

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


timestep = 80
no_of_feature = 1
input_col = [0]
output_col = [0]

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

prediction_days = int(input("Enter the no_of_days: "))

###############################################################################

# loading the model
path_name = "./model/future_stock"
model = save_load.load_model(path_name)

sc_output = MinMaxScaler(feature_range = (0,1))
sc_output.fit(input_set[:, output_col]) 

# feature scaling
X_test_scaled = sc.transform(testing_set[:timestep, input_col ])
y_test = testing_set[timestep: timestep+prediction_days]

###############################################################################

output = []
for i in range(timestep, timestep+prediction_days):
    
    X_test = []
    X_test.append(X_test_scaled[i-timestep: i])
    X_test = np.array(X_test).reshape(1, -1)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_of_feature))

    output.append(model.predict(X_test))

    X_test_scaled = np.append(X_test_scaled, output)


output = sc.inverse_transform(np.array(output).reshape(-1,1))
actual = np.array(y_test)


# plotting error
error_list = []

for i in range(len(actual)):
    error = ((actual[i] - output[i])/actual[i])*100
    error_list.append(error)

###############################################################################

# saving the results in excel format
date = pd.DataFrame(df_test['Date'].iloc[:prediction_days])
actual_price_df = pd.DataFrame(actual).round(3)
predict_price_df = pd.DataFrame(output).round(3)
error_df = pd.DataFrame(error_list).round(3)
combined_df = pd.concat([date, actual_price_df, predict_price_df, error_df], axis = 1 )
combined_df.columns = ['date','actual_open', 'predicted_open', 'error_percent']
combined_df.to_excel('./model/future_stock/result.xlsx', index = False)


