'''
this script helps to select the best features which contribute the most in the
prediction.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from utils import train
from utils import save_load


# loading training data
df_train = pd.read_csv('./dataset/train.csv')
df_test = pd.read_csv('./dataset/test.csv')
df = pd.concat([df_train, df_test], axis=0)

# Adding average feature in the dataframe
df = pd.concat([df, pd.DataFrame((df['High'] + df['Low'])/2, columns=['Avg.val'])], axis=1)

'''
we have used open, close, volume, avg. data of the stock data here, open price,
close price, volume and avg. is used for creating the pair of features.
'''

columns = [1, 4, 6, 7]
no_of_feature = 4
timestep = 80
output_col = [1]

input_set = df.iloc[:, columns].values

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(input_set) 

# training data
df_train = pd.concat([df_train, pd.DataFrame((df_train['High'] + df_train['Low'])/2, columns=['Avg.val'])], axis=1)
training_set = df_train.iloc[:, columns].values

# Feature Scaling
training_set_scaled = sc.transform(training_set)

'''
creating a data structure with 60 timestamp and predicting 1 output, later it is
reshaped, resulting in 3D tensor
'''
X_train = []
y_train = []
for i in range(timestep, len(training_set_scaled)):
    X_train.append(training_set_scaled[i-timestep: i])     
    y_train.append(training_set_scaled[i, output_col])

# converting to numpy array
X_train, y_train = np.array(X_train), np.array(y_train)
# Reshaping to create 3D tensor
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], no_of_feature)


# loading testing data

# including the avg attribute in the test set
df_test = pd.concat([df_test, pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])], axis=1)
testing_set = df_test.iloc[:, columns].values

x1 = pd.DataFrame(training_set[len(training_set)-timestep:])
x2 = pd.DataFrame(testing_set)
testing_set = np.array(pd.concat([x1, x2]))
# feature scaling
testing_set_scaled = sc.transform(testing_set)

X_test = []
y_test = []
for i in range(timestep, len(testing_set_scaled)):
    X_test.append(testing_set_scaled[i-timestep: i])
    y_test.append(testing_set_scaled[i, output_col])
    
# converting to numpy array
X_test, y_test = np.array(X_test), np.array(y_test)

# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_of_feature))

# ============================================================================
import os

count = 0
epochs = 150

combination = []
columns_index = [0, 1, 2, 3]

from itertools import combinations
for i in range(1,len(columns_index)+1):
    combination.append(list((combinations(columns_index, i))))


for i in range(no_of_feature):
    for j in range(len(combination[i])):
        feature = np.array(combination[i][j])
        model = train.training(X_train[:, :, feature], y_train, feature.shape[0] , epochs)
        path_name = "./model/feature_importance_close" + "/" + str(count)
        
        os.mkdir(path_name)
        # Saving the model
        save_load.save_model(path_name, model)
        count = count + 1

# =============================================================================

path_name = "./model/feature_importance_close" 

sc_output = MinMaxScaler(feature_range = (0,1))
sc_output.fit(input_set[:,output_col])
 
test_actual = sc_output.inverse_transform(y_test)

results = pd.DataFrame(columns=['feature_col', 'r2_score', 'mse_score'])
count = 0

for i in range(no_of_feature):
    for j in range(len(combination[i])):
        feature = np.array(combination[i][j])
        model = save_load.load_model(path_name + "/" + str(count))
        
        pred_test_scaled = model.predict(X_test[:, :, feature])
        test_predict = sc_output.inverse_transform(pred_test_scaled)

        model_accuracy_r2 = r2_score(test_actual, test_predict)
        model_accuracy_mse = mean_squared_error(test_actual, test_predict)
        print("feature: {}\n r2_score: {}\n mse_score: {}\n".format(feature, model_accuracy_r2, model_accuracy_mse))
        #plot.time_series_plot(test_close, test_predict, 'red', 'blue', 'actual_close', \
        #         'predicted_close', 'days', 'price', 'Neural Network (multiple attributes - train data)')

        results.loc[count] = [feature, model_accuracy_r2, model_accuracy_mse]
        count = count + 1

results.to_excel("./result/feature_importance_close.xlsx")

