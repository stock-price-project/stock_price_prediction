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
from sklearn.metrics import r2_score
from utils import train
from utils import save_load
from utils import plot


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


# loading testing data
df_test = pd.read_csv('./dataset/test.csv')

# including the avg attribute in the test set
df_test = pd.concat([df_test, pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])], axis=1)
test_set = df_test.iloc[:, [1, 4, 6, 7]].values

# feature scaling
sc_t = MinMaxScaler(feature_range = (0,1))
test_set_scaled = sc_t.fit_transform(test_set)

# creating X_test, y_test
X_test = []
y_test = [] 
for i in range(60, len(test_set_scaled)):
    X_test.append(test_set_scaled[i-60: i])
    y_test.append(test_set_scaled[i, [0,1]])
    
# converting to numpy array
X_test = np.array(X_test)
y_test = np.array(y_test)

# creating 3D tensor
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 4))

# ============================================================================

'''
first loop for training model with open price
second loop for training model with close price
'''

combination = []
columns_index = [0, 1, 2, 3]

from itertools import combinations
for i in range(1,len(columns_index)+1):
    combination.append(list((combinations(columns_index, i))))

count = 0
epochs = 120
no_of_features = 4

for i in range(no_of_features):
    for j in range(len(combination[i])):
        feature = np.array(combination[i][j])
        model = train.training(X_train[:, :, feature], y_train, feature.shape[0] , epochs)
        path_name = "./model/optimisation" 
        # Saving the model
        save_load.save_model(path_name + "/" + str(count), model)
        count = count + 1

# =============================================================================


path_name = "./model/optimisation" 
 
test_close = test_set[60:len(test_set),1]
count = 0
max_accuracy = 0
for i in range(no_of_features):
    for j in range(len(combination[i])):
        feature = np.array(combination[i][j])
        model = save_load.load_model(path_name + "/" + str(count))
        #model = train.training(X_train[:, :, feature], y_train, feature.shape[0] , epochs)
        predicted_value = model.predict(X_test[:, :, feature])
        scpred = MinMaxScaler(feature_range = (0,1))
        scpred = scpred.fit(test_set[:,1].reshape(-1,1)) 
        test_predict = scpred.inverse_transform(predicted_value)
        model_accuracy = r2_score(test_close, test_predict)
        print("model_accuracy : ", model_accuracy)
        plot.time_series_plot(test_close, test_predict, 'red', 'blue', 'actual_close', \
                 'predicted_close', 'days', 'price', 'Neural Network (multiple attributes - train data)')
        if model_accuracy > max_accuracy:
            max_accuracy = model_accuracy
            max_model = feature
            
        count = count + 1

print("max accuarcy : ", max_accuracy)
print("feature column : ", max_model)



