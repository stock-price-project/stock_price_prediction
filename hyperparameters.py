'''
this script is for hyperparameter optimisation.
'''

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
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
training_set = df.iloc[:, [4]].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

'''
creating a data structure with 60 timestamp and predicting 1 output, later it is
reshaped, resulting in 3D tensor
'''


# loading testing data
df_test = pd.read_csv('./dataset/test.csv')

# including the avg attribute in the test set
df_test = pd.concat([df_test, pd.DataFrame((df_test['High'] + df_test['Low'])/2, columns=['Avg.val'])], axis=1)
test_set = df_test.iloc[:, [4]].values

# feature scaling
sc_t = MinMaxScaler(feature_range = (0,1))
test_set_scaled = sc_t.fit_transform(test_set)


epochs = [90, 120, 150]
neurons = [5, 20, 60]
optimiser = ['adam', 'rmsprop']
activation = ['linear', 'tanh', 'relu', 'sigmoid']

count = 0
no_of_features = 1

# ============================================================================
'''
first loop for training model with open price
second loop for training model with close price
'''
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

for epoch in epochs:
    for neuron in neurons:
        X_train = []
        y_train = []
        for i in range(neuron, len(training_set_scaled)-1):
            X_train.append(training_set_scaled[i-neuron: i])     
            y_train.append(training_set_scaled[i, 0])
        # Feature Scaling
        sc = MinMaxScaler(feature_range = (0,1))
        training_set_scaled = sc.fit_transform(training_set)
        # converting to numpy array
        X_train, y_train = np.array(X_train), np.array(y_train)
        # Reshaping to create 3D tensor
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], no_of_features)

        for optim in optimiser:
            for func in activation:
                    model = Sequential()
                    # adding the first lstm layer and some dropout regularization
                    model.add(LSTM(units = X_train.shape[1], return_sequences = True, input_shape = (X_train.shape[1], no_of_features)))
                    model.add(Dropout(0.2))

                    model.add(LSTM(units = 100, return_sequences = True))
                    model.add(Dropout(0.2))

                    model.add(LSTM(units = 100, return_sequences = True))
                    model.add(Dropout(0.2))

                    model.add(LSTM(units = 40))
                    model.add(Dropout(0.2))

                    model.add(Dense(units = 1, activation = func))

                    # compiling the rnn
                    model.compile(optimizer = optim, loss = 'mean_squared_error')

                    # fitting the rnn to the training set
                    model.fit(X_train, y_train, epochs = epoch, batch_size = 32)

                    path_name = "./model/hyperParaModels" 
                    # Saving the model
                    
                    save_load.save_model(path_name + "/" + str(count), model)
                    count = count + 1


# =============================================================================

path_name = "./model/hyperParaModels" 

 
results = pd.DataFrame(columns=['epoch', 'neuron', 'optim', 'activation', 'r2_score', 'MSE'])
max_accuracy = 0
count=0
for epoch in epochs:
    for neuron in neurons:
        # creating X_test, y_test
        X_test = []
        y_test = [] 
        test_close = test_set[neuron:len(test_set), 0]
        
        for i in range(neuron, len(test_set_scaled)):
            X_test.append(test_set_scaled[i-neuron: i])
            y_test.append(test_set_scaled[i, 0])
            
        # converting to numpy array
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        # creating 3D tensor
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_of_features))
        
        for optim in optimiser:
            for func in activation:
                model = save_load.load_model(path_name + "/" + str(count))
                predicted_value = model.predict(X_test)
                scpred = MinMaxScaler(feature_range = (0,1))
                scpred = scpred.fit(test_close.reshape(-1,1)) 
                test_predict = scpred.inverse_transform(predicted_value)
                model_accuracy_r2 = r2_score(test_close, test_predict)
                model_accuracy_mse = mean_squared_error(test_close, test_predict)
                print("r2 : ", model_accuracy_r2)
                print("mse : ", model_accuracy_mse)
                plot.time_series_plot(test_close, test_predict, 'red', 'blue', 'actual_close', \
                         'predicted_close', 'days', 'price', 'Neural Network (multiple attributes - train data)')
                count = count +1
                
                results.loc[count] = [epoch, neuron, optim, func, model_accuracy_r2, model_accuracy_mse]

results.to_excel("./model/hyperParaModels/output.xlsx")