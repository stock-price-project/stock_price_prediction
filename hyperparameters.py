'''
this script is for hyperparameter optimisation where we will get best pair of
hyperparameters.
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

columns = [1]

input_set = df.iloc[:, columns].values

sc = MinMaxScaler(feature_range = (0,1))
sc.fit(input_set)

'''
we have used open, close, volume, avg. data of the stock data here. Open price,
close price, volume and avg. is used as the input data trend
'''

no_of_features = 1
input_col = [0]
output_col = [0]

training_set = df_train.iloc[:, columns].values

# Feature Scaling
training_set_scaled = sc.transform(training_set)

# including the avg attribute in the test set



epochs = [90, 120, 150]
neurons = [5, 60, 80] 
optimiser = ['adam', 'rmsprop']
activation = ['linear', 'tanh', 'relu', 'sigmoid']

count = 0

# ============================================================================

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

for epoch in epochs:
    for neuron in neurons:
        X_train = []
        y_train = []
        for i in range(neuron, len(training_set_scaled)-1):
            X_train.append(training_set_scaled[i-neuron: i, input_col])     
            y_train.append(training_set_scaled[i, output_col])
            
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

count=0

for epoch in epochs:
    for neuron in neurons:
        # creating X_test, y_test
        X_test = []
        y_test = []
        testing_set = df_test.iloc[:, columns].values
        #test_output = testing_set_scaled[neuron:len(testing_set_scaled), output_col]
        x1 = pd.DataFrame(training_set[len(training_set)-neuron:])
        x2 = pd.DataFrame(testing_set)
        testing_set = np.array(pd.concat([x1, x2]))
        
        # feature scaling
        testing_set_scaled = sc.transform(testing_set)
        
        for i in range(neuron, len(testing_set_scaled)):
            X_test.append(testing_set_scaled[i-neuron: i, input_col])
            y_test.append(testing_set_scaled[i, output_col])
            
        # converting to numpy array
        X_test, y_test = np.array(X_test), np.array(y_test)
        
        # creating 3D tensor
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], no_of_features))
        
        for optim in optimiser:
            for func in activation:
                model = save_load.load_model(path_name + "/" + str(count))
                
                pred_test_scaled = model.predict(X_test)
                test_predict = sc.inverse_transform(pred_test_scaled)
                test_actual = sc.inverse_transform(y_test)
                
                model_accuracy_r2 = r2_score(test_actual, test_predict)
                model_accuracy_mse = mean_squared_error(test_actual, test_predict)
                print("r2 : ", model_accuracy_r2)
                print("mse : ", model_accuracy_mse)
                plot.time_series_plot(test_actual, test_predict, 'red', 'blue', 'actual_close', \
                         'predicted_close', 'days', 'price', 'Neural Network (multiple attributes - train data)')
                count = count +1
                
                results.loc[count] = [epoch, neuron, optim, func, model_accuracy_r2, model_accuracy_mse]

results.to_excel("./model/hyperParaModels/hyperparameter_optim.xlsx")

