## Model Information

* Input_Close : Open, Close, Volume, Avg.
* Output_Close : Close
* Input_Open : Close, Volume, Avg.
* Output_Open : Open
* Optimizer : adam
* Activation : tanh
* neuron : 80
* Epochs : 150

# for prediction of close price
### Train Score
* r2 score = 0.997514
* mse score = 191.477

### Test Score
* r2 score = 0.769922
* mse score = 528.859

# for prediction of open price
### Train Score
* r2 score = 0.999436
* mse score = 43.3748

### Test Score
* r2 score = 0.867536
* mse score = 311.462

