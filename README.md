# Stock Price Prediction

## Introduction
Stock Price Prediction is one of the most difficult task to do, in financial sector. There are many factors involved in the prediction like physical factors vs. psychological, rational and irrational behaviour, etc. All these aspects combine to make share prices volatile and very difficult to predict with a high degree of accuracy. But we can predict a similar trend using previous data trends of a company's stock and to do so we will use a LSTM model to mimic the strategy used by quants.

<img src="https://github.com/stock-price-project/stock_price_prediction/blob/master/model/single_attr_pred_open_from_open/output_test.png" width="500px">

## Dataset
We are using Google's stock dataset.   
* Training data [link](https://github.com/stock-price-project/stock_price_prediction/blob/master/train.csv)
* Testing data [link](https://github.com/stock-price-project/stock_price_prediction/blob/master/test.csv)

## Motivation
The motivation of this project was to implement the model using LSTM and to compare its performance with existing models.

## Installation
* Numpy  
<code>conda install numpy</code>
* Pandas  
<code>conda install pandas</code>
* Matplotlib  
<code>conda install matplotlib</code>
* Keras  
<code>conda install keras</code>
* Spyder IDE  
<code>conda install spyder</code>

## Table of Content
* single attribute prediction
  * predicted open price
  * predicted close price
* hyperparameter optimization
* feature importance
* multiple attribute prediction
  * predicted open price
  * predicted close price
* trading application
* future stock predictionults

## Results
for opening price prediction
* maximum r2 score on test set : 0.867536
* maximum r2 score on train set : 0.999436

for closing price prediction
* maximum r2 score on test set : 0.769922
* maximum r2 score on train set : 0.997514

