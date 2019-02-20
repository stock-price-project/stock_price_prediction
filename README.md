# Stock Price Prediction

## Introduction
Stock Price Prediction is one of the most difficult task to do, in financial sector. There are many factors involved in the prediction like physical factors vs. physhological, rational and irrational behaviour, etc. All these aspects combine to make share prices volatile and very difficult to predict with a high degree of accuracy. But we can predict a similar trend using previous data trends of a company's stock and to do so we will use a LSTM model to mimick the strategy used by quant's.

<img src="https://github.com/stock-price-project/stock_price_prediction/blob/master/output.png" width="500px">

## Dataset
We are using Google's stock dataset.   
* Training data [link](https://github.com/stock-price-project/stock_price_prediction/blob/master/train.csv)
* Testing data [link](https://github.com/stock-price-project/stock_price_prediction/blob/master/test.csv)

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

## Saved Models
* model.h5 and model.json   
Input is only single attribute __"Open"__ price and predicts __"Open"__ price.
* model1.h5 and model1.json                                                                                                                                                                                                                                           
Input are 2 attributes __"Open" "Close"__ price and predicts __"Open"__ price.

## Accuracy
For test set
* r2 score = 0.78885  
* mse score = 316.776

For train set
* r2 score = 0.97971
* mse score = 1563.557
