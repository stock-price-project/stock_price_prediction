'''
This script helps in comparing the results of the following methods used for 
predicting the closing price of the stock.
'''
# importing libraries
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from utils import plot


# loading results
single_df = pd.read_csv('./result/prediction_single_result.csv')
multiple_df = pd.read_csv('./result/prediction_multiple_result.csv')
regression_df = pd.read_csv('./result/prediction_regression_result.csv')
svm_df = pd.read_csv('./result/prediction_svm_result.csv')

# storing actual close
actual_close = single_df['actual_close']

# storing predicted values
single_close = single_df['predicted_close']
multiple_close = multiple_df['predicted_close']
regression_close = regression_df['predicted_close']
svm_close = svm_df['predicted_close']

# storing errors
single_error = single_df['error_value']
multiple_error = multiple_df['error_value']
regression_error = regression_df['error_value']
svm_error = svm_df['error_value']

# analysis
# r2 score
single_r2_score = r2_score(actual_close, single_close)
multiple_r2_score = r2_score(actual_close, multiple_close)
regression_r2_score = r2_score(actual_close, regression_close)
svm_r2_score = r2_score(actual_close, svm_close)

# mse score
single_mse_score = sqrt(mean_squared_error(actual_close, single_close))
multiple_mse_score = sqrt(mean_squared_error(actual_close, multiple_close))
regression_mse_score = sqrt(mean_squared_error(actual_close, regression_close))
svm_mse_score = sqrt(mean_squared_error(actual_close, svm_close))

# combining results
predictions = [single_close, multiple_close, regression_close, svm_close]
r2_scores = [single_r2_score, multiple_r2_score, regression_r2_score, svm_r2_score]
mse_scores = [single_mse_score, multiple_mse_score, regression_mse_score, svm_mse_score]
labels = ["single attr. lstm", "multiple attr. lstm", "poly regression", "SVR"]

# comparison graph
plot.compare_plot(actual_close, predictions, labels)
# comparing mse value
plot.bar_plot(mse_scores, labels, len(mse_scores), "MSE value comparison", "MSE value")
# comparing r2 score
plot.bar_plot(r2_scores, labels, len(r2_scores), 'R2 value comparison', 'R2 score')

# creating dataframe
actual_close_df = pd.DataFrame(actual_close)
single_close_df = pd.DataFrame(single_close)
multiple_close_df = pd.DataFrame(multiple_close)
regression_close_df = pd.DataFrame(regression_close)
svm_close_df = pd.DataFrame(svm_close)
single_error_df = pd.DataFrame(single_error)
multiple_error_df = pd.DataFrame(multiple_error)
regression_error_df = pd.DataFrame(regression_error)
svm_error_df = pd.DataFrame(svm_error)

# saving results for prediction table
combined_df = pd.concat([actual_close_df, single_close_df, single_error_df, \
                         multiple_close_df, multiple_error_df, regression_close_df, \
                         regression_error_df, svm_close_df, svm_error_df], axis=1)
combined_df.to_csv('./result/prediction_comparison_result.csv', index = False)
print("results saved to csv file")


# saving results for error table
score_dict = {'lstm single': np.around([single_r2_score, single_mse_score], 6), \
              'lstm multiple': np.around([multiple_r2_score, multiple_mse_score], 6), \
              'poly regression': np.around([regression_r2_score, regression_mse_score], 6), \
              'svr': np.around([svm_r2_score, svm_mse_score], 6)}
              
combined_score_df = pd.DataFrame(score_dict, index=['r2 score', 'rmse score'])
combined_score_df.to_csv('./result/error_comparison_result.csv', index = True)
print("results saved to csv file")

