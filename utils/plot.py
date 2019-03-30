'''
This script is for plotting the time series graph
'''

import matplotlib.pyplot as plt

def time_series_plot(actual, predict, actual_color, predict_color, label_actual, label_predict, xlabel, ylabel, title):
    plt.plot(actual , color= actual_color, label = label_actual)
    plt.plot(predict, color= predict_color, label = label_predict)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()    
    
    