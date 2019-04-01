'''
This script is for plotting the time series graph
'''
import numpy as np
import matplotlib.pyplot as plt

def time_series_plot(actual, predict, actual_color, predict_color, label_actual, label_predict, xlabel, ylabel, title):
    plt.plot(actual , color= actual_color, label = label_actual)
    plt.plot(predict, color= predict_color, label = label_predict)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc='best')
    plt.show()  
    
def bid_plot(bid_list):
    plt.plot(bid_list[0] , color='red', label="predicted open")
    plt.plot(bid_list[1], color='blue', label="predicted close")
    plt.title("Predicted Opening - Closing per day")
    plt.xlabel('days')
    plt.ylabel('price')
    plt.legend()
    plt.show()
    
    plt.plot(bid_list[1]-bid_list[0], color='red', label='bid_value')
    plt.hlines(0, 0, len(bid_list[0]), colors='blue', linestyles='dashed', label = "reference open price")
    plt.title("bidding risk graph")
    plt.xlabel('days')
    plt.ylabel('bidding range')
    plt.legend()
    plt.show()
    
def error_plot(error_list, title, label):
    plt.plot(error_list, color='red', label=label)
    plt.hlines(0, 0, len(error_list), colors='blue', linestyles='dashed', label = "perfect prediction")
    plt.title(title)
    plt.xlabel('days')
    plt.ylabel('difference')
    plt.legend()
    plt.show()
    
def compare_plot(actual, predictions, labels):
    plt.plot(actual, color='red', label = "actual_close")
    plt.plot(predictions[0], label = labels[0])
    plt.plot(predictions[1], label = labels[1])
    plt.plot(predictions[2], label = labels[2])
    plt.plot(predictions[3], label = labels[3])
    plt.xlabel("days")
    plt.ylabel("price")
    plt.title("comparision graph")
    plt.legend(loc='best')
    plt.show()


def bar_plot(score, labels, no_of_scores, title, xlabel):
    x = np.arange(no_of_scores)
    plt.barh(x, score)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.yticks(x, labels)
    plt.show()
    
    