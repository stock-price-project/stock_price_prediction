'''
This script is for saving and loading the models
'''
import os
from keras.models import model_from_json


def save_model(path_name, model):
    # serialize model to JSON
    model_json = model.to_json()
    #os.makedirs(path_name)
    with open(path_name + "/model.json", "w") as json_file:
        json_file.write(model_json)
        
    # serialize weights to HDF5
    model.save_weights(path_name + "/model.h5")
    print("Saved model to disk")


def load_model(path_name):
    # loading the model
    json_file = open(path_name + '/model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(path_name + "/model.h5")
    print("Loaded model from disk")
    return loaded_model
