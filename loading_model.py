#If used custom loss or metric to compile model then load with compile as False and later recompile it otherwise set compile as True
from tensorflow import keras

def get_model(model_name, compile = False):
    return keras.models.load_model(model_name, compile=False)