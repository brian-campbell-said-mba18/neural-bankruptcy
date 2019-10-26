import numpy as np
import pandas as pd 

class Preprocess():
    '''
    After the data has been separated into training and testing data,
    this class oversamples and normalizes the datasets.
    '''
    def __init__ (self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data


# References
