import numpy as np 
import pandas as pd 

def range_function(column):
    range = column.max() - column.min()
    return range
