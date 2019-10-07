# This imports the necessary libraries.
import numpy as np 
import pandas as pd
import math

def xnames_creator():
    '''
    This function creates the names of the initial "X"
    feature values that will be used to create the names
    of the "X_missing" values.
    '''

    # This for loop creates the names of the feature variables
    # "X1-X64"
    num_list = list(range(1,65))
    x_var = "X"
    feature_list = []
    for i in num_list:
        feature_var = x_var + str(i)
        feature_list.append(feature_var)
    return feature_list

def missing_names_converter(vars_for_conv):
    '''
    This function uses a for loop to convert
    the names of the feature variables in the list
    vars_for_conv into the names of dummy variables.
    The conversion starts with the "feature" and
    turns it into "feature_missing."
    '''
    converted_features = []
    missing_str = '_missing'
    for i in vars_for_conv:
        new_name = i + missing_str
        converted_features.append(new_name)
    return converted_features

def actual_null_creator(df, x_names, missing_names):
    for i in range(0,len(x_names)):
        x = x_names[i]
        column = df[x]
        dummy_var = []
        for row in column:
            if str(row).lower() == "nan":
                dummy_var.append(1)
            else:
                dummy_var.append(0)
        unique_missing = missing_names[i]
        dummy_var = np.asarray(dummy_var)
        df[unique_missing] = dummy_var
    return df
    