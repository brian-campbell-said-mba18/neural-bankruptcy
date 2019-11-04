import numpy as np
import pandas as pd 

# This imports the algorithm that will resample the training data.
# This comes from Reference 1 in References.
from sklearn.utils import resample

def oversample(x_data, y_data):
    '''
    The purpose of this function is to oversample the minority
    cases of bankruptcy. Oversampling is used when the disparity
    between the minority and majority instances of a target variable
    are so great that the algorithm is at risk of simply picking
    the majority algorithm without giving much weight to the x 
    variables. This function returns a minority class that has a number
    of rows equal to that of the majority class.
    '''
    # This sets the random_state_variable for 
    # reproducable results.
    random_state_variable = 42

    # This converts the x and y training data
    # back to pandas dataframes.
    # This comes from Reference 1 in References.
    X_train_convert = pd.DataFrame(x_data)
    y_train_convert = pd.DataFrame(y_data)

    # This renames the y varaible to "bankrupt."
    y_train_convert = y_train_convert.rename(columns={0: "bankrupt"})

    # This concatenates the x and y training data.
    # This comes from Reference 1 in References.
    yr_concat = pd.concat([X_train_convert, y_train_convert],
    axis=1)

    # This separates the bankrupt cases from the non-bankrupt
    # cases.
    # This comes from Reference 1 in References.
    yes = yr_concat[(yr_concat['bankrupt'] == 1)]
    no = yr_concat[(yr_concat['bankrupt'] == 0)]

    # This oversamples the minority class, the instances of bankrtupt.
    # This comes from Reference 1 in References.
    yes_oversampled = resample(yes, replace = True, n_samples =
        len(no), random_state = random_state_variable)

    # This combines the not bankrupt cases with the oversampled
    # bankrupt cases.
    # This comes from Reference 1 in References.
    year_oversampled = pd.concat([no, yes_oversampled])


    # This splits the x variables from the y variable, "bankrupt".
    y_oversampled = np.array(year_oversampled['bankrupt'])
    X_oversampled = np.array(year_oversampled.drop(columns = 'bankrupt'))

    return X_oversampled, y_oversampled 

def drop_nulls(x_data, y_data):
    '''
    This concatenates the testing x and y values together
    and then drops all the null values. It then separates
    the x data from the y data.
    '''

    # This converts the x_data to a pandas dataframe.
    x_data = pd.DataFrame(x_data)

    # This concatenates the x and y testing data.
    # This comes from Reference 1 in References.
    data_concat = pd.concat([x_data, y_data],
    axis=1)

    # This drops all the null values in the testing data.
    # This comes from Reference 2 in References.
    data_concat = data_concat.dropna()

    # This seperates the x freatures from the target variable.
    new_y_data = data_concat['bankrupt']
    new_x_data = data_concat.drop(columns = 'bankrupt')

    # This converts new_x_data to a numpy array.
    new_x_data = np.array(new_x_data)

    # This returns the x testing and y testing data.
    return new_x_data, new_y_data

def null_test(np_array):
    df_test = pd.DataFrame(np_array)
    return df_test.info(verbose=True)
    
# References
# 1. https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18
# 2. https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html