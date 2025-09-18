# -*- coding: utf-8 -*-
"""
This file is part of the EStreams catalogue/dataset. See https://github.com/ 
for details.

Coded by: Thiago Nascimento
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_correlation_score(target_column, data_subset, range_exclude, model='linear'):
    """
    Compute the R2 coefficient for a given target column and data subset.

    Parameters:
        target_column (str): Name of the target column.
        data_subset (DataFrame): DataFrame containing the subset of data.
        range_exclude(int): Range of variables to be excluded from the MLR. Originally it excludes also the 
        streamflow signatures, so the analysis encompasses only climatic and landscape attributes. 
        model (str): Choice of model. 'linear' for Linear Regression (default) or 'random_forest' for Random Forest.
    Returns:
        DataFrame: DataFrame with R2 coefficient as a row.
    """

    data_subset.dropna(inplace=True)

    # Extract target variable and features
    y = data_subset[target_column]
    X = data_subset.iloc[:, range_exclude:]

    # Check if there are enough samples to compute R2 coefficient
    if len(X) <= 1:
        return pd.DataFrame({'Feature': ['R2_coefficient'], 'Coefficient': [np.nan]})
    
    # Define preprocessing steps
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Define column transformer to apply preprocessing only to numeric features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ])

    # Define the pipeline based on the chosen model
    if model == 'linear':
        regression_model = LinearRegression()
    elif model == 'random_forest':
        regression_model = RandomForestRegressor()
    else:
        raise ValueError("Invalid model choice. Use 'linear' for Linear Regression or 'random_forest' for Random Forest.")

    # Define the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regression_model', regression_model)])

    # Fit the pipeline
    pipeline.fit(X, y)
    
    # Compute R2 coefficient
    y_pred = pipeline.predict(X)
    r2_coefficient = r2_score(y, y_pred)
    
    return r2_coefficient

def find_unique_nested_catchments(df):
    """
    Return a list with unique nested catchemnts within the given initial list. When dealing with nested catchments
    groups that have intersection between each other, this code assumes the nested group with the higher number of 
    nested catchments. 

    Parameters:
        df (DataFrame): The DataFrame containing rows with lists of values.

    Returns:
        list: A list of indices of rows with the maximum number of unique values.
    """
    max_unique_rows = []  # List to store the indices of rows with maximum number of values
    col_name = df.columns[0]  # Get the name of the column containing the lists of values
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        #current_row_values = row[col_name].split(', ')  # Convert string to list of values
        current_row_values = row[col_name]
        max_unique_count = len(current_row_values)  # Initialize the maximum count to the number of values in the current row
        
        # Loop through other rows to find potential overlapping rows
        for other_index, other_row in df.iterrows():
            if index != other_index:  # Skip the current row
                other_row_values = other_row[col_name]
                #other_row_values = other_row[col_name].split(', ')  # Convert string to list of values
                
                # Find the intersection of values between the current row and the other row
                intersection = set(current_row_values).intersection(other_row_values)
                if intersection:  # If there's any intersection
                    other_row_count = len(other_row_values)
                    if other_row_count > max_unique_count:  # If the other row has more values than the current maximum
                        max_unique_count = other_row_count  # Update the maximum count
        
        # Add the index of the current row to the list if it has the maximum count of values
        if len(current_row_values) == max_unique_count:
            max_unique_rows.append(index)
    
    return max_unique_rows

def find_directly_connected_catchments(df, starting_catchment):
    """
    Return a list of indices of rows with catchments that are somehow directly connected to the given starting catchment.
    For example, you can go from the selected headwater (starting_catchment) until the outlet of the watershed. 

    Parameters:
        df (DataFrame): The DataFrame containing rows with lists of values.
        starting_catchment (str): The starting catchment.

    Returns:
        list: A list of indices of rows with catchments that are somehow directly connected to the starting catchment.
    """
    connected_rows = []  # List to store indices of connected rows
    col_name = df.columns[0]  # Get the name of the column containing the lists of values
    
    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        #catchments = row[col_name].split(', ')  # Convert string to list of values
        catchments = row[col_name]

        # Check if the starting catchment is in the current row
        if starting_catchment in catchments:
            connected_rows.append(index)  # Add the index of the current row to the list
    
    return connected_rows

def find_max_unique_rows(df):
    """
    Identify rows with the maximum number of values that are unique, i.e., not repeated in other rows.

    Parameters:
        df (DataFrame): The DataFrame containing rows with lists of values.

    Returns:
        list: A list of indices of rows with the maximum number of unique values.
    """
    max_unique_rows = []  # List to store indices of rows with the maximum unique values
    col_name = df.columns[0]  # Assume the values are in the first column

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        current_row_values = row[col_name]  # Extract values for the current row
        max_unique_count = len(current_row_values)  # Initialize max count with the current row's values count
        
        # Loop through other rows to find overlaps
        for other_index, other_row in df.iterrows():
            if index != other_index:  # Skip comparing the row to itself
                other_row_values = other_row[col_name]  # Extract values for the other row

                # Find common values between current and other rows
                intersection = set(current_row_values).intersection(other_row_values)
                
                # Update max count if the other row has more values and overlaps with the current row
                if intersection:
                    other_row_count = len(other_row_values)
                    if other_row_count > max_unique_count:
                        max_unique_count = other_row_count
        
        # Add current row index if its value count matches the maximum found
        if len(current_row_values) == max_unique_count:
            max_unique_rows.append(index)
    
    return max_unique_rows
