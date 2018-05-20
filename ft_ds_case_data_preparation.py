# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:55:54 2018

@author: Ilja
"""

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import StandardScaler

#all the helper functions below
def drop_rows_and_cols_with_NA_below_thresholds(input_df, key_names, col_thresh=0.20, row_thresh=0.05):
    df = input_df.copy(deep=True)
    
    number_of_cols = len(list(df.columns))
    row_threshold_integer = round(row_thresh * number_of_cols)
    df = df.dropna(axis=0, thresh=row_threshold_integer) # droping rows that have non-NA cell count below threshold
    
    number_of_rows = len(df)
    col_threshold_integer = round(col_thresh * number_of_rows)
    output_df = df.dropna(axis=1, thresh=col_threshold_integer).loc[:] # droping columns that have non-NA cell count below threshold
    return output_df


def get_col_names_without_target(dataframe, target):
    column_names_list = list(dataframe.columns)
    if target in column_names_list:
        column_names_list.remove(target)
    return column_names_list


def replace_commas_with_dots_in_string(single_string):
    if type(single_string) == str:
        single_string = single_string.replace(',','.')
    return single_string


def replace_spec_chars_with_space_in_string(single_string):
    if type(single_string) == str:
        single_string = single_string.replace('+',' ')
        single_string = single_string.replace('_',' ')
    return single_string


def strings_to_lower_case(col):
    if col.dtype=='O':
        col = col.str.lower()
    return col        


def convert_floats_in_string_to_floats(element):
    if type(element) == str:
        try:
            return float(element)
        except (ValueError, TypeError):
            return element
    return element


def get_common_predictors_list(predictor_list_1, predictor_list_2):
    common_predictor_set = set(predictor_list_1).intersection(set(predictor_list_2))
    common_predictor_list = list(common_predictor_set)
    return common_predictor_list


def string_dates_to_sec_after_epoch_as_float(input_col):
    if input_col.dtype=='O': #in pandas dataframe columns containing Strings, has type Object, or 'O'
        col_datetime=pd.to_datetime(input_col, format='%Y-%m-%d %H:%M', errors='ignore') #convert to datetime only if format is '%Y-%m-%d %H:%M'. Otherwise keep it unchanged
        if col_datetime.dtype=='datetime64[ns]': 
            epoch_timestamp_col = col_datetime - dt.datetime(1970, 1, 1)
            sec_float_col = epoch_timestamp_col / np.timedelta64(1, 's')
            return sec_float_col
        return col_datetime
    else:
        return input_col           


def remove_string_cols_with_unique_value_count_over_threshold(input_df, unique_count_threshold = 50):
    df = input_df.copy(deep=True)
    unique_counts_in_string_cols_series = df.select_dtypes(include=[object]).nunique() #string is 'object' type
    string_cols_over_threshold_series = unique_counts_in_string_cols_series[unique_counts_in_string_cols_series>unique_count_threshold]
    list_of_string_cols_over_threshold = list(string_cols_over_threshold_series.keys())
    df = df.drop(list_of_string_cols_over_threshold, axis=1)
    return df


def normalize_df(input_df, target):
    dataset_filled = input_df.copy(deep=True)
    
    y = pd.DataFrame(dataset_filled[target].values)  #saving target column separately, as it should not be transformed (we need 1 and 0 for classification)
    predictors = get_col_names_without_target(dataset_filled, target)
    
    scaler = StandardScaler() #creating an instance fo StandardScaler
    scaler.fit(dataset_filled[predictors]) #normalizing only predictors
    dataset_normalized_np_array = scaler.transform(dataset_filled[predictors].values) #creating numpy dataframe
    dataset_normalized = pd.DataFrame(dataset_normalized_np_array, columns=predictors) #converting numpy to pandas dataframe. Adding columns
    dataset_normalized[target] = y #adding target back
    dataset_normalized = dataset_normalized[[target] + predictors] #for convenience putting target as the first column
    return dataset_normalized


def exclude_similar_features_and_get_unique(corr_with_target_series, similarity_thresh = 0.000001):
    #getting 2 arrays from series - 1 array without the first element, and another without the last element
    corr_with_target_array_less_first_el = corr_with_target_series.tail(len(corr_with_target_series)-1).values
    corr_with_target_array_less_last_el = corr_with_target_series.head(len(corr_with_target_series)-1).values
    
    #calculating absoluste differences between these arrays, which is equivalent to finding difference between
    #subsequent elements in the original series
    corr_with_target_array_subseq_el_differences = np.abs(
            corr_with_target_array_less_first_el - corr_with_target_array_less_last_el)
    
    #finding which elements has differences over similarity_thresh
    corr_with_target_array_subseq_el_differences_over_thresh = (
            corr_with_target_array_subseq_el_differences > similarity_thresh )
    
    #we will keep the first element, so assume set it to True to keep
    first_el_diff = np.array([True])
    corr_with_target_array_subseq_el_differences_over_thresh = np.append(
            first_el_diff, corr_with_target_array_subseq_el_differences_over_thresh)
    
    #geetting boolean list that shows which elements are unique and thus should stay and which to be excluded
    corr_with_target_array_subseq_el_differences_over_thresh_bool_list = list(
            corr_with_target_array_subseq_el_differences_over_thresh)
    
    #getting a list of unique features
    corr_with_target_series_unique = (
            corr_with_target_series[corr_with_target_array_subseq_el_differences_over_thresh_bool_list])
    unique_feature_list = list(corr_with_target_series_unique.index)
    return unique_feature_list


def keep_unique_features_in_df(input_df, target):
    dataset_normalized = input_df.copy(deep=True)
    corr_matrix = dataset_normalized.corr()
    corr_with_target_abs_desc = corr_matrix[target].apply(np.abs).sort_values(ascending=False)
    unique_features = exclude_similar_features_and_get_unique(corr_with_target_abs_desc)
    dataset_normalized_unique_features=dataset_normalized[unique_features] 
    return dataset_normalized_unique_features


#this is the main function
def data_setup(input_df, key_names, target, col_thresh=0.60, row_thresh=0.05, 
               test_set_flag=False, predictor_list_for_test_set=[]):
    
    dataset_full_not_cleaned = input_df.copy(deep=True)
    dataset_full_not_cleaned_keys_dropped = dataset_full_not_cleaned.drop(key_names, axis=1)
    dataset_full = drop_rows_and_cols_with_NA_below_thresholds(dataset_full_not_cleaned_keys_dropped, 
                                                               key_names, col_thresh=col_thresh, 
                                                               row_thresh=row_thresh)
        
    dataset_full = dataset_full.applymap(replace_commas_with_dots_in_string)
    dataset_full = dataset_full.applymap(replace_spec_chars_with_space_in_string)
    dataset_full = dataset_full.apply(strings_to_lower_case)
    dataset_full = dataset_full.applymap(convert_floats_in_string_to_floats)
        
    dataset_full_time_converted = dataset_full.apply(string_dates_to_sec_after_epoch_as_float)
    
    if not test_set_flag:
        dataset_full_with_some_string_cols_removed = remove_string_cols_with_unique_value_count_over_threshold(
            dataset_full_time_converted, unique_count_threshold = 50)
    else:
        dataset_full_with_some_string_cols_removed = dataset_full_time_converted
    
    dataset_full_with_dummies = pd.get_dummies(dataset_full_with_some_string_cols_removed, dummy_na=True)
    dataset_filled = dataset_full_with_dummies.fillna(dataset_full_with_dummies.median())
    dataset_normalized = normalize_df(dataset_filled, target)
    
    if not test_set_flag:
        dataset_normalized_unique_features = keep_unique_features_in_df(dataset_normalized, target)
    else:
        predictors = get_col_names_without_target(dataset_normalized, target)
        common_predictor_list = get_common_predictors_list(predictors, predictor_list_for_test_set)
        dataset_normalized_unique_features = dataset_normalized[common_predictor_list]
    
    return dataset_normalized_unique_features
    
