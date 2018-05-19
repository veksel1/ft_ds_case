# -*- coding: utf-8 -*-
"""
Created on Sat May 19 14:44:29 2018

@author: Ilja
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from ft_ds_case_data_preparation import data_setup, get_col_names_without_target

def print_col_row_and_cell_count(df):
    row_count, column_count = df.shape
    element_count = column_count*row_count
    print('column count:  ', column_count)
    print('row count:     ', row_count)
    print('element count: ', element_count)
    print()
    
    
def print_scores(model, X_features, Y_target):
    scores = cross_val_score(model, X_features, Y_target, cv=5)
    print('accuracies =', scores)
    print('mean accuracy =', scores.mean())   


def main():
    # parameters
    key1, key2 = 'key1', 'key2'
    key_names = [key1, key2]
    target = 'response'
    
    #TASK 1 - combining datasets    
    name_dataset_0 = 'app_dataset.csv'
    name_dataset_1 = 'dataset_1.csv'
    name_dataset_2 = 'dataset_2.csv'
        
    dataset_0 = pd.read_csv(name_dataset_0, sep=';')
    dataset_1 = pd.read_csv(name_dataset_1, sep=';')
    dataset_2 = pd.read_csv(name_dataset_2, sep=';')
        
    dataset_0_and_1 = pd.merge(dataset_0, dataset_1, how='left', on=key2)    
    dataset_full_not_cleaned = pd.merge(dataset_0_and_1, dataset_2, how='left', on=key1)
    dataset_full_not_cleaned.to_csv('output_dataset_full_not_cleaned.csv', sep=';')
    
    #TASK 2
    #transforming the dataset    
    dataset_normalized_unique_features = data_setup(dataset_full_not_cleaned, key_names, target, col_thresh=0.60, row_thresh=0.05)
    
    #preparing and training logistic regression model
    full_set = dataset_normalized_unique_features
    train, test = train_test_split(full_set, test_size=0.2, random_state=1)
    predictors = get_col_names_without_target(full_set, target)
    log_reg_final = LogisticRegression(penalty='l1', C=0.1, fit_intercept=True, random_state=1)
    print()
    print('Performance of Logistic regression model:')
    print_scores(log_reg_final, train[predictors], train[target])
    log_reg_final.fit(full_set[predictors], full_set[target])
    
    #getting the most important factors
    coef_array = log_reg_final.coef_
    coef_series = pd.Series(coef_array[0], predictors)
    sorted_coefficients = coef_series.apply(np.abs).sort_values(ascending=False)
    print()
    print('Predictors sorted from the most important to unimportant:')
    print(sorted_coefficients)
    
    #TASK 3 
    #preparing model for predictions
    Random_Forest_model = RandomForestClassifier(max_depth=26, min_samples_split=10, 
                                             n_estimators=50, n_jobs=-1, random_state=1)
    print()
    print('Performance of Random Forest model:')
    print_scores(Random_Forest_model, train[predictors], train[target])
     
    #preparing data for which predictions should be made
    #set_to_predict_raw = train_test_split(dataset_full_not_cleaned,   # this is an example. In production can read file from CSV instead
    #                                      test_size=0.3, random_state=1)[1]
    set_to_predict_raw = pd.read_csv('set_to_predict.csv', sep=';') # uncomment to read a file from CSV
    
    dataset_full_not_cleaned_appended_to_set_to_predict_raw = set_to_predict_raw.append(dataset_full_not_cleaned)
    transformed_combined_set = data_setup(dataset_full_not_cleaned_appended_to_set_to_predict_raw,
                                          key_names, target, col_thresh=0.60, row_thresh=0, test_set_flag=True,
                                          predictor_list_for_test_set = predictors)
    transformed_set_to_predict = transformed_combined_set[:len(set_to_predict_raw)].copy(deep=True)
    
    #adjusting the model to the fact that the test case can have a different columns
    predictors_for_test = get_col_names_without_target(transformed_set_to_predict, target)
    Random_Forest_model.fit(full_set[predictors_for_test], full_set[target])
    
    #making a predictions       
    predictions = Random_Forest_model.predict(transformed_set_to_predict[predictors_for_test])
    transformed_set_to_predict = transformed_set_to_predict.assign(predictions=predictions)
    transformed_set_to_predict.to_csv('out_predictions.csv', sep=';')  
    
if __name__ == "__main__":
    main()
