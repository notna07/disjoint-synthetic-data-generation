# Description: Script for executing a big loop to make, time, privacy, and utility measurements
# Author: Anton D. Lautrup
# Date: 27-11-2024

import os
import time
import pandas as pd

import sys
sys.path.append('.')

from typing import Dict
from pandas import DataFrame

from itertools import product
from joblib import Parallel, delayed

from synthcity.plugins import Plugins
from sklearn.ensemble import RandomForestClassifier

from disjoint_generative_model import DisjointGenerativeModels
from disjoint_generative_model.utils.joining_validator import JoiningValidator
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator, Concatenating
from disjoint_generative_model.utils.generative_model_adapters import generate_synthetic_data

from syntheval import SynthEval

Plugins().list()

def count_number_of_missing_items(path, model_name, data_name, num_parts, num_reps):
    """ Read the file if it exists and return the number of missing elements to be computed """
    try :
        df = pd.read_csv(path)
        assert (df['model'].str.contains(model_name).any() and df['data'].str.contains(data_name).any())
    except:
        return product([model_name], [data_name], range(1,num_parts+1), range(num_reps))
    else:
        all_items = product([model_name], [data_name], range(1,num_parts+1), range(num_reps))
        missing_items = []
        for item in all_items:
            if not df[(df['model']==item[0]) & (df['data']==item[1]) & (df['partitions']==item[2]) & (df['rep_idx']==item[3])].any().any():
                missing_items.append(item)
        return missing_items

def worker(iterable: tuple, train_data: Dict[str, DataFrame],  test_data: Dict[str, DataFrame], target_vars: Dict[str,str], results_file: str, metrics: Dict[str, dict], joining_strategy: str) -> None:
    """ Worker function for generating synthetic data and evaluating it. """
    
    model_name, data_name, num_parts, rep_idx = iterable

    df_train = train_data[data_name]
    df_test = test_data[data_name]
    target_var = target_vars[data_name]
    
    SE = SynthEval(df_train, df_test, verbose=False)

    start = time.time()
    if num_parts == 1:
        df_temp = generate_synthetic_data(df_train, model_name, id = num_parts*100+rep_idx*10)
    else:
        if joining_strategy == 'validate':
            parameter_grid = {'n_estimators': [10, 50, 100], 
                              'max_depth': [5, 10, 15, None], 
                              'criterion': ['gini', 'entropy', 'log_loss']}
            
            JV = JoiningValidator(classifier_model_base=RandomForestClassifier(), 
                                    model_parameter_grid=parameter_grid,  
                                    calibration_method='sigmoid',
                                    save_proba=True,
                                    verbose=False)
            
            JS = UsingJoiningValidator(JV, behaviour='adaptive')
        else:
            JS = Concatenating()

        dgms = DisjointGenerativeModels(df_train, num_parts*[model_name], joining_strategy=JS, worker_id = num_parts*100+rep_idx*10)

        if joining_strategy == 'adaptive': dgms.join_multiplier = 4

        df_temp = dgms.fit_generate()[:len(df_train)]
    end = time.time()
    
    print("generation finished, now evaluating")
    res = SE.evaluate(df_temp, analysis_target_var=target_var, **metrics)

    res_dict = {"model": model_name, "data": data_name, "partitions": num_parts, "rep_idx": rep_idx, "time": end-start}
    res.index = res['metric']
    res_dict.update(res['val'].to_dict())

    if os.path.exists(results_file):
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False, mode='a', header=False)
    else:
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False)
    pass

def make_data(models, train_data, test_data, target_vars, num_parts, num_reps, results_file, metrics, joining_strategy: str):
    """ Make the data for the time, privacy and utility figure """

    for model_name in models:
        for data_name in train_data.keys():
            missing_items = count_number_of_missing_items(results_file, model_name, data_name, num_parts, num_reps)
            Parallel(n_jobs=1)(delayed(worker)(item, train_data, test_data, target_vars, results_file, metrics, joining_strategy) for item in missing_items)
    pass

if __name__ == '__main__':

    NUM_REPEATS = 10
    MAX_PARTITIONS = 4

    metrics = {
        "pca"       : {},
        "h_dist"    : {},
        "corr_diff" : {"mixed_corr": True},
        "auroc_diff" : {"model": "rf_cls"},
        "cls_acc"   : {"F1_type": "macro"},
        "eps_risk"  : {},
        "mia"       : {"num_eval_iter": 5},
    }

    joining_strategy = 'concat'

    models = ['synthpop','datasynthesizer', 'ctgan']

    train_data = {
        'al':pd.read_csv('experiments/datasets/alzheimers_train.csv'),
        'bc':pd.read_csv('experiments/datasets/breast_cancer_train.csv'),
        'cc':pd.read_csv('experiments/datasets/cervical_cancer_train.csv'),
        # 'dm' : pd.read_csv('experiments/datasets/diabetic_mellitus_train.csv'),
        'hd':pd.read_csv('experiments/datasets/heart_train.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_train.csv'),
        'kd':pd.read_csv('experiments/datasets/kidney_disease_train.csv'),
        'st':pd.read_csv('experiments/datasets/stroke_train.csv'),
        }

    test_data = {
        'al':pd.read_csv('experiments/datasets/alzheimers_test.csv'),
        'bc':pd.read_csv('experiments/datasets/breast_cancer_test.csv'), 
        'cc':pd.read_csv('experiments/datasets/cervical_cancer_test.csv'),
        # 'dm' : pd.read_csv('experiments/datasets/diabetic_mellitus_test.csv'),
        'hd':pd.read_csv('experiments/datasets/heart_test.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_test.csv'),
        'kd':pd.read_csv('experiments/datasets/kidney_disease_test.csv'),
        'st':pd.read_csv('experiments/datasets/stroke_test.csv'),
        }

    target_vars = {
        'al':'Diagnosis',
        'bc':'Status', 
        'cc':'Biopsy',
        # 'dm': 'TYPE',
        'hd':'target',
        'hp':'b_class',
        'kd':'class',
        'st':'stroke',
        }

    if joining_strategy == 'concat':
        results_file = 'experiments/results/01_utility_privacy_concat.csv'
    elif joining_strategy == 'validate':
        results_file = 'experiments/results/01_utility_privacy_validate.csv'
    else:
        raise ValueError("joining_strategy must be either 'concat' or 'validate'")

    res = make_data(models, train_data, test_data, target_vars, MAX_PARTITIONS, NUM_REPEATS, results_file, metrics, joining_strategy)