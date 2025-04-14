# Description: Script for executing a big loop to make, time, privacy, and utility measurements
# Author: Anton D. Lautrup
# Date: 04-04-2025

import os
import time
import pandas as pd

import sys
sys.path.append('.')

from typing import List, Dict
from pandas import DataFrame

from itertools import product
from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier

from disjoint_generative_model import DisjointGenerativeModels
from disjoint_generative_model.utils.joining_validator import JoiningValidator
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator, Concatenating
from disjoint_generative_model.utils.generative_model_adapters import generate_synthetic_data

from syntheval import SynthEval

def count_number_of_missing_items(path, model_name, data_name, exp_name, num_parts, num_reps):
    """ Read the file if it exists and return the number of missing elements to be computed """
    try :
        df = pd.read_csv(path)
        assert (df['model'].str.contains(model_name).any() and df['data'].str.contains(data_name).any() and df['experiment'].str.contains(exp_name).any())
    except:
        return product([model_name], [data_name], [exp_name], range(1,num_parts+1), range(num_reps))
    else:
        all_items = product([model_name], [data_name], [exp_name], range(1,num_parts+1), range(num_reps))
        missing_items = []
        for item in all_items:
            if not df[(df['model']==item[0]) & (df['data']==item[1]) & (df['experiment']==item[2]) & (df['partitions']==item[3]) & (df['rep_idx']==item[4])].any().any():
                missing_items.append(item)
        return missing_items

def worker(iterable: tuple, df_train: DataFrame,  df_test: DataFrame, target_var: str, results_file: str, metrics: Dict[str, dict]) -> None:
    """ Worker function for generating synthetic data and evaluating it. """
    
    model_name, data_name, exp_name, num_parts, rep_idx = iterable

    parameter_grid = {
        'n_estimators': [10, 20, 50, 100], 
        'max_depth': [5, 10, 15, 20, None], 
        'criterion': ['gini', 'entropy', 'log_loss']
        }
    
    SE = SynthEval(df_train, df_test, verbose=False)

    start = time.time()
    if num_parts == 1:
        df_temp = generate_synthetic_data(df_train, model_name, id = num_parts*100+rep_idx*10)
    else:
        match exp_name:
            case 'concat':
                dgms = DisjointGenerativeModels(df_train, 
                                                generative_models=num_parts*[model_name], 
                                                prepared_splits='random',
                                                joining_strategy=Concatenating(), 
                                                worker_id = num_parts*100+rep_idx*10
                                                )
            case 'val_random':
                JV = JoiningValidator(classifier_model_base=RandomForestClassifier(), 
                                    model_parameter_grid=parameter_grid,  
                                    calibration_method='sigmoid',
                                    save_proba=True,
                                    verbose=False
                                    )

                dgms = DisjointGenerativeModels(df_train, 
                                                generative_models=num_parts*[model_name], 
                                                prepared_splits='random',
                                                joining_strategy=UsingJoiningValidator(JV, behaviour='adaptive'), 
                                                worker_id = num_parts*100+rep_idx*10
                                                )
            case 'val_corr':
                JV = JoiningValidator(classifier_model_base=RandomForestClassifier(), 
                                    model_parameter_grid=parameter_grid,  
                                    calibration_method='sigmoid',
                                    save_proba=True,
                                    verbose=False
                                    )
                
                dgms = DisjointGenerativeModels(df_train, 
                                                generative_models=num_parts*[model_name], 
                                                prepared_splits='correlated',
                                                joining_strategy=UsingJoiningValidator(JV, behaviour='adaptive'),
                                                worker_id = num_parts*100+rep_idx*10
                                                )

        dgms.join_multiplier = 4

        df_temp = dgms.fit_generate()[:len(df_train)]
    end = time.time()
    
    print("generation finished, now evaluating")

    res = SE.evaluate(df_temp, analysis_target_var=target_var, **metrics)

    res_dict = {"model": model_name, "data": data_name, "experiment": exp_name, "partitions": num_parts, "rep_idx": rep_idx, "time": end-start}
    res.index = res['metric']
    res_dict.update(res['val'].to_dict())

    if os.path.exists(results_file):
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False, mode='a', header=False)
    else:
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False)
    pass

def make_data(model: str, 
              dataset_name: str, 
              experiment_series: List[str],
              train_data: DataFrame,
              test_data: DataFrame,
              target_var: str, 
              metrics: dict, 
              results_file: str,
              num_reps: int, 
              num_parts: int) -> None:
    """ Make the data for the line plots """

    for experiment_name in experiment_series:
        missing_items = count_number_of_missing_items(results_file, model, dataset_name, experiment_name, num_parts, num_reps)
        Parallel(n_jobs=6)(delayed(worker)(item, train_data, test_data, target_var, results_file, metrics) for item in missing_items)
    pass

if __name__ == '__main__':

    NUM_REPS = 20
    MAX_PARTITIONS = 10

    metrics = {
        "pca"       : {},
        "h_dist"    : {},
        "corr_diff" : {"mixed_corr": True},
        "auroc_diff" : {"model": "rf_cls"},
        "cls_acc"   : {"F1_type": "macro"},
        "eps_risk"  : {},
        "mia"       : {"num_eval_iter": 5},
    }

    experiment_series = ['concat', 'val_random', 'val_corr']

    model = 'datasynthesizer'

    dataset_name = 'dm'

    train_data = pd.read_csv('experiments/datasets/diabetic_mellitus_train.csv')
    test_data = pd.read_csv('experiments/datasets/diabetic_mellitus_test.csv')

    target_var = 'TYPE'

    results_file = 'experiments/results/02_validator_vs_concat.csv'

    res = make_data(model, dataset_name, experiment_series, train_data, test_data, target_var, metrics, results_file, NUM_REPS, MAX_PARTITIONS)