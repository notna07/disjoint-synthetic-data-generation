# Description: Script for running a bunch of noise injection baselines for comparison with our method
# Author: Anton D. Lautrup
# Date: 19-03-2026

import os
import time
import pandas as pd
import numpy as np

import sys
sys.path.append('.')

from typing import Dict
from pandas import DataFrame

from itertools import product
from joblib import Parallel, delayed

from synthcity.plugins import Plugins
from disjoint_generation.utils.generative_model_adapters import generate_synthetic_data

from syntheval import SynthEval

Plugins().list()

def count_number_of_missing_items(path, model_name, exp_name, data_name, exp_values, num_reps):
    """ Read the file if it exists and return the number of missing elements to be computed """
    try :
        df = pd.read_csv(path)
        assert (df['model'].str.contains(model_name).any() and df['data'].str.contains(data_name).any())
    except:
        return product([model_name], [exp_name], [data_name], exp_values, range(num_reps))
    else:
        all_items = product([model_name], [exp_name], [data_name], exp_values, range(num_reps))
        missing_items = []
        for item in all_items:
            if not df[(df['model']==item[0]) & (df['exp_name']==item[1]) & (df['data']==item[2]) & (df['degree']==item[3]) & (df['rep_idx']==item[4])].any().any():
                missing_items.append(item)
        return missing_items

def shuffle_values(df, degree, seed):
    """Shuffle each column independently for a shared subset of rows."""
    df_shuffled = df.copy()
    num_rows = len(df)
    num_to_shuffle = int(num_rows * degree)
    if num_to_shuffle <= 1:
        return df_shuffled

    rng = np.random.default_rng(seed)
    indices_to_shuffle = rng.choice(df.index.to_numpy(), size=num_to_shuffle, replace=False)

    for col in df.columns:
        col_values = df.loc[indices_to_shuffle, col].to_numpy(copy=True)
        shuffled_values = col_values[rng.permutation(num_to_shuffle)]
        df_shuffled.loc[indices_to_shuffle, col] = shuffled_values

    return df_shuffled

def replace_values(df, degree, seed):
    """ Randomly replace a percentage of values in the dataframe with the mode of the column. """
    df_replaced = df.copy()
    num_rows = len(df)
    num_to_replace = int(num_rows * degree)
    if num_to_replace <= 1:
        return df_replaced
    
    rng = np.random.default_rng(seed)
    for col in df.columns:
        mode_value = df[col].mode()[0]
        indices_to_replace = rng.choice(df.index.to_numpy(), size=num_to_replace, replace=False)
        df_replaced.loc[indices_to_replace, col] = mode_value

    return df_replaced

def add_gaussian_noise(df, degree, seed):
    """ Add Gaussian noise to numerical columns in the dataframe. The noise is scaled by the 
    standard deviation of each column and the specified degree. """
    df_noisy = df.copy()
    rng = np.random.default_rng(seed)
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.floating):
            std_dev = df[col].std()
            noise = rng.normal(loc=0, scale=std_dev * degree, size=len(df))
            df_noisy[col] += noise
    return df_noisy

def worker(iterable: tuple, train_data: Dict[str, DataFrame],  test_data: Dict[str, DataFrame], target_vars: Dict[str,str], results_file: str, metrics: Dict[str, dict]) -> None:
    """ Worker function for generating synthetic data and evaluating it. """
    model_name, exp_name, data_name, degree, rep_idx = iterable

    df_train = train_data[data_name]
    df_test = test_data[data_name]
    target_var = target_vars[data_name]
    
    SE = SynthEval(df_train, df_test, verbose=False)

    start = time.time()
    if exp_name == 'random_shuffle':
        df_noisy = shuffle_values(df_train, degree, seed=int(degree*100)+rep_idx*10)
    elif exp_name == 'random_replace':
        df_noisy = replace_values(df_train, degree, seed=int(degree*100)+rep_idx*10)
    elif exp_name == 'gaussian_noise':
        df_noisy = add_gaussian_noise(df_train, degree, seed=int(degree*100)+rep_idx*10)

    df_temp = generate_synthetic_data(df_noisy, model_name, id = int(degree*100)+rep_idx*10)
    end = time.time()

    res = SE.evaluate(df_temp, analysis_target_var=target_var, **metrics)

    res_dict = {"model": model_name, "exp_name": exp_name, "data": data_name, "degree": degree, "rep_idx": rep_idx, "time": end-start}
    res.index = res['metric']
    res_dict.update(res['val'].to_dict())

    if os.path.exists(results_file):
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False, mode='a', header=False)
    else:
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False)
    pass

def make_data(models, train_data, test_data, target_vars, experiments, num_reps, results_file, metrics):
    """ Make the data for noisy baselines. """
    
    for model_name in models:
        for exp_name, exp_values in experiments.items():
            for data_name in train_data.keys():
                missing_items = list(count_number_of_missing_items(results_file, model_name, exp_name, data_name, exp_values, num_reps))
                num_tasks = len(missing_items)
                if num_tasks == 0:
                    continue

                print(f"[progress] model={model_name} exp={exp_name} data={data_name}: {num_tasks} tasks")
                Parallel(n_jobs=6)(delayed(worker)(item, train_data, test_data, target_vars, results_file, metrics) for item in missing_items)
    pass

if __name__ == '__main__':

    NUM_REPEATS = 10
    
    experiments = {
        'random_shuffle': [0.1, 0.25, 0.5, 0.9],
        'random_replace': [0.1, 0.25, 0.5, 0.9],
        'gaussian_noise': [0.1, 0.25, 0.5, 0.9],
    }

    metrics = {
        "pca"       : {},
        "h_dist"    : {},
        "corr_diff" : {"mixed_corr": True},
        "auroc_diff" : {"model": "rf_cls"},
        "cls_acc"   : {"F1_type": "macro"},
        "eps_risk"  : {},
        "mia"       : {"num_eval_iter": 5},
    }

    models = ['synthpop', 'datasynthesizer', 'ctgan']

    train_data = {
        'al':pd.read_csv('experiments/datasets/alzheimers_train.csv'),
        'bc':pd.read_csv('experiments/datasets/breast_cancer_train.csv'),
        'cc':pd.read_csv('experiments/datasets/cervical_cancer_train.csv'),
        'hd':pd.read_csv('experiments/datasets/heart_train.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_train.csv'),
        'kd':pd.read_csv('experiments/datasets/kidney_disease_train.csv'),
        'st':pd.read_csv('experiments/datasets/stroke_train.csv'),
        }

    test_data = {
        'al':pd.read_csv('experiments/datasets/alzheimers_test.csv'),
        'bc':pd.read_csv('experiments/datasets/breast_cancer_test.csv'), 
        'cc':pd.read_csv('experiments/datasets/cervical_cancer_test.csv'),
        'hd':pd.read_csv('experiments/datasets/heart_test.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_test.csv'),
        'kd':pd.read_csv('experiments/datasets/kidney_disease_test.csv'),
        'st':pd.read_csv('experiments/datasets/stroke_test.csv'),
        }

    target_vars = {
        'al':'Diagnosis',
        'bc':'Status', 
        'cc':'Biopsy',
        'hd':'target',
        'hp':'b_class',
        'kd':'class',
        'st':'stroke',
        }


    results_file = 'experiments/results/01_noise_baseline.csv'
    res = make_data(models, train_data, test_data, target_vars, experiments, NUM_REPEATS, results_file, metrics)