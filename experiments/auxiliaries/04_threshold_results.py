# Description: Script for creating data for using different validators for joining the results of the experiments.
# Author: Anton D. Lautrup
# Date: 24-01-2025

import os
import sys
import time
sys.path.append('.')

import pandas as pd

from pandas import DataFrame
from typing import Dict, List

from itertools import product
from joblib import Parallel, delayed

from disjoint_generative_model.utils.joining_validator import JoiningValidator, OneClassValidator, OutlierValidator
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator
from disjoint_generative_model import DisjointGenerativeModels

from syntheval import SynthEval

from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from diffprivlib.models import RandomForestClassifier as DPRandomForestClassifier

def count_number_of_missing_items(path, exp_name, validator_name, data_name, num_reps, threshold):
    """ Read the file if it exists and return the number of missing elements to be computed """
    try :
        df = pd.read_csv(path)
        assert (df['experiment'].str.contains(exp_name).any() and df['data'].str.contains(data_name).any())
    except:
        return product([exp_name], [validator_name], [threshold], [data_name], range(num_reps))
    else:
        all_items = product([exp_name], [validator_name], [threshold], [data_name], range(num_reps))
        missing_items = []
        for item in all_items:
            if not df[(df['experiment']==item[0]) & (df['validator']==item[1]) & (df['threshold']==item[2]) & (df['data']==item[3]) & (df['rep_idx']==item[4])].any().any():
                missing_items.append(item)
        return missing_items

def worker(iterable: tuple, models: List[str], train_data: Dict[str,DataFrame], test_data: Dict[str,DataFrame], target_vars: Dict[str,str], validator: object, results_file: str, metrics: Dict[str, dict]) -> None:
    """ Worker function for generating synthetic data and evaluating it. """

    experiment_name, validator_name, threshold, data_name, rep_idx = iterable

    df_train = train_data[data_name]
    df_test = test_data[data_name]
    target_var = target_vars[data_name]

    SE = SynthEval(df_train, df_test, verbose=False)

    cat_atts = df_train.select_dtypes(include='object').columns.tolist()

    for att in [att for att in df_train.columns if att not in cat_atts]:
        if len(df_train[att].unique()) <= 5:
            cat_atts.append(att)

    num_atts = [att for att in df_train.columns if att not in cat_atts]

    gen_models = {models[0]: cat_atts, models[1]: num_atts}

    start = time.time()
    JS = UsingJoiningValidator(join_validator_model=validator, patience=10, threshold=threshold, threshold_decay=0)
    dgms = DisjointGenerativeModels(df_train, gen_models, joining_strategy=JS, worker_id = rep_idx*10)
    dgms.join_multiplier = 12   # set this high to ensure enough sucessfull joins

    df_temp = dgms.fit_generate()[:len(df_train)]
    end = time.time()
    
    assert df_temp.shape == df_train.shape, f"Shape of generated data is not the same as the training data. {df_temp.shape} != {df_train.shape}"

    res = SE.evaluate(df_temp, analysis_target_var=target_var, **metrics)

    res_dict = {"experiment": experiment_name, "validator": validator_name, "threshold": threshold, "data": data_name, "rep_idx": rep_idx, "time": end-start}
    res.index = res['metric']
    res_dict.update(res['val'].to_dict())

    if os.path.exists(results_file):
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False, mode='a', header=False)
    else:
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False)
    pass

def make_data(experiment_name, models, train_data, test_data, target_var, validators, thresholds, num_reps, results_file, metrics):
    """ Make the data for the validator case study """
    for threshold in thresholds:
        for validator_name in validators.keys():
            for data_name in train_data.keys():
                missing_items = count_number_of_missing_items(results_file, experiment_name, validator_name, data_name, num_reps, threshold)
                Parallel(n_jobs=-1)(delayed(worker)(item, models, train_data, test_data, target_var, validators[validator_name], results_file, metrics) for item in missing_items)
    pass

if __name__ == '__main__':
    
    NUM_REPS = 10

    experiment_name = 'sp_ad'

    gen_models = ['synthpop', 'adsgan']

    thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]

    metrics = {
        "pca"       : {},
        "h_dist"    : {},
        "corr_diff" : {"mixed_corr": True},
        "auroc_diff" : {"model": "rf_cls"},
        "cls_acc"   : {"F1_type": "macro"},
        "eps_risk"  : {},
        "mia"       : {"num_eval_iter": 5},
        }

    validator_models = {
        # 'rf(base)': JoiningValidator(verbose=False),
        # 'lgbm': JoiningValidator(classifier_model_base=LGBMClassifier(), verbose=False),
        # 'knn': JoiningValidator(classifier_model_base=KNeighborsClassifier(), verbose=False),
        # 'svm': JoiningValidator(classifier_model_base=SVC(probability=True, gamma='auto'), verbose=False),
        # 'mlp': JoiningValidator(classifier_model_base=MLPClassifier(max_iter=1000), verbose=False),
        # 'dp_rf': JoiningValidator(classifier_model_base=DPRandomForestClassifier(n_estimators=100), verbose=False),
        'ocsvm': OneClassValidator(verbose=False),
        'out_if': OutlierValidator(verbose=False)
        }

    train_data = {
        # 'al':pd.read_csv('experiments/datasets/alzheimers_train.csv'),
        # 'bc':pd.read_csv('experiments/datasets/breast_cancer_train.csv'), 
        # 'cc':pd.read_csv('experiments/datasets/cervical_cancer_train.csv'),
        # 'hd':pd.read_csv('experiments/datasets/heart_train.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_train.csv'),
        # 'kd':pd.read_csv('experiments/datasets/kidney_disease_train.csv'),
        # 'st':pd.read_csv('experiments/datasets/stroke_train.csv'),
        }

    test_data = {
        # 'al':pd.read_csv('experiments/datasets/alzheimers_test.csv'),
        # 'bc':pd.read_csv('experiments/datasets/breast_cancer_test.csv'), 
        # 'cc':pd.read_csv('experiments/datasets/cervical_cancer_test.csv'),
        # 'hd':pd.read_csv('experiments/datasets/heart_test.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_test.csv'),
        # 'kd':pd.read_csv('experiments/datasets/kidney_disease_test.csv'),
        # 'st':pd.read_csv('experiments/datasets/stroke_test.csv'),
        }

    target_vars = {
        # 'al':'Diagnosis',
        # 'bc':'Status', 
        # 'cc':'Biopsy',
        # 'hd':'target',
        'hp':'b_class',
        # 'kd':'class',
        # 'st':'stroke',
        }

    results_file = 'experiments/results/validator_case_study/full_threshold_results.csv'

    # print(list(count_number_of_missing_items(results_file, experiment_name, 'lgbm', 'hp', NUM_REPS, 0.2)))

    res = make_data(experiment_name,
                    gen_models,
                    train_data,
                    test_data,
                    target_vars,
                    validator_models,
                    thresholds,
                    NUM_REPS,
                    results_file,
                    metrics
                    )