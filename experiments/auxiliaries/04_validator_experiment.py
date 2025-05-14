# Description: Script for creating data for using different validators for joining the results of the experiments.
# Author: Anonymous
# Date: 24-01-2025

import os
import sys
import time
import json
sys.path.append('.')

import pandas as pd

from pandas import DataFrame
from typing import Dict, List

from itertools import product
from joblib import Parallel, delayed

from disjoint_generative_model.utils.joining_validator import JoiningValidator, OneClassValidator, OutlierValidator
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator, Concatenating
from disjoint_generative_model import DisjointGenerativeModels

from syntheval import SynthEval


from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def _evaluate_synthetic_data(evaluator: SynthEval, syn_data, target_var, metrics, metadata, results_file):
    res = evaluator.evaluate(syn_data, analysis_target_var=target_var, **metrics)

    (experiment_name, validator_name, data_name, rep_idx) = metadata

    res_dict = {"experiment": experiment_name, "validator": validator_name, "data": data_name, "rep_idx": rep_idx}
    res.index = res['metric']
    res_dict.update(res['val'].to_dict())

    if os.path.exists(results_file):
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False, mode='a', header=False)
    else:
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False)
    pass

def generate_results(experiment_name,
                    gen_models,
                    dataset_name,
                    train_data,
                    test_data,
                    target_var,
                    validator_models,
                    rep_idx: int,
                    results_file: str,
                    metrics: dict) -> None:
    """ Generate the results for the validator case study """

    SE = SynthEval(train_data, test_data, verbose=False)

    with open(f'experiments/results/saved_partitionings/{dataset_name}_corr_parts.json', 'r') as file:
        prepared_splits = json.load(file)

    JS = Concatenating()
    dgms = DisjointGenerativeModels(train_data, generative_models=gen_models, prepared_splits=prepared_splits, joining_strategy=JS, parallel_worker_id = rep_idx*100)
    dgms.join_multiplier = 8

    syn_temp = dgms.fit_generate()

    for val_model_name, validator_model in validator_models.items():
        print(f"Validator model: {val_model_name}")

        metadata = (experiment_name, val_model_name, dataset_name, rep_idx)

        if validator_model is None:
            _evaluate_synthetic_data(SE, syn_temp, target_var, metrics, metadata, results_file)
        else:
            dgms.strategy = UsingJoiningValidator(validator_model, behaviour='adaptive', threshold='auto')
            dgms._strategy.join_validator.fit_classifier(dgms.training_data, num_batches_of_bad_joins=2)

            try:
                syn_temp = dgms.conduct_joining()
                syn_temp = dgms.dm.postprocess(syn_temp)
                print("Shape of data: ", syn_temp.shape)

                metadata = (experiment_name, val_model_name, dataset_name, rep_idx)

                _evaluate_synthetic_data(SE, syn_temp, target_var, metrics, metadata, results_file)
            except Exception as e:
                print(f"Error: {e}")
                continue


if __name__ == '__main__':
    NUM_REPS = 10

    experiment_name = 'sp_dp'

    gen_models = ['synthpop', 'dpgan']

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
        'concat': None,
        'rf': JoiningValidator(classifier_model_base=RandomForestClassifier(criterion='entropy', class_weight='balanced_subsample', n_estimators=20, max_depth=10), verbose=False, calibration_method='sigmoid'),
        'lgbm': JoiningValidator(classifier_model_base=LGBMClassifier(learning_rate=0.1, n_estimators=100, objective='binary'), verbose=False),
        'knn': JoiningValidator(classifier_model_base=KNeighborsClassifier(algorithm='kd_tree', n_neighbors=15, weights='distance'), verbose=False),
        'svm': JoiningValidator(classifier_model_base=make_pipeline(MinMaxScaler(), SVC(gamma='auto', probability=True, kernel='rbf', C=1000, class_weight='balanced')), verbose=False),
        'mlp': JoiningValidator(classifier_model_base=make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000,activation='tanh', alpha=0.0001, hidden_layer_sizes=(100, 100), tol=1e-05)), verbose=False, calibration_method='sigmoid'),
        # 'dp_rf': JoiningValidator(classifier_model_base=DPRandomForestClassifier(n_estimators=100), verbose=False),
        'ocsvm': OneClassValidator(verbose=False),
        'out_if': OutlierValidator(verbose=False)
        }

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

    results_file = 'experiments/results/04_validator_models.csv'

    for dataset_name in train_data.keys():
        Parallel(n_jobs=5)(delayed(generate_results)(experiment_name,
                            gen_models,
                            dataset_name,
                            train_data[dataset_name],
                            test_data[dataset_name],
                            target_vars[dataset_name],
                            validator_models,
                            i,
                            results_file,
                            metrics) for i in range(NUM_REPS))
