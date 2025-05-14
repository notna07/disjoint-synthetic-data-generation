# Description: Overhaul of the original experiment to be based off the same model
# Author: Anonymous
# Date: 21-03-2025

import os
import sys
import pickle, json
sys.path.append('.')

import numpy as np
import pandas as pd

from disjoint_generative_model import DisjointGenerativeModels
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator, Concatenating

from syntheval import SynthEval

def _evaluate_synthetic_data(evaluator: SynthEval, syn_data, target_var, metrics, metadata, results_file):
    res = evaluator.evaluate(syn_data, analysis_target_var=target_var, **metrics)

    (experiment_name, validator_name, threshold, data_name, rep_idx) = metadata

    res_dict = {"experiment": experiment_name, "validator": validator_name, "threshold": threshold, "data": data_name, "rep_idx": rep_idx}
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
                    label,
                    validator_models,
                    validator_model_names,
                    thresholds,
                    percentage: bool,
                    NUM_REPS: int,
                    results_file: str,
                    metrics: dict):
    
    SE = SynthEval(train_data, test_data, verbose=False)

    with open(f'experiments/results/saved_partitionings/{dataset_name}_corr_parts.json', 'r') as file:
        prepared_splits = json.load(file)

    # JS = UsingJoiningValidator(validator_model, max_iter=250, patience=20, threshold_decay=0)
    JS = Concatenating()
    dgms = DisjointGenerativeModels(train_data, generative_models=gen_models, prepared_splits=prepared_splits, joining_strategy=JS)
    dgms.join_multiplier = 8

    _ = dgms.fit_generate()
    
    # dgms._make_calibration_plot(test_data, save=True, stats=False)

    for val_model, val_model_name in zip(validator_models, validator_model_names):
        print(f"Validator model: {val_model_name}")
        with open(f'experiments/validator_models/{val_model}.obj', 'rb') as file:
            validator_model = pickle.load(file)
        
        dgms.strategy = UsingJoiningValidator(validator_model, max_iter=250, patience=20, threshold_decay=0)

        for threshold in thresholds:
            # for i in range(NUM_REPS):
            print(f"Threshold: {threshold}, Repetition: {NUM_REPS}")
                
            if percentage:
                dgms._strategy.join_validator.threshold = 'auto'
                dgms._strategy.join_validator.auto_threshold_percentage = threshold
            else:
                dgms._strategy.join_validator.threshold = threshold

            try:
                syn_temp = dgms.conduct_joining()
                syn_temp = dgms.dm.postprocess(syn_temp)
                print("Shape of data: ", syn_temp.shape)
                used_threshold = dgms._strategy.join_validator.threshold

                metadata = (experiment_name, val_model_name, used_threshold, dataset_name, NUM_REPS)

                _evaluate_synthetic_data(SE, syn_temp, label, metrics, metadata, results_file)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
if __name__ == '__main__':
    NUM_REPS = 10

    experiment_name = 'sp_dp'

    gen_models = ['synthpop', 'dpgan']

    validator_models = ['hp_rf_scikit', 'hp_rf_optimised', 'hp_rf_undcon']
    validator_model_names = ['rf_scikit', 'rf_optimised', 'rf_subopt']

    percentage = False
    thresholds = np.linspace(0.0, 1.0, 21)

    metrics = {
        "pca"       : {},
        "h_dist"    : {},
        "corr_diff" : {"mixed_corr": True},
        "auroc_diff" : {"model": "rf_cls"},
        "cls_acc"   : {"F1_type": "macro"},
        "eps_risk"  : {},
        "dcr"       : {},
        "mia"       : {"num_eval_iter": 5},
        }

    dataset_name = 'hp'
    train_data = pd.read_csv('experiments/datasets/hepatitis_train.csv')
    test_data = pd.read_csv('experiments/datasets/hepatitis_test.csv')

    label ='b_class'

    results_file = 'experiments/results/validator_case_study/04_threshold_results.csv'

    for i in range(NUM_REPS):
        res = generate_results(experiment_name,
                            gen_models,
                            dataset_name,
                            train_data,
                            test_data,
                            label,
                            validator_models,
                            validator_model_names,
                            thresholds,
                            percentage,
                            i,
                            results_file,
                            metrics)