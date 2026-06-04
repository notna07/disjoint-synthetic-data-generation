# Description: Script for repeating the results of the previous experiment on 
#   additional datasets and save to a format that can easily be plotted
# Author: Anton D. Lautrup
# Date: 11-12-2024

import os
import pandas as pd

import sys
import warnings
sys.path.append('.')

import pickle, json

import numpy as np
import pandas as pd

from pandas import DataFrame
from typing import List, Dict

from joblib import Parallel, delayed

from syntheval import SynthEval

from sklearn.ensemble import RandomForestClassifier
from disjoint_generation import DisjointGenerativeModels
from disjoint_generation.utils.joining_validator import JoiningValidator
from disjoint_generation.utils.joining_strategies import UsingJoiningValidator

from disjoint_generation.utils.generative_model_adapters import generate_synthetic_data

NUM_EXP = 10

class CompatibilityUnpickler(pickle.Unpickler):
    """Unpickler that remaps old module names to new ones for compatibility."""
    def find_class(self, module, name):
        # Remap old module name prefixes to new ones
        old_module = 'disjoint_generative_model'
        new_module = 'disjoint_generation'
        
        if module == old_module:
            module = new_module
        elif module.startswith(old_module + '.'):
            # Handle submodules like disjoint_generative_model.utils.X
            module = module.replace(old_module, new_module, 1)
        
        return super().find_class(module, name)


def worker(data_name:str, df_train: DataFrame, df_test: DataFrame, model: str, id: int, target_var: str, results_file: str, metrics) -> None:
    """ Worker function for generating synthetic data and evaluating it. """

    SE = SynthEval(df_train, df_test, verbose=False)

    match model:
        case 'dpgan':
            df_temp = generate_synthetic_data(df_train, model, id = np.random.randint(0, 100))
        case 'synthpop':
            df_temp = generate_synthetic_data(df_train, model, id = np.random.randint(0, 100))
        case 'datasynthesizer':
            df_temp = generate_synthetic_data(df_train, model, id = np.random.randint(0, 100))
        case 'tabdiff':
            df_temp = generate_synthetic_data(df_train, model, id = np.random.randint(0, 100))
        case 'dgms':           
            # JV = JoiningValidator(classifier_model_base=RandomForestClassifier(), 
            #                         model_parameter_grid=parameter_grid,  
            #                         calibration_method='sigmoid',
            #                         save_proba=False,
            #                         verbose=False)
            
            with open(f'disjoint-synthetic-data-generation/experiments/validator_models/{data_name}_rf_opt.obj', 'rb') as file:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    joining_validator = CompatibilityUnpickler(file).load()
                # joining_validator = pickle.load(file)
            JS = UsingJoiningValidator(joining_validator, behaviour='adaptive')

            with open(f'disjoint-synthetic-data-generation/experiments/results/saved_partitionings/{data_name}_corr_parts.json', 'r') as file:
                prepared_splits = json.load(file)

            dgms = DisjointGenerativeModels(df_train, [models[0], models[1]], prepared_splits=prepared_splits,
                                             joining_strategy=JS, parallel_worker_id = np.random.randint(0, 100))
            dgms.join_multiplier = 4

            df_temp = dgms.fit_generate()[:len(df_train)]
        case _:
            raise ValueError(f"Model {model} not recognized for generating synthetic data.")

    
    res = SE.evaluate(df_temp, analysis_target_var=target_var, **metrics)

    res_dict = {"dataset": model}
    res.index = res['metric']
    res_dict.update(res['val'].to_dict())

    if os.path.exists(results_file):
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False, mode='a', header=False)
    else:
        res = pd.DataFrame(res_dict, index=[0])
        res.to_csv(results_file, index=False)
    pass


def check_specified_splits_for_mixed_model(models: List[str], data_name_key: str, df_train: DataFrame, df_test: DataFrame, target_var: str, metrics):
    """ Check the performance of the mixed model setup on a random split of the dataset. """

    SE = SynthEval(df_train, df_test, verbose=False)

    results_file = f'disjoint-synthetic-data-generation/experiments/results/03_mixed_models_results/{models[0]}_{models[1]}_{data_name_key}.csv'

    # Check if the results file exists
    if os.path.exists(results_file):
        results_df = pd.read_csv(results_file)
    else:
        results_df = pd.DataFrame(columns=['dataset'])

    missing_counts = {model: max(0, NUM_EXP - results_df[results_df['dataset'] == model].shape[0]) for model in models+['dgms']}

    res = Parallel(n_jobs=5)(delayed(worker)(data_name_key, df_train, df_test, model, i, target_var, results_file, metrics) for model in missing_counts.keys() for i in range(missing_counts[model]))
    pass


if __name__ == '__main__':

    models = ['tabdiff', 'dpgan']

    metrics = {
        "pca"       : {},
        "h_dist"    : {},
        "corr_diff" : {"mixed_corr": True},
        "auroc_diff": {"model": "rf_cls"},
        "cls_acc"   : {"F1_type": "macro"},
        "eps_risk"  : {},
        "dcr"       : {},
        "mia"       : {"num_eval_iter": 5},
        }

    train_data = {
        'al':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/alzheimers_train.csv'),
        'bc':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/breast_cancer_train.csv'), 
        'cc':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/cervical_cancer_train.csv'),
        'hd':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/heart_train.csv'),
        'hp': pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/hepatitis_train.csv'),
        'kd':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/kidney_disease_train.csv'),
        'st':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/stroke_train.csv'),
        }

    test_data = {
        'al':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/alzheimers_test.csv'),
        'bc':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/breast_cancer_test.csv'), 
        'cc':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/cervical_cancer_test.csv'),
        'hd':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/heart_test.csv'),
        'hp': pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/hepatitis_test.csv'),
        'kd':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/kidney_disease_test.csv'),
        'st':pd.read_csv('disjoint-synthetic-data-generation/experiments/datasets/stroke_test.csv'),
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

    for key in train_data.keys():
        res = check_specified_splits_for_mixed_model(models, key, train_data[key], test_data[key], target_vars[key], metrics)