# Description: Overhaul of the original experiment to be based off the same model
# Author: Anton D. Lautrup
# Date: 21-03-2025

import os
import sys
import time
import pickle
sys.path.append('.')

import pandas as pd

from pandas import DataFrame
from typing import Dict, List

from disjoint_generative_model.utils.joining_validator import JoiningValidator, OneClassValidator, OutlierValidator
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator
from disjoint_generative_model import DisjointGenerativeModels

from disjoint_generative_model import DisjointGenerativeModels
from disjoint_generative_model.utils.joining_validator import JoiningValidator
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator

from syntheval import SynthEval

from lightgbm import LGBMClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

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
                    validator_model,
                    validator_model_name,
                    thresholds,
                    percentage: bool,
                    NUM_REPS: int,
                    results_file: str,
                    metrics: dict):
    
    cat_atts = train_data.select_dtypes(include='object').columns.tolist()

    for att in [att for att in train_data.columns if att not in cat_atts]:
        if len(train_data[att].unique()) <= 5:
            cat_atts.append(att)

    num_atts = [att for att in train_data.columns if att not in cat_atts]

    SE = SynthEval(train_data, test_data, verbose=False, cat_cols=cat_atts)

    JS = UsingJoiningValidator(validator_model, max_iter=250, patience=20, threshold_decay=0)
    dgms = DisjointGenerativeModels(train_data, generative_models=gen_models, prepared_splits={'cats': cat_atts, 'nums': num_atts}, joining_strategy=JS)
    dgms.join_multiplier = 16

    _ = dgms.fit_generate()
    
    dgms._make_calibration_plot(test_data, save=True, stats=False)

    for threshold in thresholds:
        for i in range(NUM_REPS):
            print(f"Threshold: {threshold}, Repetition: {i}")
            
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

                metadata = (experiment_name, validator_model_name, used_threshold, dataset_name, i)

                _evaluate_synthetic_data(SE, syn_temp, label, metrics, metadata, results_file)
            except Exception as e:
                print(f"Error: {e}")
                continue
        
if __name__ == '__main__':
    NUM_REPS = 10

    experiment_name = 'sp_ad'

    gen_models = ['synthpop', 'adsgan']

    percentage = False
    thresholds = [0 ,0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 0.95]

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

    # validator_model_name = 'rf_scikit'
    # validator_model = JoiningValidator(classifier_model_base=RandomForestClassifier(), 
    #                                     calibration_method=None,
    #                                     save_proba=True,
    #                                     verbose=False)
    # validator_model_name = 'rf_optimised'
    # with open('experiments/validator_models/hp_rf_optimised.obj', 'rb') as file:
    #     validator_model = pickle.load(file)
    #     validator_model.pre_fit = True

    validator_model_name = 'rf_undcon'
    with open('experiments/validator_models/hp_rf_undcon.obj', 'rb') as file:
        validator_model = pickle.load(file)

    # validator_model_name = 'rf_ovrcon'
    # with open('experiments/validator_models/hp_rf_ovrcon.obj', 'rb') as file:
    #     validator_model = pickle.load(file)
    
    dataset_name = 'hp'
    train_data = pd.read_csv('experiments/datasets/hepatitis_train.csv')
    test_data = pd.read_csv('experiments/datasets/hepatitis_test.csv')

    label ='b_class'

    results_file = 'experiments/results/validator_case_study/full_threshold_results.csv'

    res = generate_results(experiment_name,
                            gen_models,
                            dataset_name,
                            train_data,
                            test_data,
                            label,
                            validator_model,
                            validator_model_name,
                            thresholds,
                            percentage,
                            NUM_REPS,
                            results_file,
                            metrics)