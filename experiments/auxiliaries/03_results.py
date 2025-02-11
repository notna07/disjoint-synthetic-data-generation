# Description: Script for runing the code of notebook 3 without the process being interrupted.
# Author: Anton D. Lautrup
# Date: 20-12-2024

### Imports
import sys
sys.path.append('.')

import pandas as pd

from pandas import DataFrame
from typing import List, Dict

from joblib import Parallel, delayed

from syntheval import SynthEval

from sklearn.ensemble import RandomForestClassifier
from disjoint_generative_model import DisjointGenerativeModels
from disjoint_generative_model.utils.joining_validator import JoiningValidator
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator
from disjoint_generative_model.utils.generative_model_adapters import generate_synthetic_data

### Constants
NUM_REPS = 10

### Metrics
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


### Load training and testing datasets and define categorical and numerical attributes
df_train = pd.read_csv('experiments/datasets/hepatitis_train.csv')
df_test = pd.read_csv('experiments/datasets/hepatitis_test.csv')

label = 'b_class'

cat_atts = df_train.select_dtypes(include='object').columns.tolist()

for att in [att for att in df_train.columns if att not in cat_atts]:
    if len(df_train[att].unique()) <= 5:
        cat_atts.append(att)

num_atts = [att for att in df_train.columns if att not in cat_atts]


def model_experiment(df_train: DataFrame, df_test: DataFrame, label: str, model: str, metrics: Dict[str, dict]) -> DataFrame:
    """ Function to do repeated runs of a generative model on the same 
    dataset and return best estimate of the mean of the various metrics. 
    """
    SE = SynthEval(df_train, df_test, verbose=False)

    dfs_list = Parallel(n_jobs=-1)(delayed(generate_synthetic_data)(df_train, model, id=i) for i in range(NUM_REPS))
    dfs = {f"rep_{i}": df_synth for i, df_synth in enumerate(dfs_list)}
    
    res, _ = SE.benchmark(dfs, analysis_target_var=label,**metrics, rank_strategy='summation')
    
    res = res.drop(columns=[col for col in res.columns if 'error' in col])
    res = res.drop(columns=['rank', 'u_rank', 'p_rank', 'f_rank'])
    res = res.droplevel(1, axis=1)

    mean_values = res.mean()
    sem_error = res.sem()
    results = pd.concat([mean_values, sem_error], axis=1, keys=['mean', 'sem'])

    return results

def _single_mixed_model_experiment(df_train: DataFrame, gms: Dict[str, List[str]], id) -> DataFrame:
    """ Function to do runs of the mixed model. """
    parameter_grid = {'n_estimators': [10, 50, 100], 'max_depth': [5, 10, 15, None], 'criterion': ['gini', 'entropy', 'log_loss']}
    JV = JoiningValidator(classifier_model_base=RandomForestClassifier(), 
                          model_parameter_grid=parameter_grid,  
                          calibration_method='sigmoid',
                          save_proba=True,
                          verbose=False)
    JS = UsingJoiningValidator(JV, behaviour='adaptive')

    dgms = DisjointGenerativeModels(df_train, gms, joining_strategy=JS, worker_id=id*10)
    dgms.join_multiplier = 4    # to ensure high enough resolution

    df_dgms = dgms.fit_generate()

    return df_dgms[:len(df_train)]

def mixed_model_experiment(df_train: DataFrame, df_test: DataFrame, model1: str, model2: str, 
                           cat_atts: List[str], num_atts: List[str], label: str, metrics: Dict[str, dict]) -> DataFrame:
    """ Function to do repeated runs of the mixed model. """
    
    SE = SynthEval(df_train, df_test, verbose=False)
    
    gms = {model1: cat_atts, model2: num_atts}
    dfs_list = Parallel(n_jobs=-1)(delayed(_single_mixed_model_experiment)(df_train, gms, id=i) for i in range(NUM_REPS))
    dfs = {f"rep_{i}": df_synth for i, df_synth in enumerate(dfs_list)}

    res, _ = SE.benchmark(dfs, analysis_target_var=label,**metrics, rank_strategy='summation')
    
    res = res.drop(columns=[col for col in res.columns if 'error' in col])
    res = res.drop(columns=['rank', 'u_rank', 'p_rank', 'f_rank'])
    res = res.droplevel(1, axis=1)

    mean_values = res.mean()
    sem_error = res.sem()
    results = pd.concat([mean_values, sem_error], axis=1, keys=['mean', 'sem'])

    return results

### Run experiments
cart_results = model_experiment(df_train, df_test, label, 'synthpop', metrics)
cart_results.to_csv('experiments/results/hepatitis_case_study_adapt/synthpop.csv')

bn_results = model_experiment(df_train, df_test, label, 'datasynthesizer', metrics)
bn_results.to_csv('experiments/results/hepatitis_case_study_adapt/datasynthesizer.csv')

ctgan_results = model_experiment(df_train, df_test, label, 'ctgan', metrics)
ctgan_results.to_csv('experiments/results/hepatitis_case_study_adapt/ctgan.csv')

adsgan_results = model_experiment(df_train, df_test, label, 'adsgan', metrics)
adsgan_results.to_csv('experiments/results/hepatitis_case_study_adapt/adsgan.csv')

dpgan_results = model_experiment(df_train, df_test, label, 'dpgan', metrics)
dpgan_results.to_csv('experiments/results/hepatitis_case_study_adapt/dpgan.csv')

df_dgms = mixed_model_experiment(df_train, df_test, 'synthpop', 'ctgan', cat_atts, num_atts, label, metrics)
df_dgms.to_csv('experiments/results/hepatitis_case_study_adapt/synthpop_ctgan.csv')

df_dgms = mixed_model_experiment(df_train, df_test, 'synthpop', 'adsgan', cat_atts, num_atts, label, metrics)
df_dgms.to_csv('experiments/results/hepatitis_case_study_adapt/synthpop_adsgan.csv')

df_dgms = mixed_model_experiment(df_train, df_test, 'synthpop', 'dpgan', cat_atts, num_atts, label, metrics)
df_dgms.to_csv('experiments/results/hepatitis_case_study_adapt/synthpop_dpgan.csv')

df_dgms = mixed_model_experiment(df_train, df_test, 'datasynthesizer', 'ctgan', cat_atts, num_atts, label, metrics)
df_dgms.to_csv('experiments/results/hepatitis_case_study_adapt/datasynthesizer_ctgan.csv')

df_dgms = mixed_model_experiment(df_train, df_test, 'datasynthesizer', 'adsgan', cat_atts, num_atts, label, metrics)
df_dgms.to_csv('experiments/results/hepatitis_case_study_adapt/datasynthesizer_adsgan.csv')

df_dgms = mixed_model_experiment(df_train, df_test, 'datasynthesizer', 'dpgan', cat_atts, num_atts, label, metrics)
df_dgms.to_csv('experiments/results/hepatitis_case_study_adapt/datasynthesizer_dpgan.csv')

print('Done!')