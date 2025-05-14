# Description: Script repeated random partitions with measuring correlation of the disjoint parts
# Author: Anonymous
# Date: 07-04-2025

import os

import sys
sys.path.append('.')

import numpy as np
import pandas as pd

from typing import List
from pandas import DataFrame

from syntheval import SynthEval

from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier

from disjoint_generative_model import DisjointGenerativeModels
from disjoint_generative_model.utils.joining_strategies import UsingJoiningValidator, Concatenating
from disjoint_generative_model.utils.joining_validator import JoiningValidator

def measure_correlation(df: DataFrame, variables1: List[str], variables2: List[str]) -> float:
    """ Measure the correlation of the disjoint parts of the dataset. """
    correlation = df.corr()
    correlation = correlation.loc[variables1, variables2]
    if variables1 == variables2:
        np.fill_diagonal(correlation.values, 0)

    correlation = correlation.values
    frobenius_norm = np.linalg.norm(correlation, ord='fro')
    return frobenius_norm

def _random_correlation(size: int) -> np.ndarray:
    """ Generate a random correlation matrix of a given size. """
    A = 2*(np.random.rand(size, size)-0.5)
    A = (A + A.T) / 2
    np.fill_diagonal(A, 1)
    corr_mat = np.clip(A, -1, 1)
    return corr_mat

def _random_cross_correlation(size: int, strength: float) -> np.ndarray:
    """ Generate a random cross-correlation matrix of a given size. """
    corr_matrix = np.random.rand(size, size)
    corr_matrix = np.abs(strength + corr_matrix * strength)

    mask = np.random.rand(size, size) < 0.5
    corr_matrix[mask] *= -1

    corr_matrix = np.clip(corr_matrix, -1, 1)
    return corr_matrix

def cor2cov(C):
    diag = np.sqrt(np.diag(np.diag(C)))
    gaid = np.linalg.inv(diag)
    return gaid @ C @ gaid

def _generate_dataset(dummy_feats_size: int, dummy_items_size: int, strength: float) -> DataFrame:
    """ Generate a random dataset with correlated features. """

    split0 = _random_correlation(dummy_feats_size//2)
    split1 = _random_correlation(dummy_feats_size//2)

    cross_corr = _random_cross_correlation(dummy_feats_size//2, strength)

    correlation_matrix = np.zeros((dummy_feats_size, dummy_feats_size))
    correlation_matrix[:dummy_feats_size//2, :dummy_feats_size//2] = split0
    correlation_matrix[dummy_feats_size//2:, dummy_feats_size//2:] = split1
    correlation_matrix[:dummy_feats_size//2, dummy_feats_size//2:] = cross_corr
    correlation_matrix[dummy_feats_size//2:, :dummy_feats_size//2] = cross_corr.T

    covariance_matrix = cor2cov(correlation_matrix)

    data = np.random.multivariate_normal(mean=np.zeros(dummy_feats_size), cov=covariance_matrix, size=dummy_items_size)

    columns = [f'f{i}' for i in range(dummy_feats_size)]
    df = pd.DataFrame(data, columns=columns)

    df['f0'] = (df['f0'] <= 0).astype(int)

    return df


def generate_measurement(model_name: str, dummy_feats_size, dummy_items_size, strength: float, results_file, metrics, id):
    """ Generate a measurement of the correlation of the disjoint parts of the dataset. """

    results = {"model": model_name, "corr_str": strength, "id": id}
    
    df = _generate_dataset(dummy_feats_size, dummy_items_size, strength)
    df_train = df.sample(frac=0.66, random_state=42)
    df_test = df.drop(df_train.index)

    partitions = {'split0': list(df_train.columns[:dummy_feats_size//2]), 'split1': list(df_train.columns[dummy_feats_size//2:])}

    SE = SynthEval(df_train, df_test, verbose=False)

    corr_11 = measure_correlation(df_train, partitions['split0'], partitions['split0'])
    corr_22 = measure_correlation(df_train, partitions['split1'], partitions['split1'])

    corr_12 = measure_correlation(df_train, partitions['split0'], partitions['split1'])
    corr_21 = measure_correlation(df_train, partitions['split1'], partitions['split0'])

    results['corr_intra'] = corr_11 + corr_22
    results['corr_inter'] = corr_12 + corr_21

    parameter_grid = {
        'n_estimators': [10, 15, 50, 100], 
        'max_depth': [5, 10, 25, None], 
        'criterion': ['gini', 'entropy'],
        }

    JV = JoiningValidator(classifier_model_base=RandomForestClassifier(), 
                                    model_parameter_grid=parameter_grid,  
                                    calibration_method='sigmoid',
                                    save_proba=False,
                                    verbose=False
                                    )

    dgms = DisjointGenerativeModels(df_train, 
                                    generative_models=2*[model_name], 
                                    prepared_splits=partitions,
                                    joining_strategy=UsingJoiningValidator(JV, behaviour='adaptive'), 
                                    parallel_worker_id = 10*id
                                    )
    
    df_syn_validate = dgms.fit_generate()

    res = SE.evaluate(df_syn_validate, analysis_target_var='f0', **metrics)
    res.index = res['metric']
    results_val = results.copy()
    results_val['joining'] = 'validation'
    results_val.update(res['val'].to_dict())

    dgms.strategy = Concatenating()
    df_syn_concat = dgms.conduct_joining()
    df_syn_concat = dgms.dm.postprocess(df_syn_concat)

    res = SE.evaluate(df_syn_concat, analysis_target_var='f0',**metrics)
    res.index = res['metric']
    results_con = results.copy()
    results_con['joining'] = 'concatenation'
    results_con.update(res['val'].to_dict())

    results_df = pd.DataFrame([results_val, results_con])

    if os.path.exists(results_file):
        results_df.to_csv(results_file, index=False, mode='a', header=False)
    else:
        results_df.to_csv(results_file, index=False)
    pass


def experiment_runner(model, dummy_feats_size, dummy_items_size, results_file, metrics):

    NUM_REPS = 10
    NUM_STEPS = 21

    steps = np.linspace(0, 1, NUM_STEPS)

    for step in steps:
        _ = _ = Parallel(n_jobs=5)(delayed(generate_measurement)(
            model, dummy_feats_size, dummy_items_size, step, results_file, metrics, id) for id in range(NUM_REPS))

if __name__ == "__main__":
    
    model = 'synthpop'

    dummy_feats_size = 10        # make sure it is divisible by 2
    dummy_items_size = 1200

    results_file = 'experiments/results/02_correlation_tradeoff.csv'

    metrics = {
        "pca"       : {},
        "h_dist"    : {},
        "corr_diff" : {"mixed_corr": True},
        "auroc_diff" : {"model": "rf_cls"},
        "cls_acc"   : {"F1_type": "macro"},
        "eps_risk"  : {},
        "mia"       : {"num_eval_iter": 5},
    }

    experiment_runner(model, dummy_feats_size, dummy_items_size, results_file, metrics)


