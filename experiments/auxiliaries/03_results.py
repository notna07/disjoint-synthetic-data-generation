# Description: Script for runing the code of notebook 3 without the process being interrupted.
# Author: Anton D. Lautrup
# Date: 20-12-2024

### Imports
import sys
import warnings
sys.path.append('.')

import pickle

import pandas as pd

from pandas import DataFrame
from typing import Literal, List, Dict

from joblib import Parallel, delayed

from syntheval import SynthEval

from disjoint_generation import DisjointGenerativeModels
from disjoint_generation.utils.joining_strategies import UsingJoiningValidator, Concatenating
from disjoint_generation.utils.generative_model_adapters import generate_synthetic_data

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

def _single_mixed_model_experiment(df_train: DataFrame, gms: List[str], parts: Dict[str, List[str]], joining_strat: str, id) -> DataFrame:
    """ Function to do runs of the mixed model. """

    if joining_strat == 'valid':
        with open('experiments/validator_models/hp_rf_opt.obj', 'rb') as file:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                joining_validator = CompatibilityUnpickler(file).load()

        JS = UsingJoiningValidator(joining_validator, behaviour='adaptive')
    elif joining_strat == 'concat':
        JS = Concatenating()
    dgms = DisjointGenerativeModels(df_train, gms, prepared_splits=parts, joining_strategy=JS, parallel_worker_id=id*10)
    dgms.join_multiplier = 4

    df_dgms = dgms.fit_generate()

    return df_dgms[:len(df_train)]

def mixed_model_experiment(df_train: DataFrame, df_test: DataFrame, model1: str, model2: str, 
                           partitions: Dict[str, List[str]], label: str, joining_strat: Literal['concat', 'valid'],
                           metrics: Dict[str, dict]) -> DataFrame:
    """ Function to do repeated runs of the mixed model. """
    
    SE = SynthEval(df_train, df_test, verbose=False)
    
    gms = [model1, model2]
    dfs_list = Parallel(n_jobs=-1)(delayed(_single_mixed_model_experiment)(df_train, gms, partitions, joining_strat, id=i*10) for i in range(NUM_REPS))
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
if __name__ == '__main__':

    exp_series = 'valid' # 'concat'

    parts = {
    'split0': ['RNA EOT', 'ALT 24', 'Diarrhea', 'BMI', 'Epigastric pain', 'Age', 'Headache', 'Plat', 'Fatigue & generalized bone ache', 
               'AST 1', 'Nausea/Vomting', 'Gender', 'ALT 36', 'Fever', 'RNA Base', 'b_class'], 
    'split1': ['RNA EF', 'ALT 48', 'HGB', 'Jaundice', 'RNA 12', 'ALT 1', 'ALT 12', 'ALT 4', 'RNA 4', 
               'Baseline histological Grading', 'RBC', 'ALT after 24 w', 'WBC']
               }
    
    cart_results = model_experiment(df_train, df_test, label, 'synthpop', metrics)
    cart_results.to_csv('experiments/results/03_hepatitis_case_study/synthpop.csv')

    bn_results = model_experiment(df_train, df_test, label, 'datasynthesizer-dp', metrics)
    bn_results.to_csv('experiments/results/03_hepatitis_case_study/datasynthesizer.csv')

    dpgan_results = model_experiment(df_train, df_test, label, 'dpgan', metrics)
    dpgan_results.to_csv('experiments/results/03_hepatitis_case_study/dpgan.csv')

    tabdiff_results = model_experiment(df_train, df_test, label, 'tabdiff', metrics)
    tabdiff_results.to_csv('experiments/results/03_hepatitis_case_study/tabdiff.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'synthpop', 'synthpop', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/synthpop_synthpop_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'synthpop', 'dpgan', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/synthpop_dpgan_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'synthpop', 'datasynthesizer-dp', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/synthpop_datasynthesizer_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'synthpop', 'tabdiff', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/synthpop_tabdiff_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'datasynthesizer-dp', 'datasynthesizer-dp', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/datasynthesizer_datasynthesizer_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'datasynthesizer-dp', 'synthpop', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/datasynthesizer_synthpop_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'datasynthesizer-dp', 'dpgan', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/datasynthesizer_dpgan_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'datasynthesizer-dp', 'tabdiff', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/datasynthesizer_tabdiff_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'dpgan', 'dpgan', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/dpgan_dpgan_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'dpgan', 'synthpop', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/dpgan_synthpop_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'dpgan', 'datasynthesizer-dp', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/dpgan_datasynthesizer_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'dpgan', 'tabdiff', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/dpgan_tabdiff_{exp_series}.csv')
    
    df_dgms = mixed_model_experiment(df_train, df_test, 'tabdiff', 'tabdiff', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/tabdiff_tabdiff_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'tabdiff', 'synthpop', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/tabdiff_synthpop_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'tabdiff', 'datasynthesizer-dp', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/tabdiff_datasynthesizer_{exp_series}.csv')

    df_dgms = mixed_model_experiment(df_train, df_test, 'tabdiff', 'dpgan', parts, label, exp_series, metrics)
    df_dgms.to_csv(f'experiments/results/03_hepatitis_case_study/tabdiff_dpgan_{exp_series}.csv')

    print('Done!')