# Description: Script for executing a big loop to make, time, privacy, and utility measurements
# Author: Anton D. Lautrup
# Date: 27-11-2024

import os
import time
import pandas as pd

import sys
sys.path.append('.')

from typing import Dict
from pandas import DataFrame

from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier

from disjoint_generative_model import DisjointGenerativeModels

from disjoint_generative_model.utils.joining_validator import JoiningValidator
from disjoint_generative_model.utils.joining_strategies import JoinStrategy, UsingJoiningValidator

from disjoint_generative_model.utils.generative_model_adapters import generate_synthetic_data

from syntheval import SynthEval

STEPS = [2,3,4]

metrics = {
    "corr_diff" : {"mixed_corr": True},
    "auroc_diff" : {"model": "rf_cls"},
    "cls_acc"   : {"F1_type": "macro"},
    "eps_risk"  : {},
    "mia_risk"  : {"num_eval_iter": 5},
}

def part_to_be_repeated(index, data_df, model, dgms_strategy, data_name, multiplier=3):
    """ Generate synthetic data due to subdivisions of the training data """
 
    syn_data = {}
    results_df = pd.DataFrame(columns=['model','data','step','rep_idx','time'])
                                       
    start = time.time()
    df_syn = generate_synthetic_data(data_df, model, index*100)
    end = time.time()

    syn_data[data_name+'0'+'r'+str(index)] = df_syn
    results_df.loc[len(results_df)] = {'model': model, 'data':data_name, 'step': 1, 'rep_idx':index,'time':end-start}
    
    for step in STEPS:
        start = time.time()
        dgms_strategy.max_size = len(data_df)
        dgms = DisjointGenerativeModels(data_df, step*[model], joining_strategy = dgms_strategy, worker_id=index*100)
        dgms.join_multiplier = multiplier # Needed for acceptable amount of result from diabetic_mellitus dataset   
        df_dgms = dgms.fit_generate()
        end = time.time()

        syn_data[data_name+str(step)+'r'+str(index)] = df_dgms
        results_df.loc[len(results_df)] = {'model': model, 'data':data_name, 'step': step, 'rep_idx':index, 'time':end-start}
    
    return (syn_data, results_df)

def data_for_time_privacy_and_utility_figure(
    model: str,
    num_repetitions: int,
    dgms_strategy: JoinStrategy,
    train_data: Dict[str, DataFrame],
    test_data: Dict[str, DataFrame],
    target_vars: Dict[str, str],
    multiplier: int = 3) -> DataFrame:
    """ Generate the data for one model of training times, utility and privacy metrics """

    results_df = None
    for data_name, data_df in train_data.items():
        
        results = Parallel(n_jobs=num_repetitions)(delayed(part_to_be_repeated)(index, data_df, model, dgms_strategy, data_name, multiplier) for index in range(num_repetitions))

        syn_data_list, results_df_list = zip(*results)

        syn_data = {}
        for syn_data_dict in syn_data_list:
            syn_data.update(syn_data_dict)
            
        results_df_temp = pd.concat(results_df_list)

        if results_df is None:
            results_df = results_df_temp
        else: results_df = pd.concat([results_df, results_df_temp], ignore_index=True)

        SE = SynthEval(data_df, test_data[data_name], verbose=False)
        res, _ = SE.benchmark(syn_data, analysis_target_var=target_vars[data_name], **metrics, rank_strategy='summation')

        results_df.loc[results_df['data']==data_name, 'corr_mat_diff'] = res['corr_mat_diff']['value'].tolist()
        results_df.loc[results_df['data']==data_name, 'auroc'] = res['auroc']['value'].tolist()
        results_df.loc[results_df['data']==data_name, 'cls_F1_diff'] = res['cls_F1_diff']['value'].tolist()
        results_df.loc[results_df['data']==data_name, 'cls_F1_diff_hout'] = res['cls_F1_diff_hout']['value'].tolist()
        results_df.loc[results_df['data']==data_name, 'eps_identif_risk'] = res['eps_identif_risk']['value'].tolist()
        results_df.loc[results_df['data']==data_name, 'priv_loss_eps'] = res['priv_loss_eps']['value'].tolist()
        results_df.loc[results_df['data']==data_name, 'mia_cls_risk'] = res['mia_cls_risk']['value'].tolist()

        results_df.loc[results_df['data']==data_name, 'utility'] = res['u_rank'].tolist()
        results_df.loc[results_df['data']==data_name, 'privacy'] = res['p_rank'].tolist()

    for filePath in os.listdir():
        if "SE_" in filePath:
            os.remove(filePath)
            
    return results_df

if __name__ == '__main__':

    models = ['synthpop', 'datasynthesizer', 'ctgan']

    train_data = {
        'bc':pd.read_csv('experiments/datasets/breast_cancer_train.csv'), 
        'cc':pd.read_csv('experiments/datasets/cervical_cancer_train.csv'),
        'de':pd.read_csv('experiments/datasets/derm_train.csv'),
        # 'dm':pd.read_csv('experiments/datasets/diabetic_mellitus_train.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_train.csv'),
        'kd':pd.read_csv('experiments/datasets/kidney_disease_train.csv'),
        'st':pd.read_csv('experiments/datasets/stroke_train.csv'),
        }

    test_data = {
        'bc':pd.read_csv('experiments/datasets/breast_cancer_test.csv'), 
        'cc':pd.read_csv('experiments/datasets/cervical_cancer_test.csv'),
        'de':pd.read_csv('experiments/datasets/derm_test.csv'),
        # 'dm':pd.read_csv('experiments/datasets/diabetic_mellitus_test.csv'),
        'hp': pd.read_csv('experiments/datasets/hepatitis_test.csv'),
        'kd':pd.read_csv('experiments/datasets/kidney_disease_test.csv'),
        'st':pd.read_csv('experiments/datasets/stroke_test.csv'),
        }

    target_vars = {
        'bc':'Status', 
        'cc':'Biopsy',
        'de':'b_class',
        # 'dm':'TYPE',
        'hp':'b_class',
        'kd':'class',
        'st':'stroke',
        }

    for model in models:
        Rf = RandomForestClassifier(n_estimators=100)
        JS = UsingJoiningValidator(JoiningValidator(Rf), patience=5)

        res = data_for_time_privacy_and_utility_figure(model, 5, JS, train_data, test_data, target_vars)

        res.to_csv(f'experiments/results/time_privacy_utility_{model}_diabetes.csv', index=False)