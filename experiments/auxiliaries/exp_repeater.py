# Description: Function to be imported which takes a function that generates synthetic data 
#   and repeatedly evaluates it a number of times.
# Author: Anton D. Lautrup
# Date: 13-01-2025

import time
import pandas as pd

from pandas import DataFrame
from typing import List, Dict

from syntheval import SynthEval

from scipy.stats import sem
from joblib import Parallel, delayed

def _generate_function(df_train: DataFrame, id: int) -> DataFrame:
    """ Template function.
    
    Args:
        df_train (DataFrame): The training dataframe.
        id (int): An identifier for the repetition.
    
    Returns:
        DataFrame: The generated synthetic dataframe.
    """
    # Implement the synthetic data generation logic here
    synthetic_data = df_train.copy()  # Placeholder for actual synthetic data generation logic
    return synthetic_data

def repeated_experiment(df_train: DataFrame, df_test: DataFrame, label: str, 
                        generate_function: _generate_function, num_reps: int,
                        metrics: Dict[str, dict]) -> DataFrame:
    """ Function to do repeated runs of an inputted _generate_function. """
    
    SE = SynthEval(df_train, df_test, verbose=False)
    
    dfs_list = Parallel(n_jobs=-1)(delayed(generate_function)(df_train, id=i) for i in range(num_reps))
    dfs = {f"rep_{i}": df_synth for i, df_synth in enumerate(dfs_list)}

    res, _ = SE.benchmark(dfs, analysis_target_var=label,**metrics, rank_strategy='summation')
    
    res = res.drop(columns=[col for col in res.columns if 'error' in col])
    res = res.drop(columns=['rank', 'u_rank', 'p_rank', 'f_rank'])
    res = res.droplevel(1, axis=1)

    mean_values = res.mean()
    sem_error = res.sem()
    results = pd.concat([mean_values, sem_error], axis=1, keys=['mean', 'sem'])

    return results

def repeated_experiment_timed(df_train: DataFrame, df_test: DataFrame, label: str, 
                            generate_function: _generate_function, num_reps: int,
                            metrics: Dict[str, dict]) -> DataFrame:
    """ Function to do repeated runs of an inputted _generate_function not in parallel """
    
    SE = SynthEval(df_train, df_test, verbose=False)
    
    dfs_list = []
    times_list = []
    for i in range(num_reps):
        start = time.time()
        df_synth = generate_function(df_train, id=i)
        end = time.time()
        dfs_list.append(df_synth)
        times_list.append(end-start)
        
    dfs = {f"rep_{i}": df_synth for i, df_synth in enumerate(dfs_list)}

    res, _ = SE.benchmark(dfs, analysis_target_var=label,**metrics, rank_strategy='summation')
    
    res = res.drop(columns=[col for col in res.columns if 'error' in col])
    res = res.drop(columns=['rank', 'u_rank', 'p_rank', 'f_rank'])
    res = res.droplevel(1, axis=1)

    mean_values = res.mean()
    sem_error = res.sem()

    results = pd.concat([mean_values, sem_error], axis=1, keys=['mean', 'sem'])
    results.loc['time'] = [sum(times_list)/num_reps, sem(times_list)]
    return results