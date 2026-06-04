# Description: Functions to present the results in readible ways. 
# Author: Anton D. Lautrup
# Date: 17-01-2025

import numpy as np
import pandas as pd

from pandas import DataFrame

from scipy.stats import norm

def create_comparison_table(filepath_A: str, filepath_B: str, name_A: str, name_B: str) -> DataFrame:
    """ Create a comparison table for results from the two filepaths."""

    res_A = pd.read_csv(filepath_A, index_col=0)
    res_B = pd.read_csv(filepath_B, index_col=0)
    
    # check that they hav the same indexes and drop the ones that are not in both
    res_A = res_A[res_A.index.isin(res_B.index)]
    res_B = res_B[res_B.index.isin(res_A.index)]

    # calculate percentage difference
    res_diff = (res_A['mean'] - res_B['mean']) / res_B['mean'] * 100

    # calculate the z-score for the difference
    z_score = (res_A['mean'] - res_B['mean']) / np.sqrt(res_A['sem']**2+res_B['sem']**2)

    # get p-values
    p_values = 2 * (1 - norm.cdf(np.abs(z_score)))
    p_values = pd.Series(p_values, index=res_B.index)

    res = pd.concat([res_A['mean'], res_B['mean'], res_diff, z_score, p_values], axis=1, keys=[name_A, name_B, 'diff %', 'z_score', 'p_value'])
    
    return res

def analyze_overlapping_confidence_intervals(df_meansem, metrics=None, higher_is_better=None):
    """
    Analyze overlapping 95% confidence intervals when input is a mean/sem layout.

    Expected input formats supported:
    - DataFrame with a MultiIndex row index (model, stat) where stat is 'mean' or 'sem',
      and columns are metric names. Example: df.loc[('sp','mean'), 'auroc']
    - If your raw CSV has the first two columns as model and stat, read it with
      `pd.read_csv(..., index_col=[0,1])` before passing it here.

    Returns a dict mapping metric -> {best_model, best_value, overlapping_models}.
    """
    import pandas as pd

    # Normalize input to MultiIndex (model, stat)
    df = df_meansem.copy()

    if not isinstance(df.index, pd.MultiIndex):
        # try to detect and convert if there are two unnamed index-like columns
        # if first two columns are strings like 'sp' and 'mean', user should re-read CSV
        raise ValueError("Input must have a MultiIndex row index (model, stat). Read CSV with index_col=[0,1].")

    # Check expected stat level values
    stat_level = df.index.levels[1]
    if not set(['mean', 'sem']).issubset(set(stat_level)):
        raise ValueError("Second level of the row MultiIndex must include 'mean' and 'sem'.")

    # Split means and sems
    means = df.xs('mean', level=1)
    sems = df.xs('sem', level=1)

    # Determine metrics
    if metrics is None:
        metrics = [c for c in means.columns]

    # Default higher-is-better set (metrics where larger is better)
    if higher_is_better is None:
        higher_is_better = set(['avg_F1_diff', 'avg_F1_diff_hout', 'auroc', 'median_DCR'])

    results = {}

    for metric in metrics:
        if metric not in means.columns:
            continue

        metric_stats = pd.DataFrame({'mean': means[metric], 'sem': sems[metric]})
        metric_stats['lower'] = metric_stats['mean'] - 1.96 * metric_stats['sem']
        metric_stats['upper'] = metric_stats['mean'] + 1.96 * metric_stats['sem']

        # pick best model
        if metric in higher_is_better:
            best_model = metric_stats['mean'].idxmax()
        else:
            best_model = metric_stats['mean'].idxmin()

        best_lower = metric_stats.loc[best_model, 'lower']
        best_upper = metric_stats.loc[best_model, 'upper']

        overlapping = []
        for model in metric_stats.index:
            if model == best_model:
                continue
            lower = metric_stats.loc[model, 'lower']
            upper = metric_stats.loc[model, 'upper']
            if (lower <= best_upper) and (upper >= best_lower):
                overlapping.append(model)

        results[metric] = {
            'best_model': best_model,
            'best_value': metric_stats.loc[best_model, 'mean'],
            'overlapping_models': overlapping
        }

    return results
    