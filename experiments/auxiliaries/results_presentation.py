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
    