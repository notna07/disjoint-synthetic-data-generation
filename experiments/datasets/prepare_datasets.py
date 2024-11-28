# Description: Setup for small adjustments for the datasets
# Author: Anton D. Lautrup
# Date: 28-11-2024

import pandas as pd

datasets = ['breast_cancer', 'cervical_cancer', 'derm', 'diabetic_mellitus', 'hepatitis', 'kidney_disease', 'stroke']

def load_and_join(df_name: str) -> pd.DataFrame:
    df_train = pd.read_csv('experiments/datasets/' + df_name + '_train.csv')
    df_test = pd.read_csv('experiments/datasets/' + df_name + '_test.csv')

    df = pd.concat([df_train, df_test], ignore_index=True)
    return df

def split_and_save(df: pd.DataFrame, frac: float, df_name: str) -> None:
    df_train = df.sample(frac=frac, random_state=42)
    df_test = df.drop(df_train.index)

    df_train.to_csv('experiments/datasets/' + df_name + '_train.csv', index=False)
    df_test.to_csv('experiments/datasets/' + df_name + '_test.csv', index=False)

for dataset in datasets:
    df = load_and_join(dataset)    
    split_and_save(df, 0.8, dataset)


# derm and hepatitis need to have binarized output
# def binarize_multilevel_outcome_variable(df: pd.DataFrame, outcome_col: str, outcome_new_name: str) -> pd.DataFrame:
#     df[outcome_col] = df[outcome_col].apply(lambda x: 1 if x != 0 else 0)
#     df = df.rename(columns={outcome_col: outcome_new_name})
#     return df

# def binarize_outcome_dataset(df_name: str, outcome_col: str, outcome_new_name: str) -> None:
#     df = pd.read_csv('experiments/datasets/' + df_name + '_train.csv')
#     df = binarize_multilevel_outcome_variable(df, outcome_col, outcome_new_name)
#     df.to_csv('experiments/datasets/' + df_name + '_train.csv', index=False)

#     df = pd.read_csv('experiments/datasets/' + df_name + '_test.csv')
#     df = binarize_multilevel_outcome_variable(df, outcome_col, outcome_new_name)
#     df.to_csv('experiments/datasets/' + df_name + '_test.csv', index=False)
#     pass

# binarize_outcome_dataset('derm', 'class', 'b_class')
# binarize_outcome_dataset('hepatitis', 'Baselinehistological staging', 'b_class')