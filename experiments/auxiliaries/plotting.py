# Description: Script for making nice plots for the paper
# Author: Anton D. Lautrup
# Date: 09-12-2024

import pandas as pd

from glob import glob

from pandas import DataFrame
from typing import List, Dict

import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

rcp = {'font.size': 9, 'font.family': 'sans', "mathtext.fontset": "dejavuserif"}
plt.rcParams.update(**rcp)

def _format_correctly(res_df: pd.DataFrame, dataset: str, metric: str):
    df = res_df[["dataset",metric]].copy()
    df.rename(columns={"dataset": "model"}, inplace=True)
    df.rename(columns={metric: "value"}, inplace=True)
    df["dataset"] = dataset
    df["metric"] = metric
    return df[["dataset", "model", "metric", "value"]]

def _construct_dataframe(datasets: List[str], models, metrics: List):
    df_results = None
    for dataset in datasets:
        res = pd.read_csv(f'experiments/results/mixed_model_results/{models[0]}_{models[1]}_{dataset}.csv')

        res_dgms = res[res['dataset'] == 'dgms'].reset_index(drop=True)
        res_dpgan = res[res['dataset'] == models[1]].reset_index(drop=True)
        res_synthpop = res[res['dataset'] == models[0]].reset_index(drop=True)

        for metric in metrics:
            dgms_df = _format_correctly(res_dgms, dataset, metric)
            dpgan_df = _format_correctly(res_dpgan, dataset, metric)
            synthpop_df = _format_correctly(res_synthpop, dataset, metric)

            if df_results is None:
                df_results = pd.concat([dgms_df, dpgan_df, synthpop_df])
            else:
                df_results = pd.concat([df_results, dgms_df, dpgan_df, synthpop_df])
    return df_results

def make_relative_derviation_histogram(datasets: List[str], models):

    dict_data_names = {
        # "corr_mat_diff" : "Correlation matrix difference",
        "auroc" : "AUROC difference",
        "cls_F1_diff" : "F1 score difference",
        "cls_F1_diff_hout" : "F1 score difference (holdout)",
        "eps_identif_risk" : "Epsilon identifiability risk",
        "priv_loss_eps" : "Privacy loss (in eps. risk)",
        "mia_recall": "MIA recall"
    }

    df_results = _construct_dataframe(datasets, models, metrics=dict_data_names.keys())
    fig, ax = plt.subplots(figsize=(12, 4))
    
    colors = sns.color_palette("rocket", n_colors=len(datasets))

    sns.pointplot(
        data=df_results[df_results['model']=='dgms'], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", errwidth=1, capsize=0.05,
        markers="*", markersize=8, linestyle="none", ax=ax
    )
    sns.stripplot(
        data=df_results[df_results['model']=='dgms'], x="metric", y="value", hue="dataset",
        dodge=True, palette=colors, jitter=False, alpha=.2, legend=False, marker="*", size=8, ax=ax
    )
    sns.pointplot(
        data=df_results[df_results['model']==models[1]], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", errwidth=1, capsize=0.05,
        markers="^", markersize=6, linestyle="none", ax=ax
    )
    sns.pointplot(
        data=df_results[df_results['model']==models[0]], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", errwidth=1, capsize=0.05,
        markers="s", markersize=6, linestyle="none", ax=ax
    )
    
    # remove legend
    ax.get_legend().remove()

    # setting the custom legend with only the model names and the marker types

    custom_lines = [
        Line2D([0], [0], color=colors[0], marker='*', linestyle='None', linewidth=2, markersize=8, label='Mixed Model'),
        Line2D([0], [0], color=colors[0], marker='^', linestyle='None', markersize=8, label='DP-GAN'),
        Line2D([0], [0], color=colors[0], marker='s', linestyle='None', markersize=8, label='DataSynthesizer')
    ]

    ax.legend(handles=custom_lines, title='Models')

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    # renaming the axes
    ax.set(xlabel="metric", ylabel="metric value")
    # ax.set_title("Results for different metrics on datasets using mixed model, DP-GAN and Synthpop")
    
    # visualizing illustration
    plt.savefig('experiments/results/figures/metrics_results_for_mixed_model.png', dpi=300, bbox_inches='tight')
    plt.savefig('experiments/results/figures/metrics_results_for_mixed_model.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    pass

if __name__ == '__main__':
    datasets = ['al', 'bc', 'cc', 'hd', 'kd', 'st']
    make_relative_derviation_histogram(datasets, models=['datasynthesizer', 'dpgan'])

        

        
    




