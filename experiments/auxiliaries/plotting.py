# Description: Script for making nice plots for the paper
# Author: Anonymous
# Date: 09-12-2024

import math
import pandas as pd

from glob import glob

from pandas import DataFrame
from typing import List, Dict

import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

rcp = {'font.size': 10, 'font.family': 'sans', "mathtext.fontset": "dejavuserif"}
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
        res = pd.read_csv(f'experiments/results/other_datasets_adapt/{models[0]}_{models[1]}_{dataset}.csv')

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

def histograms_of_attributes(real_df, syn_df, atts, discrete=False, ncols=4, save_name=None):

    df_combined = pd.concat([real_df, syn_df], axis=0)
    df_combined['data'] = ['Real']*len(real_df) + ['Synthetic']*len(syn_df)

    fig, axes = plt.subplots(math.ceil(len(atts)/ncols), ncols, figsize=(10, 2*math.ceil(len(atts)/ncols)))
    axes = axes.flatten()

    for i, att in enumerate(atts):
        if discrete: sns.histplot(data=df_combined, x=att, hue='data', multiple="dodge", alpha=0.5, shrink=.8, ax=axes[i])
        else: sns.histplot(data=df_combined, x=att, hue='data', bins = 10, common_norm=False, alpha=0.5, shrink=.8, ax=axes[i])
        axes[i].set_ylabel('')
        axes[i].set_xlabel('')
        axes[i].set_title(f"Variable '{att}'", fontsize=8)

        axes[i].get_legend().remove()
        axes[i].grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    custom_boxes = [
        plt.Line2D([0], [0], color=sns.color_palette()[0], alpha=0.5, lw=6),
        plt.Line2D([0], [0], color=sns.color_palette()[1], alpha=0.5, lw=6)
    ]
    fig.legend(custom_boxes, ['Real', 'Synthetic'], loc='upper center', ncol=2, fontsize=9, bbox_to_anchor=(0.5, 1.04), bbox_transform=fig.transFigure)

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(f'{save_name}.pdf', dpi=300, bbox_inches='tight')
        plt.close()
    pass

def make_relative_derviation_histogram(datasets: List[str], models):

    dict_data_names = {
        # "corr_mat_diff" : "Correlation matrix difference",
        "auroc" : "AUROC difference",
        "avg_F1_diff" : "F1 score difference",
        "avg_F1_diff_hout" : "F1 score difference (holdout)",
        "eps_identif_risk" : "Epsilon identifiability risk",
        "priv_loss_eps" : "Privacy loss (in eps. risk)",
        "mia_recall": "MIA recall"
    }

    df_results = _construct_dataframe(datasets, models, metrics=dict_data_names.keys()).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    
    colors = sns.color_palette("mako", n_colors=len(datasets))

    sns.pointplot(
        data=df_results[df_results['model']=='dgms'], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
        markers="*", markersize=8, linestyle="none", ax=ax, legend=False
    )
    sns.stripplot(
        data=df_results[df_results['model']=='dgms'], x="metric", y="value", hue="dataset",
        dodge=True, palette=colors, jitter=False, alpha=.2, legend=False, marker="*", size=8, ax=ax
    )
    sns.pointplot(
        data=df_results[df_results['model']==models[1]], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
        markers="^", markersize=6, linestyle="none", ax=ax, legend=False
    )
    sns.pointplot(
        data=df_results[df_results['model']==models[0]], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
        markers="s", markersize=6, linestyle="none", ax=ax
    )

    

    # setting the custom legend with only the model names and the marker types

    custom_lines = [
        Line2D([0], [0], color=colors[0], marker='*', linestyle='None', linewidth=2, markersize=8, label='sp-dpgan DGM'),
        Line2D([0], [0], color=colors[0], marker='^', linestyle='None', markersize=8, label='dpgan'),
        Line2D([0], [0], color=colors[0], marker='s', linestyle='None', markersize=8, label='synthpop')
    ]

    leg = plt.legend(handles=custom_lines, title='Models')
    ax.add_artist(leg)

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.12), bbox_transform=ax.transAxes)

    # renaming the axes
    ax.set(ylabel="metric value", xlabel="")
    # ax.set_title("Results for different metrics on datasets using mixed model, DP-GAN and Synthpop")
    
    # visualizing illustration
    plt.savefig('experiments/figures/figure6_mixed_model_other_datasets.png', dpi=300, bbox_inches='tight')
    plt.savefig('experiments/figures/figure6_mixed_model_other_datasets.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    pass

if __name__ == '__main__':
    datasets = ['al', 'bc', 'cc', 'hd', 'kd', 'st']
    make_relative_derviation_histogram(datasets, models=['datasynthesizer', 'dpgan'])

        

        
    




