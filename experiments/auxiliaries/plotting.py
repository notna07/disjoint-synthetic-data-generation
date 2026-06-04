# Description: Script for making nice plots for the paper
# Author: Anton D. Lautrup
# Date: 09-12-2024

import math
import numpy as np
import pandas as pd

from glob import glob

from pandas import DataFrame
from typing import List, Dict, Literal

import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker


plt.rcParams.update({
    'font.size': 7, 
    'font.family': 'sans-serif', 
    "mathtext.fontset": "dejavuserif"
    })


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
        res = pd.read_csv(f'experiments/results/03_mixed_models_results/{models[0]}_{models[1]}_{dataset}.csv')

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


def figure3_metrics_vs_partitions(results, models, metrics, return_flag=True):

    fig, axes = plt.subplots(len(models), len(metrics), figsize=(11, 6), sharex=True)

    for j, metric in enumerate(metrics.keys()):
        for i, model in enumerate(models):
            axes[i, j].minorticks_on()
            axes[i, j].xaxis.set_tick_params(which='minor', bottom=False)
            axes[i, j].yaxis.grid(True, which='major', linestyle='--', linewidth='0.5', alpha=0.3)
            sns.pointplot(data=results[results['model']==model], x="partitions", y=metric, hue="data", ax=axes[i, j], palette='mako', capsize=.1, linewidth=1.5, errorbar='sd')
            axes[i, j].get_legend().remove()

            if i == 0: axes[i, j].set_title(metrics[metric], fontsize=10, loc='left')

            if j == 0: 
                axes[i, j].set_ylabel(model, fontsize=8)
            else:
                axes[i, j].set_ylabel('')

            if i == len(models)-1:
                axes[i, j].set_xlabel('# parts', fontsize=8)
                ticks = axes[i, j].get_xticks()
                axes[i, j].set_xticks(ticks, labels=['base', '2', '3', '4'])

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=7, fontsize=8, bbox_to_anchor=(0.52, 1.07), bbox_transform=fig.transFigure)
    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.6)

    if return_flag == True:
        return fig
    else:
        plt.savefig(f"experiments/figures/figure3_metrics_vs_partitions.pdf", bbox_inches='tight')
        pass

def figure4_joining_method_metric_vs_partitions(results, model, dataset, metrics, return_flag=True):

    results = results[results['model']==model]
    results = results[results['data']==dataset]
    
    fig, axes = plt.subplots(int(np.ceil(len(metrics)/3)), 3, figsize=(7, 4), sharex=True)

    axes = axes.flatten()
    for i, metric in enumerate(metrics.keys()):
        axes[i].minorticks_on()
        axes[i].xaxis.set_tick_params(which='minor', bottom=False)
        axes[i].yaxis.grid(True, which='major', linestyle='--', linewidth='0.5', alpha=0.3)
        sns.pointplot(data=results, x="partitions", y=metric, hue='experiment', ax=axes[i], palette='mako', capsize=.1, linewidth=1.5, errorbar='sd')
        axes[i].get_legend().remove()
        axes[i].set_title(metrics[metric], fontsize=10, loc='left')

        if i % 3 == 0:
            axes[i].set_ylabel("metric value", fontsize=8)
        else:
            axes[i].set_ylabel("")

        if i >= len(metrics)-3:
            axes[i].set_xlabel('# parts', fontsize=8)
            ticks = axes[i].get_xticks()
            axes[i].set_xticks(ticks, labels=['base', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    axes[4].legend(loc='upper right')
    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.8)

    if return_flag == True:
        return fig
    else:
        plt.savefig(f"experiments/figures/figure4_concat_vs_validate.pdf", bbox_inches='tight')

def figure5a_time_vs_partitions(results, models, return_flag=False):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharex=True)

    for j, model in enumerate(models):
        axes[j].minorticks_on()
        axes[j].xaxis.set_tick_params(which='minor', bottom=False)
        axes[j].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        axes[j].yaxis.grid(True, which='major', linestyle='--', linewidth='0.5', alpha=0.3)
        sns.pointplot(data=results[results['model']==model], x="partitions", y='time', hue="data", ax=axes[j], palette='mako', capsize=.1, linewidth=1.5, errorbar='sd')
        axes[j].get_legend().remove()
        axes[j].annotate(model, xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)

        axes[j].set_ylabel("Time (s)", fontsize=8)

        axes[j].set_xlabel('# parts', fontsize=8)
    ticks = axes[j].get_xticks()
    axes[j].set_xticks(ticks, labels=['base', '2', '3', '4'])
    axes[1].set_ylabel("")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc='upper right', fontsize=8)
    plt.tight_layout(pad=0.0, w_pad=0.8)
    if return_flag:
        return fig, axes
    else:
        plt.savefig(f"experiments/figures/figure5a_time_vs_partitions.pdf", bbox_inches='tight')

def figure5b_time_vs_partitions(results, return_flag=False):
    fig, ax = plt.subplots(1,1,figsize=(3, 3))

    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    ax.yaxis.grid(True, which='major', linestyle='--', linewidth='0.5', alpha=0.3)

    sns.pointplot(data=results, x="partitions", y='time', hue='experiment', ax=ax, palette='mako', capsize=.1, linewidth=1.5, errorbar='sd')
    ax.legend(loc='center right', fontsize=8)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.set_ylabel("Time (s)", fontsize=8)
    ax.set_xlabel('# parts', fontsize=8)
    ax.annotate('datasynthesizer', xy=(0.5, 0.9), xycoords='axes fraction', ha='center', fontsize=10)

    ticks = ax.get_xticks()
    ax.set_xticks(ticks, labels=['base', '2', '3', '4', '5', '6', '7', '8', '9', '10'])

    plt.tight_layout(pad=0.0)
    if return_flag:
        return fig, ax
    plt.savefig(f"experiments/figures/figure5b_time_vs_partitions.pdf", bbox_inches='tight')

def figure6_inter_correlation_concat_tradeoff(results, model, metrics, return_flag=True, mode:Literal['inter', 'intra', 'ratio']='ratio'):
    results = results[results['model']==model]
    
    results['corr_ratio'] = results['corr_inter'] / (results['corr_intra'])

    match mode:
        case 'inter':
            plot_var = 'corr_inter'
            print("Plotting metric performance vs. inter-partition correlation")
        case 'intra':
            plot_var = 'corr_intra'
            print("Plotting metric performance vs. intra-partition correlation")
        case 'ratio':
            plot_var = 'corr_ratio'
            print("Plotting metric performance vs. inter/intra correlation ratio")

    fig, axes = plt.subplots(int(np.ceil(len(metrics)/3)), 3, figsize=(7, 4), sharex=True)

    axes = axes.flatten()
    for i, metric in enumerate(metrics.keys()):
        axes[i].minorticks_on()
        axes[i].yaxis.grid(True, which='major', linestyle='--', linewidth='0.5', alpha=0.3)
        
        sns.lineplot(data=results, x=plot_var, y=metric, hue='joining', ax=axes[i], palette='mako', linewidth=1.5, errorbar=None, alpha=0.2)

        palette = sns.color_palette("mako", n_colors=2)
        for j,join_type in enumerate(results['joining'].unique()):
            subset = results[results['joining'] == join_type]
            subset = subset.sort_values(by=plot_var)
            moving_avg = subset[metric].rolling(window=10, min_periods=1).mean()
            axes[i].plot(subset[plot_var], moving_avg, label=f"{join_type} (MA)", linestyle='--', linewidth=1.5, color=palette[j])

        axes[i].get_legend().remove()
        axes[i].set_title(metrics[metric], fontsize=10, loc='left')

        if i % 3 == 0:
            axes[i].set_ylabel("metric value", fontsize=8)
        else:
            axes[i].set_ylabel("")

        if i >= len(metrics)-3:
            if mode == 'ratio':
                axes[i].set_xlabel(r"corr. ratio. ($C_{p,p'} / C_{p,p}$)", fontsize=9)
            elif mode == 'inter':
                axes[i].set_xlabel(r"corr. ($C_{p,p'}$)", fontsize=9)
            elif mode == 'intra':
                axes[i].set_xlabel(r"corr. ($C_{p,p}$)", fontsize=9)

    custom_lines = [Line2D([0], [0], color=palette[0], lw=3),
                    Line2D([0], [0], color=palette[1], lw=3),]

    axes[3].legend(custom_lines, ['valid', 'concat'], loc='lower left', fontsize=8)
    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.8)

    if return_flag == True:
        return fig
    else:
        plt.savefig(f"experiments/figures/figure6_correlation_tradeoff_{mode}.pdf", bbox_inches='tight')

def figure7_mixed_model_results_pointplot(results, datasets: List[str], models, dict_metric_names: Dict[str, str], mixed_model_label: str = "sp-dpgan DGM" ):

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = sns.color_palette("mako", n_colors=len(datasets))
    sns.pointplot(
        data=results[results['model']=='dgms'], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
        markers="*", markersize=8, linestyle="none", ax=ax, legend=False
    )
    sns.stripplot(
        data=results[results['model']=='dgms'], x="metric", y="value", hue="dataset",
        dodge=True, palette=colors, jitter=False, alpha=.2, legend=False, marker="*", size=8, ax=ax
    )
    sns.pointplot(
        data=results[results['model']==models[1]], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
        markers="^", markersize=6, linestyle="none", ax=ax, legend=False
    )
    sns.pointplot(
        data=results[results['model']==models[0]], x="metric", y="value", hue="dataset",
        dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
        markers="s", markersize=6, linestyle="none", ax=ax
    )

    # setting the custom legend with only the model names and the marker types
    custom_lines = [
        Line2D([0], [0], color=colors[0], marker='*', linestyle='None', linewidth=2, markersize=8, label=mixed_model_label),
        Line2D([0], [0], color=colors[0], marker='^', linestyle='None', markersize=8, label=models[1]),
        Line2D([0], [0], color=colors[0], marker='s', linestyle='None', markersize=8, label=models[0])
    ]

    leg = plt.legend(handles=custom_lines, title='Models', fontsize=8, title_fontsize='9')
    ax.add_artist(leg)

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.12), bbox_transform=ax.transAxes, fontsize=8)

    # use the metric names to the from the dictionary
    metric_names = [dict_metric_names[metric] for metric in dict_metric_names.keys()]
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_xticks(range(len(metric_names)))

    # renaming the axes
    ax.set_ylabel("metric value", fontsize=8)
    ax.set_xlabel("", fontsize=8)

    # plt.savefig('experiments/figures/figure7_mixed_model_other_datasets.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'experiments/figures/figure7_{models[0]}_{models[1]}_other_datasets.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def figure8_baseline_model_results_pointplot(results, datasets, models, dict_metric_names, dict_plot_params):

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = sns.color_palette("mako", n_colors=len(datasets))

    for model, params in dict_plot_params.items():
        lines_before = len(ax.lines)

        if model == 'tvae':
            sns.pointplot(
                data=results[results['model'] == model], x="metric", y="value", hue="dataset",
                dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
                markers=params['marker'], markersize=params['markersize'], linestyle="none", ax=ax, markeredgecolor='white', markeredgewidth=0.5
            )
        elif model == 'dgms':
            sns.pointplot(
                data=results[results['model'] == model], x="metric", y="value", hue="dataset",
                dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
                markers=params['marker'], markersize=params['markersize'], linestyle="none", ax=ax, markeredgecolor='red', markeredgewidth=0.5, legend=False, zorder=10
            )
        else:
            sns.pointplot(
                data=results[results['model'] == model], x="metric", y="value", hue="dataset",
                dodge=.8 - .8 / len(datasets), palette=colors, errorbar="se", err_kws={'linewidth': 1}, capsize=0.05,
                markers=params['marker'], markersize=params['markersize'], linestyle="none", ax=ax, markeredgecolor='white', markeredgewidth=0.5, legend=False
            )

    # setting the custom legend with only the model names and the marker types
    custom_lines = [
        Line2D([0], [0], color=colors[0], marker=params['marker'], linestyle='None', linewidth=2, markersize=params['markersize'], label=params['name'])
        for model, params in dict_plot_params.items()
    ]

    leg = plt.legend(handles=custom_lines, title='Models', fontsize=8, title_fontsize='10')
    ax.add_artist(leg)

    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

    ax.legend(loc='upper center', ncol=7, bbox_to_anchor=(0.5, 1.12), bbox_transform=ax.transAxes, fontsize=8)

    # use the metric names to the from the dictionary
    metric_names = [dict_metric_names[metric] for metric in dict_metric_names.keys()]
    ax.set_xticklabels(metric_names, fontsize=8)
    ax.set_xticks(range(len(metric_names)))

    # renaming the axes
    ax.set_ylabel("metric value", fontsize=8)
    ax.set_xlabel("", fontsize=8)

    plt.savefig('experiments/figures/figure8_other_datasets_baseline.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    pass

def figure10_metrics_vs_noise_aggregate(results: pd.DataFrame, partition_results: pd.DataFrame, exp_names: Dict[str, str], models: List[str], 
                                 metrics: Dict[str, str], return_flag=True, save_name="figure10_metrics_vs_noise_aggregate"):
    """This figure has the same layout as the partitions figure, but shows the averages 
    across experiment series for each degree and overlays partition reference levels.
    """
    
    fig, axes = plt.subplots(len(models), len(metrics), figsize=(11, 6), sharex=True)
    ref_partitions = [2, 3, 4]
    ref_params = {2: '-', 3:'--', 4: ':'}

    for j, metric in enumerate(metrics.keys()):
        for i, model in enumerate(models):
            axes[i, j].minorticks_on()
            axes[i, j].xaxis.set_tick_params(which='minor', bottom=False)
            axes[i, j].yaxis.grid(True, which='major', linestyle='--', linewidth='0.5', alpha=0.3)

            df_subset = results[(results['model'] == model)].copy()
            df_subset['degree'] = pd.to_numeric(df_subset['degree'], errors='coerce')
            df_subset = df_subset.dropna(subset=['degree'])

            # Average over experiment series while preserving variability over remaining rows.
            df_subset = df_subset.groupby(['exp_name', 'degree', 'data'])[metric].agg(['mean', 'std']).reset_index()
            sns.lineplot(data=df_subset, x="degree", y="mean", hue="exp_name", ax=axes[i, j], palette='mako', linewidth=1.5, errorbar=None)

            # Plot three horizontal reference lines from partitioning results (partitions 2, 3, 4).
            df_partition_subset = partition_results[(partition_results['model'] == model)].copy()
            df_partition_subset['partitions'] = pd.to_numeric(df_partition_subset['partitions'], errors='coerce')
            df_partition_subset = df_partition_subset[df_partition_subset['partitions'].isin(ref_partitions)]
            ref_levels = df_partition_subset.groupby('partitions')[metric].mean()

            for p in ref_partitions:
                if p in ref_levels.index:
                    axes[i, j].axhline(
                        y=ref_levels.loc[p],
                        color='k',
                        linestyle=ref_params[p],
                        linewidth=1.2,
                        alpha=0.7
                    )
            axes[i, j].get_legend().remove()

            if i == 0:
                axes[i, j].set_title(metrics[metric], fontsize=10, loc='left')

            if j == 0: 
                axes[i, j].set_ylabel(model, fontsize=8)
            else:
                axes[i, j].set_ylabel('')

            if i == len(models)-1:
                axes[i, j].set_xlabel('degree', fontsize=8)

            if metric == 'pca_eigval_diff':
                axes[i, j].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
            elif metric == 'auroc' or metric == 'avg_F1_diff_hout':
                axes[i, j].yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Replace the legend labels with the experiment names
    labels = [exp_names.get(label, label) for label in labels]
    fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.35, 1.07), bbox_transform=fig.transFigure)
    
    # legend for partition reference lines
    partition_handles = [
        plt.Line2D([0], [0], color='k', linestyle=ref_params[p], linewidth=1.2) for p in ref_partitions
    ]
    partition_labels = [f'{p} parts' for p in ref_partitions]
    fig.legend(partition_handles, partition_labels, loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.7, 1.07), bbox_transform=fig.transFigure)
    plt.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.6)
    
    if return_flag == True:
        return fig
    else:
        plt.savefig(f"experiments/figures/{save_name}.pdf", bbox_inches='tight')
        pass

def figure11_validator_pointplot(datapath: str, metrics: Dict[str,str]) -> None:
    """ Function to plot a pointplot with mean results of the different validators for each metric."""

    df = pd.read_csv(datapath)

    validators = df['validator'].unique()
    df = df[['data', 'validator'] + list(metrics.keys())]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(6, 7.5), sharex=True)

    colors = sns.color_palette("mako", n_colors=len(validators))    

    for i, metric in enumerate(metrics.keys()):
        sns.stripplot(
            data=df, x="data", y=metric, hue="validator",
            alpha=0.2, legend=False, palette=colors, dodge=True, ax=axes[i]
        )
        sns.pointplot(
            data=df, x="data", y=metric, hue="validator",
            linestyle="none", dodge=.8 - .8 / len(validators), err_kws={'linewidth':1.2}, capsize=0.1, errorbar='sd',
            palette=colors, markersize = 8, markers="_", ax=axes[i]
        )
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        axes[i].set_title(metrics[metric], fontsize=10, loc='left')
        axes[i].set_ylabel("")
        axes[i].set_xlabel("")
        axes[i].get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=8, fontsize=8, bbox_to_anchor=(0.54, -0.05), bbox_transform=fig.transFigure, columnspacing=0.6, handletextpad=0.2)

    plt.tight_layout(pad=0.0, h_pad=0.6, w_pad=0.0)
    plt.savefig("experiments/figures/figure11_validator_pointplot.pdf", bbox_inches='tight')
    pass

def figure14_threshold_continuum_plot(datapath: str, series: str, dataset: str, valid_models: List[str], metrics: Dict[str,str]) -> None:
    """ This plot shows how the performance of different metrics vary 
    with using different validators at different threshold. """

    df = pd.read_csv(datapath)
    df = df[df['experiment'] == series]

    df = df[df['validator'].isin(valid_models)]
    df = df[df['data'] == dataset]
    df = df[['validator', 'threshold'] + list(metrics.keys())]

    # Create the pointplot
    fig, axes = plt.subplots(int(np.ceil(len(metrics)/3)), 3, figsize=(10, 3*int((np.ceil(len(metrics)/3)))), sharex=True)

    colors = sns.color_palette("mako", n_colors=len(valid_models))

    axes = axes.flatten()
    for i, metric in enumerate(metrics.keys()):
        sns.lineplot(
            data=df, x="threshold", y=metric, hue="validator",
            palette=colors, ax=axes[i], errorbar='se'
        )
        axes[i].grid(axis='y', linestyle='--', alpha=0.3)
        # axes[i].set_ylabel(metrics[metric])
        axes[i].set_ylabel("")
        axes[i].set_title(metrics[metric], fontsize=10, loc='left')

        if i == 0:
            axes[i].set_ylabel("metric value", fontsize=9)
            axes[i].legend(title="Validator", ncols=1, borderpad=0.2, labelspacing=0.1, columnspacing=0.5)
        elif i == 3:
            axes[i].set_ylabel("metric value", fontsize=9)
            axes[i].get_legend().remove()
        else:
            axes[i].get_legend().remove()


    if len(axes) > len(metrics):
        for j in range(len(metrics), len(axes)):
            fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.savefig("experiments/figures/figure14_threshold_continuum_plot.pdf", bbox_inches='tight')
    pass

        
    




