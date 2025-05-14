# Description: Script for holding simple plotting functions
# Author: Anton D. Lautrup
# Date: 05-02-2025

import os
import time
import numpy as np

from typing import Dict, List
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import brier_score_loss, f1_score, log_loss, precision_score, recall_score, roc_auc_score

from .joining_validator import _setup_training_data, JoiningValidator
from sklearn.decomposition import PCA

rcp = {'font.size': 8, 'font.family': 'sans', "mathtext.fontset": "dejavuserif"}
plt.rcParams.update(**rcp)

def _score_validator(validator: JoiningValidator, X_train: DataFrame, y_train: List, X_test: DataFrame, y_test: List):
    y_prob = validator.predict_proba(X_test)
    y_pred = validator.predict(X_test)

    labs = ['Brier loss', 'Log loss', 'AUC', 'Precision', 'Recall', 'F1']
    vals = [[brier_score_loss(y_test, y_prob[:, 1])],
            [log_loss(y_test, y_prob)],
            [roc_auc_score(y_test, y_prob[:, 1])],
            [precision_score(y_test, y_pred)],
            [recall_score(y_test, y_pred)],
            [f1_score(y_test, y_pred)]]

    vals = [[round(val, 3) for val in row] for row in vals]
    return vals, labs

def plot_calibration_curve(validator: JoiningValidator,
                           training_data: Dict[str, DataFrame], 
                           holdout_data: Dict[str, DataFrame],
                           stats: bool = True,
                           save_dir: str = '.', 
                           name: str = None,
                           save_fig: bool = True):
    """ Plot the calibration curve for the validator model """
    
    ### Check if directory exists
    if  (not os.path.exists(save_dir) and save_fig):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(6, 6))
    gs = GridSpec(3, 2)

    ax_cal = fig.add_subplot(gs[0:2, :])
    X_train , y_train = _setup_training_data(training_data, 1)
    disp_train = CalibrationDisplay.from_estimator(validator.model,
                                                X_train,
                                                y_train,
                                                n_bins=10,
                                                name='Training set',
                                                color='tab:blue',
                                                strategy='uniform',
                                                ax = ax_cal)
    X_test, y_test = _setup_training_data(holdout_data, 2)
    disp_test = CalibrationDisplay.from_estimator(validator.model,
                                                X_test,
                                                y_test,
                                                n_bins=10,
                                                name='Holdout set',
                                                color='tab:orange',
                                                strategy='uniform',
                                                ax = ax_cal)

    ax_cal.grid(True, alpha=0.5)
    
    ax_prob_train = fig.add_subplot(gs[2, 0])
    ax_prob_train.hist(disp_train.y_prob, bins=10, range=(0, 1), color='tab:blue')
    ax_prob_train.set_ylabel("Count")
    ax_prob_train.set_xlabel("Mean predicted probability")
    ax_prob_train.grid(axis='y', alpha=0.5)

    ax_prob_test = fig.add_subplot(gs[2, 1], sharey=ax_prob_train)
    ax_prob_test.hist(disp_test.y_prob, bins=10, range=(0, 1), color='tab:orange')
    ax_prob_test.set_xlabel("Mean predicted probability")
    ax_prob_test.grid(axis='y', alpha=0.5)

    if stats:
        tab_dat, tab_lab = _score_validator(validator.model, X_train, y_train, X_test, y_test)

        ax_cal.legend(loc=[0.48, 0.02], fontsize=8)
        table = ax_cal.table(cellText=tab_dat,
                                rowLabels=tab_lab,
                                loc='lower right',
                                cellLoc='right',
                                edges='closed',
                                bbox=[0.89, 0.02, 0.1, 0.3])  # Adjust bbox as needed for positioning
        table.auto_set_font_size(False)
        table.set_fontsize(8)

    if name is None:
        name = f'calibration_curve_{int(time.time())}'

    plt.tight_layout()
    if not save_fig:
        return fig
    else:
        plt.savefig(f'{save_dir}/{name}.png')
        plt.close()
    pass

def plot_samplespace_distribution(validator: JoiningValidator,
                                training_data: Dict[str, DataFrame], 
                                holdout_data: Dict[str, DataFrame],
                                save_dir: str = '.', 
                                name: str = None,
                                save_fig: bool = True):
    """ Plot the calibration curve for the validator model """
    
    ### Check if directory exists
    if  (not os.path.exists(save_dir) and save_fig):
        os.makedirs(save_dir)

    fig, axes = plt.subplots(1,2,figsize=(12, 6), sharex=True, sharey=True)

    X_train , y_train = _setup_training_data(training_data, 1)
    y_prob = validator.model.predict_proba(X_train)[:, 1]

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)

    X_train['PCA1'] = X_train_pca[:, 0]
    X_train['PCA2'] = X_train_pca[:, 1]
    X_train['y_prob'] = y_prob
    X_train['y'] = y_train

    sns.scatterplot(data=X_train, x="PCA1", y="PCA2", hue='y_prob', style='y', ax=axes[0], palette='vlag', hue_norm=(0, 1))
    axes[0].grid(True, alpha=0.5)
    axes[0].set_title('Training set')

    X_test , y_test = _setup_training_data(holdout_data, 1)
    y_prob = validator.model.predict_proba(X_test)[:, 1]

    X_test_pca = pca.transform(X_test)

    X_test['PCA1'] = X_test_pca[:, 0]
    X_test['PCA2'] = X_test_pca[:, 1]
    X_test['y_prob'] = y_prob
    X_test['y'] = y_test

    sns.scatterplot(data=X_test, x="PCA1", y="PCA2", hue='y_prob', style='y', ax=axes[1], palette='vlag', hue_norm=(0, 1))
    axes[1].grid(True, alpha=0.5)
    axes[1].set_title('Holdout set')

    if name is None:
        name = f'probabilities_plot_{int(time.time())}'

    plt.tight_layout()
    if not save_fig:
        return fig
    else:
        plt.savefig(f'{save_dir}/{name}.png')
        plt.close()
    pass

def plot_proba_hist(pred, save_dir='.', name = None):
    """ Plot a histogram of the predicted probabilities """

    ### Check if directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=(6, 3))
    bins = np.linspace(0, 1, 21)
    sns.histplot(pred, kde=True, bins=bins, color='blue', alpha=0.5)
    plt.xlabel('Predicted probability')
    plt.ylabel('Frequency')
    plt.tight_layout()

    if name is None:
        name = f'proba_hist_{int(time.time())}'

    plt.savefig(f'{save_dir}/{name}.png', dpi=300)
    plt.close()
    pass
