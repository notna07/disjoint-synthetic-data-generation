# Description: Script for holding simple plotting functions
# Date: 05-02-2025
# Author : Anton D. Lautrup

import os
import time
import numpy as np
import pandas as pd

from typing import Dict, List
from pandas import DataFrame

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.calibration import CalibrationDisplay

from .joining_validator import _setup_training_data, JoiningValidator

rcp = {'font.size': 8, 'font.family': 'sans', "mathtext.fontset": "dejavuserif"}
plt.rcParams.update(**rcp)


def plot_calibration_curve(validator: JoiningValidator,
                           training_data: Dict[str, DataFrame], 
                           holdout_data: Dict[str, DataFrame],
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
    X_test, y_test = _setup_training_data(holdout_data, 1)
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

    if name is None:
        name = f'calibration_curve_{int(time.time())}'

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
