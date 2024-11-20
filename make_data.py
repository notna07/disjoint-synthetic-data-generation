# Description: Script for generating synthetic dataset using the disjoint generative model
# Author: Anton D. Lautrup
# Date: 20-11-2024

import os
import time
import argparse

from rich import print
from rich.prompt import Confirm

import pandas as pd

from typing import List

from syntheval import SynthEval
from disjoint_generative_model.disjoint_model import DisjointGenerativeModels
from disjoint_generative_model.utils.joining_strategies import Concatenating, RandomJoining, UsingJoiningValidator
from disjoint_generative_model.utils.joining_validator import JoiningValidator

if __name__ == '__main__':
    """ Setup for future CLI """
    parser = argparse.ArgumentParser(description='Generate synthetic dataset using the disjoint generative model')
    parser.add_argument('-tr', '--training_data', type=str, help='The (path to) training data file')
    parser.add_argument('-lb', '--label_col', type=str, help='The label column name')
    parser.add_argument('-gms','--generative_models', nargs='+', help='The generative models to use')
    parser.add_argument('-out', '--output_file', type=str, help='The output file name', default='synthetic_data')
    parser.add_argument('-js', '--joining_strategy', type=str, help='The joining strategy to use', default='concat')
    parser.add_argument('-ev','--evaluate_flag', type=bool, help='Run evaluation?', default=True)

    args = parser.parse_args()

    print("[bold]Disjoint Generative Model Generate Script[/bold]")
    print("=============================================\n")
    time.sleep(2)

    print("Selected options:")
    print(f"Training data: {args.training_data}")
    print(f"Label column: {args.label_col}")
    print(f"Generative models: {args.generative_models}")
    print(f"Joining strategy: {args.joining_strategy}")
    print(f"Output file: {args.output_file}")
    print(f"Evaluation flag: {args.evaluate_flag}\n")
    time.sleep(1)

    prompt_ready_ask = Confirm.ask("Ready to begin? (Y/n)")

    if not prompt_ready_ask:
        print("Exiting..."); time.sleep(5)
        exit()
    # else: os.system('clear')

    print("\nLoading data...");time.sleep(1)    # Load the training data
    training_data = pd.read_csv(args.training_data + '_train.csv').dropna()

    print("Instantiate joining strategy...");time.sleep(1)    # Setup the joining strategy
    if args.joining_strategy == 'concat':
        joining_strategy = Concatenating()
    elif args.joining_strategy == 'random':
        joining_strategy = RandomJoining()
    elif args.joining_strategy == 'validate':
        from sklearn.ensemble import RandomForestClassifier
        Rf = RandomForestClassifier(n_estimators=100)
        joining_strategy = UsingJoiningValidator(JoiningValidator(Rf))
    else:
        raise ValueError(f"Joining strategy {args.joining_strategy} not supported!")

    # Setup the disjoint generative model
    print("Setting up disjoint generative model...");time.sleep(1)    # Setup the disjoint generative model
    disjoint_model = DisjointGenerativeModels(training_data, args.generative_models, joining_strategy=joining_strategy)

    print("Generating synthetic data...");time.sleep(1)    # Generate synthetic data
    synthetic_data = disjoint_model.fit_generate()

    print("Task finished! Saving..."); time.sleep(2)
    synthetic_data.to_csv(args.output_file + '.csv', index=False)

    if args.evaluate_flag:
        print("Running evaluation...");time.sleep(1)    # Run evaluation
        test_data = pd.read_csv(args.training_data + '_test.csv').dropna()
        SE = SynthEval(training_data, test_data)
        SE.evaluate(synthetic_data, args.label_col,"full_eval", cls_acc={'F1_type': 'macro'})

    print("All done! Exiting..."); time.sleep(5)
    exit()