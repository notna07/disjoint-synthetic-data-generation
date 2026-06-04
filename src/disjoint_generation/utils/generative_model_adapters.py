# Description: Script for generating synthetic datasets in a variety of ways
# Author: Anton D. Lautrup
# Date: 18-11-2024

import os
import sys
import json
import glob
import shutil
import tempfile
import subprocess

import numpy as np
import pandas as pd
import torch

from typing import List, Optional, Union
from pandas import DataFrame
from abc import ABC, abstractmethod
from importlib.resources import files

def _load_data(file_name: str) -> DataFrame:
    df_train = pd.read_csv(file_name + '.csv').dropna()
    return df_train

def _write_data(df: DataFrame, file_name: str) -> None:
    df.to_csv(file_name, index=False)

def _cleanup_files(file_names: List[str]) -> None:
    for file_name in file_names:
        if os.path.exists(file_name):
            os.remove(file_name)
    pass

class DataGeneratorAdapter(ABC):
    """ Abstract class for data generator adapters.

    Required Methods:
        generate(train_data_name: str | DataFrame, num_to_generate: int = None, id: int = 0) -> DataFrame: Generate synthetic data.
    """
    def __str__(self):
        return f"{self.name}"

    @abstractmethod
    def generate(self, train_data_name: str | DataFrame, num_to_generate: int = None, seed: int = None, id = 0, **kwargs) -> DataFrame:
        """ Generate synthetic data based on the training data.

        Args:
            train_data_name (str): The name of the training data file.

        Returns:
            DataFrame: The generated synthetic data.
        """
        pass


_SYNTHCITY_MODELS = {
    'ctgan',
    'adsgan',
    'tvae',
    'nflow',
    'ddpm',
    'arf',
    'dpgan',
    'privbayes',
}

class SynthCityAdapter(DataGeneratorAdapter):
    """ SynthCity Adapter for generating synthetic data.

    Attributes:
        gen_model (str): The generative model to use.
    """
    def __init__(self, gen_model):
        self.gen_model = gen_model

    def generate(self, train_data: str | DataFrame, num_to_generate: int = None, seed: int = None, id = 0, **kwargs) -> DataFrame:
        """ Generate synthetic data using SynthCity. See the list of enabled models above.

        Reference:
            Qian, Z., Cebere, B.-C., & van der Schaar, M. (2023). Synthcity: facilitating innovative 
            use cases of synthetic data in different data modalities. http://arxiv.org/abs/2301.07573

        Args:
            train_data (str, DataFrame): The name of the training data file or the DataFrame.
            num_to_generate (int): The number of synthetic data points to generate.

        Returns:
            DataFrame: The generated synthetic data.

        Example:
            >>> adapter = SynthCityAdapter('privbayes')
            >>> df_syn = adapter.generate('tests/dummy_train') # doctest: +ELLIPSIS
            -etc-
            >>> isinstance(df_syn, pd.DataFrame)
            True
        """
        from synthcity.plugins import Plugins
        if isinstance(train_data, str):
            df_train = _load_data(train_data)
        else:
            df_train = train_data

        if seed is None: seed = np.random.randint(0, 1000000)
        syn_model = Plugins().get(self.gen_model, random_state = seed, **kwargs)
        syn_model.fit(df_train)

        if num_to_generate is None: num_to_generate = len(df_train)
        df_syn = syn_model.generate(count=num_to_generate).dataframe()

        ## Sometimes the model does not generate enough samples (i.e., small categorical subsets have duplicates removed)
        if len(df_syn) < num_to_generate:
            df_mis = syn_model.generate(count=num_to_generate-len(df_syn)).dataframe()
            df_syn = pd.concat([df_syn, df_mis], ignore_index=True)

        if len(df_syn) < num_to_generate:
            print(f"Warning: {num_to_generate-len(df_syn)} missing samples. Re-sampling to fill.")
            df_mis = df_syn.sample(n=num_to_generate-len(df_syn), replace=True)
            df_syn = pd.concat([df_syn, df_mis], ignore_index=True)

        return df_syn[:num_to_generate]

class SynthPopAdapter(DataGeneratorAdapter):
    def generate(self, train_data: str | DataFrame, num_to_generate: int = None, seed: int = None, id = 0,  **kwargs) -> DataFrame:
        """ Generate synthetic data using SynthPop in R using subprocess.
        Be sure to check that R is installed and Rscript is a valid command in the terminal.

        Reference:
            Nowok, B., Raab, G. M., & Dibben, C. (2016). synthpop: Bespoke Creation of Synthetic Data in R. 
            Journal of Statistical Software, 74(11), 1--26. https://doi.org/10.18637/jss.v074.i11

        Args:
            train_data (str, DataFrame): The name of the training data file or the DataFrame.
            num_to_generate (int): The number of synthetic data points to generate.

        Returns:
            DataFrame: The generated synthetic data.

        Example:
            >>> adapter = SynthPopAdapter()
            >>> df_syn = adapter.generate('tests/dummy_train') # doctest: +SKIP
            >>> isinstance(df_syn, pd.DataFrame) # doctest: +SKIP
            True
        """
        # if self.r_access == 'subprocess':

        if isinstance(train_data, str):
            train_data_name = train_data
        else:
            train_data_name = f'synthpop_temp_{id}'
            _write_data(train_data, train_data_name + '.csv')

        df_train = _load_data(train_data_name)

        info_dir = 'synthesis_info_' + train_data_name.split('/')[0]
        if not os.path.exists(info_dir):
            os.makedirs(info_dir)

        r_script_path = files('disjoint_generation').joinpath('utils/subprocess/synthpop_subprocess.R')
        command = [
                    "Rscript",
                    str(r_script_path),
                    train_data_name +".csv",
                    train_data_name + "_synthpop",
                    str(num_to_generate) if num_to_generate is not None else str(len(df_train)),
                    str(seed) if seed is not None else "",
                ]
        subprocess.run(command, check=True)

        df_syn = pd.read_csv(train_data_name + '_synthpop.csv')
        df_syn.columns = [col for col in df_train.columns]

        _cleanup_files(['synthesis_info_' + train_data_name + '_synthpop.txt', 
                        train_data_name + '_synthpop.csv', 
                        f'synthpop_temp_{id}.csv'])

        os.removedirs(info_dir)
        return df_syn

class DataSynthesizerAdapter(DataGeneratorAdapter):
    """ DataSynthesizer Adapter for generating synthetic data.

    Attributes:
        epsilon (float): The privacy parameter epsilon for differential privacy (0 is turned off).
    """
    def __init__(self, epsilon: float = 0):
        self.epsilon = epsilon

    def generate(self, train_data: str | DataFrame, num_to_generate: int = None, seed: int = None, id = 0, **kwargs) -> DataFrame:
        """ Generate synthetic data using DataSynthesizer.

        Reference:
            Ping, H., Stoyanovich, J., & Howe, B. 2017. DataSynthesizer: Privacy-Preserving Synthetic Datasets. 
            In Proceedings of the 29th International Conference on Scientific and Statistical Database Management (SSDBM '17). 
            Association for Computing Machinery, New York, NY, USA, Article 42, 1--5. https://doi.org/10.1145/3085504.3091117

        Args:
            train_data (str, DataFrame): The name of the training data file or the DataFrame.
            num_to_generate (int): The number of synthetic data points to generate.

        Returns:
            DataFrame: The generated synthetic data.

        Example:
            >>> adapter = DataSynthesizerAdapter()
            >>> df_syn = adapter.generate('tests/dummy_train') # doctest: +ELLIPSIS
            -etc-
            >>> isinstance(df_syn, pd.DataFrame)
            True
        """
        from DataSynthesizer.DataDescriber import DataDescriber
        from DataSynthesizer.DataGenerator import DataGenerator
        if isinstance(train_data, str):
            train_data_name = train_data
        else:
            train_data_name = f'datasynthesizer_temp_{id}'
            _write_data(train_data, train_data_name + '.csv')
        
        df_train = _load_data(train_data_name)

        description_file = train_data_name + f"_datasynthesizer_{id}_info.json"

        describer = DataDescriber(category_threshold=10)
        describer.describe_dataset_in_correlated_attribute_mode(dataset_file = train_data_name +'.csv', 
                                                                epsilon=self.epsilon, 
                                                                k=2,
                                                                attribute_to_is_categorical={},
                                                                seed = seed
                                                                )
        describer.save_dataset_description_to_file(description_file)

        generator = DataGenerator()

        if num_to_generate is None: num_to_generate = len(df_train)
        generator.generate_dataset_in_correlated_attribute_mode(num_to_generate, description_file, seed = seed)

        df_syn = generator.synthetic_dataset

        _cleanup_files([description_file, f'datasynthesizer_temp_{id}.csv'])

        return df_syn

class DebugAdapter(DataGeneratorAdapter):
    def generate(self, train_data: str | DataFrame, num_to_generate: int = None, seed: int = None, id = 0, **kwargs) -> DataFrame:
        if isinstance(train_data, str):
            train_data = _load_data(train_data)
        return train_data


class TabDiffAdapter(DataGeneratorAdapter):
    """Adapter that trains TabDiff and generates synthetic tabular data.

    Reference:
        Juntong Shi, Minkai Xu, Harper Hua, Hengrui Zhang, Stefano Ermon, and Jure Leskovec. Tabdiff: a 
        mixed-type diffusion model for tabular data generation. In Proceedings of The Thirteenth International 
        Conference on Learning Representations, ICLR 2025, Singapore, 2025. OpenReview.net.

    Args:
        task_type: 'binclass', 'multiclass', or 'regression'.
        target_col: Name or integer index of the target column.
            Auto-detected when None (last column, or one named 'target'/'label'/etc.).
        num_col_idx: Explicit list of numerical column indices.
            Auto-detected when None.
        cat_col_idx: Explicit list of categorical column indices.
            Auto-detected when None.
        steps: Number of training epochs (default: 2500).
        device: Torch device string, e.g. 'cuda:0' or 'cpu'.
            Auto-selected when None.
        learnable_schedule: Use per-column learnable noise schedule. Default True.
        sample_batch_size: Batch size used during generation. Default 10000.

    Example:
        >>> adapter = TabDiffAdapter(task_type='binclass')
        >>> df_syn = adapter.generate(df_train, num_to_generate=200, seed=0) # doctest: +SKIP
        >>> isinstance(df_syn, pd.DataFrame) # doctest: +SKIP
        True
    """

    name = 'tabdiff'

    def __init__(
        self,
        task_type: str = 'binclass',
        target_col: Union[str, int, None] = None,
        num_col_idx: Optional[List[int]] = None,
        cat_col_idx: Optional[List[int]] = None,
        steps: int = 2500,
        device: Optional[str] = None,
        learnable_schedule: bool = True,
        sample_batch_size: int = 10000,
    ):
        self.task_type = task_type
        self.target_col = target_col
        self.num_col_idx = num_col_idx
        self.cat_col_idx = cat_col_idx
        self.steps = steps
        self.device = device
        self.learnable_schedule = learnable_schedule
        self.sample_batch_size = sample_batch_size

    def generate(
        self,
        train_data: Union[str, DataFrame],
        num_to_generate: Optional[int] = None,
        seed: Optional[int] = None,
        id: int = 0,
        **kwargs,
    ) -> DataFrame:
        """Train TabDiff on *train_data* and return a synthetic DataFrame.

        Args:
            train_data (str, DataFrame): Training DataFrame or path to a CSV file
                (without the '.csv' extension).
            num_to_generate (int): Number of synthetic rows. Defaults to len(train_data).
            seed (int): Random seed for reproducibility.
            id (int): Instance index to prevent temp-directory collisions in parallel runs.

        Returns:
            DataFrame: Synthetic data with the same columns as the training data.
        """
        from .tabdiff_implementation import build_and_train, _tabdiff_find_target_column, _tabdiff_detect_column_types, _tabdiff_write_npy_files, _tabdiff_build_info, _tabdiff_restore_column_types

        # --- Load data ---
        if isinstance(train_data, str):
            df = _load_data(train_data)
        else:
            df = train_data.dropna().reset_index(drop=True)

        if num_to_generate is None:
            num_to_generate = len(df)

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # --- Store original dtypes for restoration ---
        original_dtypes = df.dtypes.copy()

        # --- Resolve target column ---
        column_names = df.columns.tolist()
        if self.target_col is None:
            target_col_idx = _tabdiff_find_target_column(df)
        elif isinstance(self.target_col, str):
            target_col_idx = [column_names.index(self.target_col)]
        else:
            target_col_idx = [int(self.target_col)]

        # --- Resolve column types ---
        num_col_idx = self.num_col_idx
        cat_col_idx = self.cat_col_idx
        if num_col_idx is None or cat_col_idx is None:
            _num, _cat = _tabdiff_detect_column_types(df, target_col_idx)
            if num_col_idx is None:
                num_col_idx = _num
            if cat_col_idx is None:
                cat_col_idx = _cat

        # --- 90 / 10 train-test split ---
        n = len(df)
        n_train = max(1, int(n * 0.9))
        rng = np.random.default_rng(seed if seed is not None else 42)
        idx = rng.permutation(n)
        df_train = df.iloc[idx[:n_train]].reset_index(drop=True)
        df_test = df.iloc[idx[n_train:]].reset_index(drop=True)

        # --- Temporary working directory ---
        tmp_root = tempfile.mkdtemp(prefix=f'tabdiff_adapter_{id}_')
        name = f'tabdiff_tmp_{id}'
        try:
            _tabdiff_write_npy_files(df_train, df_test, num_col_idx, cat_col_idx, target_col_idx, tmp_root)
            df.to_csv(os.path.join(tmp_root, f'{name}.csv'), index=False)

            info = _tabdiff_build_info(
                df_train, df_test, num_col_idx, cat_col_idx,
                target_col_idx, self.task_type, tmp_root, name,
            )
            with open(os.path.join(tmp_root, 'info.json'), 'w') as fh:
                json.dump(info, fh, indent=2)

            device = torch.device(
                self.device if self.device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
            )

            syn_df = build_and_train(
                data_dir=tmp_root,
                info=info,
                num_samples=num_to_generate,
                device=device,
                steps=self.steps,
                learnable_schedule=self.learnable_schedule,
                sample_batch_size=self.sample_batch_size,
                model_save_path=None,
                result_save_path=None,
            )
            
            # --- Restore original column types ---
            syn_df = _tabdiff_restore_column_types(syn_df, df, original_dtypes)
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

        return syn_df

def _get_adapter(gen_model: str) -> DataGeneratorAdapter:
    if gen_model in _SYNTHCITY_MODELS:
        return SynthCityAdapter(gen_model)
    if gen_model == 'synthpop':
        return SynthPopAdapter()
    if gen_model == 'datasynthesizer':
        return DataSynthesizerAdapter()
    if gen_model == 'datasynthesizer-dp':
        return DataSynthesizerAdapter(epsilon=0.1)
    if gen_model == 'tabdiff':
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        return TabDiffAdapter(device=device)
    if gen_model == 'debug':
        return DebugAdapter()
    raise NotImplementedError("The chosen generative model could not be run!")

def generate_synthetic_data(train_data: DataFrame | str, gen_model: str, num_to_generate: int = None, seed: int = None, id = 0, **kwargs) -> DataFrame:
    """ Generate synthetic data using the specified generative model.

    Args:
        train_data (DataFrame, str): The training data DataFrame or filename.
        gen_model (str): The name of the generative model.
        id (int): instance index (to prevent overwriting files for parallel runs).
        num_to_generate (int): The number of synthetic data points to generate.

    Returns:
        DataFrame: The generated synthetic data.

    Example:
        >>> df_syn = generate_synthetic_data('tests/dummy_train', 'privbayes')
        >>> isinstance(df_syn, pd.DataFrame)
        True
    """
    if isinstance(gen_model, str):
        adapter = _get_adapter(gen_model)
        df_syn = adapter.generate(train_data, num_to_generate, seed, id, **kwargs)
    elif isinstance(gen_model, DataGeneratorAdapter):
        df_syn = gen_model.generate(train_data, num_to_generate, seed, id, **kwargs)
    else:
        raise NotImplementedError("DGMs (adapter): The chosen generative model could not be run!")
    return df_syn


if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()