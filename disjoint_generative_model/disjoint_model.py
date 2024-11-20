# Description: Disjoint Generative Model Manager Class
# Date: 14-11-2024
# Version: 0.1
# Author : Anton D. Lautrup

from pandas import DataFrame
from typing import Dict, List

from .utils.dataset_manager import DataManager
from .utils.joining_strategies import JoinStrategy, Concatenating

from .utils.generative_model_adapters import generate_synthetic_data


class DisjointGenerativeModels:
    """ Class for managing disjoint generative models.

    Attributes:
        original_data (DataFrame): The original (training) data.
        training_data (Dict[str, DataFrame]): The training data, split into different splits.
        generative_models (List[str]): The generative models to use.
        used_splits (Dict[str, List[str]]): The divisions of columns actually used.
        synthetic_data (DataFrame): The synthetic data (once generated).
    """
    def __init__(self,
                 training_data,
                 generative_models: List[str] | Dict[str, List[str]],
                 prepared_splits: Dict[str, List[str]] = None,
                 joining_strategy: JoinStrategy = None
                 ):
        """ Initialize the DisjointGenerativeModels class.

        Args:
            training_data (DataFrame): The training data (before splitting).
            generative_models (List[str] | Dict[str, List[str]]): The generative models to use (can add column name lists).
            prepared_splits (Dict[str, List[str]]): Predefined splits of columns, if none use random splits for each model.
            joining_strategy (JoinStrategy): The strategy for joining dataframes, if none defaults to concatenation.
        """
        self.original_data = training_data
        self.generative_models = generative_models
        self.used_splits = prepared_splits

        self._strategy = joining_strategy
        pass
    
    @property
    def strategy(self) -> JoinStrategy:
        """ Get the current strategy for joining dataframes."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: JoinStrategy) -> None:
        """ Change the current strategy for joining dataframes."""
        self._strategy = strategy

    def _setup(self):
        """ Perform the initial setup of the data and models."""
        if self.used_splits is None:
            if isinstance(self.generative_models, Dict):
                self.used_splits = self.generative_models

        self.dm = DataManager(self.original_data.copy(), self.used_splits, len(self.generative_models))
        
        self.training_data = self.dm.encoded_dataset_dict
        self.used_splits = self.dm.column_splits

        if hasattr(self._strategy, 'join_validator'):
            self._strategy.join_validator.fit_classifier(self.training_data)

        if isinstance(self.generative_models, Dict):     # get model names from dict to list
            self.generative_models = list(self.generative_models.keys())
        pass

    def _evaluate_splits(self):
        # TODO: Calculate fraction of identical rows between joined data and reference data
        # TODO: Calculate record number difference between joined data and reference data
        # TODO: Calculate some other metric
        pass

    def fit_generate(self):
        """ Fit the generative models to the training data and generate synthetic data.

        Returns:
            DataFrame: The synthetic data.

        Example:
            >>> import pandas as pd
            >>> df = pd.read_csv('tests/dummy_train.csv')
            >>> dgm = DisjointGenerativeModels(df, ['synthpop', 'privbayes'], joining_strategy=Concatenating())
            >>> dgm.fit_generate() # doctest: +ELLIPSIS
            -etc-
        """
        self._setup()

        syn_dfs_dict = {}
        for model, (split_name, train_data) in zip(self.generative_models, self.training_data.items()):
            df_syn = generate_synthetic_data(train_data, model)
            syn_dfs_dict[split_name] = df_syn

        synthetic_data = self.conduct_joining(syn_dfs_dict)
        
        self.synthetic_data = self.dm.postprocess(synthetic_data)

        return self.synthetic_data

    def conduct_joining(self, data: Dict[str, DataFrame]) -> DataFrame:
        """ Perform the joining of dataframes using the current strategy.
        
        Args:
            data (Dict[str, DataFrame]): The data to join.
        
        Returns:
            DataFrame: The joined data.

        Example:
            >>> import pandas as pd
            >>> dict = {'split1': pd.DataFrame({'A': [1, 2], 'B': [3, 4]}), 'split2': pd.DataFrame({'C': [5, 6], 'D': [7, 8]})}
            >>> dgm = DisjointGenerativeModels(None, None, None, Concatenating())
            >>> dgm.conduct_joining(dict) # doctest: +NORMALIZE_WHITESPACE
               A  B  C  D
            0  1  3  5  7
            1  2  4  6  8
        """
        if self._strategy is None:
            self.strategy(Concatenating())

        return self._strategy.join(data)

if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()