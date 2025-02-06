# Description: Module for holding joining strategies
# Date: 18-11-2024
# Author : Anton D. Lautrup

import time
import warnings

import pandas as pd

from pandas import DataFrame
from typing import Dict, List, Literal
from abc import ABC, abstractmethod

class JoinStrategy(ABC):
    """ Strategy interface for joining dataframes. Declares operations common 
    to all supported algorithms.

    The JoiningModule uses this interface to call the algorithm defined by concrete strategies.

    Required Methods:
        join(data: Dict[str, DataFrame]) -> DataFrame: Joins the dict of dataframes.
    """

    @abstractmethod
    def join(self, data: Dict[str, DataFrame]) -> DataFrame:
        """ Joins the dataframes using concatenation.

        Args:
            data (Dict[str, DataFrame]): A dictionary of dataframes.

        Returns:
            DataFrame: The joined dataframe.
        """
        pass

""" Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
"""

class Concatenating(JoinStrategy):
    """ Concrete Strategy for joining dataframes using concatenation."""
    def join(self, data: Dict[str, DataFrame]) -> DataFrame:
        """ Joins the dataframes using concatenation.

        Args:
            data (Dict[str, DataFrame]): A dictionary of dataframes.

        Returns:
            DataFrame: The joined dataframe.

        Example:
            >>> import pandas as pd
            >>> data = {
            ...     'df1': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),
            ...     'df2': pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
            ... }
            >>> strategy = Concatenating()
            >>> strategy.join(data) # doctest: +NORMALIZE_WHITESPACE
               A  B  C   D
            0  1  4  7  10
            1  2  5  8  11
            2  3  6  9  12
        """
        joined_data = pd.concat(data.values(), axis=1)
        return joined_data

class RandomJoining(JoinStrategy):
    """ Concrete Strategy for randomly joining dataframes."""
    def __init__(self, random_state: int = None) -> None:
        self.random_state = random_state
        pass

    def join(self, data: Dict[str, DataFrame]) -> DataFrame:
        """ Joins the dataframes using concatenation and shuffles the rows.
        
        Args:
            data (Dict[str, DataFrame]): A dictionary of dataframes.

        Returns:
            DataFrame: The joined dataframe.

        Example:
            >>> import pandas as pd
            >>> data = {
            ...     'df1': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}),
            ...     'df2': pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
            ... }
            >>> strategy = RandomJoining(42)
            >>> strategy.join(data) # doctest: +NORMALIZE_WHITESPACE
               A  B  C   D
            0  1  4  7  10
            1  2  5  8  11
            2  3  6  9  12
        """
        for key in data:
            data[key] = data[key].sample(frac=1, random_state = self.random_state)
        return pd.concat(data.values(), axis=1).reset_index(drop=True)

# Validator joining strategy behaviours
strict_behaviour = {
    "patience": 3,
    "min_iter": 5,
    "max_iter": 100,
    "threshold": 0.5,
    "threshold_decay": 0,
    'auto_threshold_percentage': None
}

adaptive_behaviour = {
    "patience": 5,
    "min_iter": 5,
    "max_iter": 100,
    "threshold": 'auto',
    "threshold_decay": 0.01,
    'auto_threshold_percentage': 0.1
}

from disjoint_generative_model.utils.joining_validator import JoiningValidator, OneClassValidator, OutlierValidator
class UsingJoiningValidator(JoinStrategy):
    """ Concrete Strategy for joining dataframes using a JoiningValidator model."""
    def __init__(self, join_validator_model: JoiningValidator | OneClassValidator | OutlierValidator = JoiningValidator(verbose=False),
                 behaviour: Literal['standard', 'strict', 'adaptive'] = 'standard',
                 patience: int = None, 
                 min_iter: int = None,
                 max_iter: int = None,
                 max_size: int = int(1e6),
                 threshold: float | Literal['auto'] = None,
                 threshold_decay: float = None,
                 auto_threshold_percentage: float = None
                 ) -> None:
        """ Joins the dataframes randomly, in an iterative process 
        where bad joins are removed by a validator model.

        Args:
            join_validator_model (JoiningValidator | OneClassValidator): The model to use for validating joins.
            behaviour (Literal['standard', 'strict', 'adaptive']): Pre-configurations of options.
            patience (int): Number of rounds without improvement before stopping.
            min_iter (int): The minimum number of iterations to perform.
            max_iter (int): The maximum number of iterations to perform.
            max_size (int): The maximum size of the joined dataframe.
            threshold (float): The threshold parameter for the validator model.
            threshold_decay (float): The decay rate for the threshold parameter.
            auto_threshold_percentage (float): The percentage of the querries to admit with first threshold choice.
        """
        self.join_validator = join_validator_model
        
        if behaviour == 'standard':
            params = self.join_validator.get_standard_behavior()
        elif behaviour == 'strict':
            params = strict_behaviour
        elif behaviour == 'adaptive':
            params = adaptive_behaviour

        self.patience = patience if patience is not None else params['patience']
        self.min_iter = min_iter if min_iter is not None else params['min_iter']
        self.max_iter = max_iter if max_iter is not None else params['max_iter']
        self.max_size = max_size
        self.join_validator.threshold = threshold if threshold is not None else params['threshold']
        self.join_validator.auto_threshold_percentage = auto_threshold_percentage if auto_threshold_percentage is not None else params['auto_threshold_percentage']
        self.threshold_decay = threshold_decay if threshold_decay is not None else params['threshold_decay']
        pass

    def join(self, data: Dict[str, DataFrame]) -> DataFrame:
        """ Joins the dataframes.

        Args:
            data (Dict[str, DataFrame]): A dictionary of dataframes.

        Returns:
            DataFrame: The joined dataframe.

        Example:
            >>> from sklearn.linear_model import LogisticRegression
            >>> import pandas as pd
            >>> dict_dfs = {
            ...    'df1': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 
            ...    'df2': pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
            ...    }
            >>> validator = JoiningValidator(LogisticRegression(), verbose = False)
            >>> validator.fit_classifier(dict_dfs, 
            ...                             number_of_stratified_k_fold=2, 
            ...                             num_batches_of_bad_joins=2, 
            ...                             random_state=42
            ...                             )
            >>> strategy = UsingJoiningValidator(validator)
            >>> result = strategy.join(dict_dfs)
            >>> isinstance(result, pd.DataFrame)
            True
            
        """
        while_index = 0
        df_good_joins = None
        
        patience_counter = 0
        while while_index < self.max_iter and len(data[list(data.keys())[0]]) > 0:
            for i, key in enumerate(data.keys()):
                data[key] = data[key].sample(frac=1).reset_index(drop=True)
            df_attempt = pd.concat(data.values(), axis=1)

            df_attempt_good_joins = self.join_validator.validate(df_attempt)

            df_attempt_good_joins_idx = list(sorted(df_attempt_good_joins.index))

            # remove the good joins from the data
            for key, _  in data.items():
                data[key].drop(df_attempt_good_joins_idx, axis=0, inplace=True)
            
            if df_good_joins is None:
                df_good_joins = df_attempt_good_joins.reset_index(drop=True)
            else:
                df_good_joins = pd.concat([df_good_joins, df_attempt_good_joins], axis=0).reset_index(drop=True)

            ### Early stopping ###
            # First check if we have enough good elements
            if len(df_good_joins) > self.max_size:
                break

            # Next check and warn if threshold is too high to begin with
            if while_index <= self.min_iter and len(df_good_joins) == 0:
                warnings.warn("No good joins found in the first iterations, consider lowering the threshold!")
            
            # Finally check if we are still adding items
            if while_index >= self.min_iter:
                if len(df_attempt_good_joins) == 0 and patience_counter < self.patience:
                    patience_counter += 1
                    self.join_validator.threshold = self.join_validator.threshold - self.threshold_decay
                elif patience_counter >= self.patience:
                    break
                else:
                    patience_counter = 0

            while_index += 1
        
        if len(df_good_joins) <= self.max_size:
            warnings.warn(f"Expected size not reached, outputting only {len(df_good_joins)} items!")
        return df_good_joins[:self.max_size]

if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()