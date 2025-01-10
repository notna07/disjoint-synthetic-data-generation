# Description: Module for holding joining strategies
# Date: 18-11-2024
# Author : Anton D. Lautrup

import warnings

import pandas as pd

from pandas import DataFrame
from typing import Dict, List
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

from disjoint_generative_model.utils.joining_validator import JoiningValidator
class UsingJoiningValidator(JoinStrategy):
    """ Concrete Strategy for joining dataframes using a JoiningValidator model."""
    def __init__(self, join_validator_model: JoiningValidator = JoiningValidator(threshold=0.5, verbose=False),
                 patience: int = 1, 
                 min_iter: int = 5,
                 max_iter: int = 100,
                 max_size: int = int(1e6),
                 threshold: float = 0.5,
                 threshold_decay: float = 0,
                 ) -> None:
        """ Joins the dataframes randomly, in an iterative process 
        where bad joins are removed by a validator model.

        Args:
            join_validator_model (JoiningValidator): The model to use for validating joins.
            patience (int): Number of rounds without improvement before stopping.
            min_iter (int): The minimum number of iterations to perform.
            max_iter (int): The maximum number of iterations to perform.
            max_size (int): The maximum size of the joined dataframe.
            threshold (float): The threshold parameter for the validator model.
            threshold_decay (float): The decay rate for the threshold parameter.
        """
        self.join_validator = join_validator_model
        self.patience = patience
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.max_size = max_size
        self.join_validator.threshold = threshold
        self.threshold_decay = threshold_decay
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
        while while_index < self.max_iter and len(data[list(data.keys())[0]]) > 0:
            for key, _ in data.items():
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
    
from disjoint_generative_model.utils.joining_validator import JoiningValidatorOneClass
class UsingOneClassValidator(JoinStrategy):
    """ Concrete Strategy for joining dataframes using an outlier detection model."""
    def __init__(self, 
                 join_validator_model: JoiningValidatorOneClass = JoiningValidatorOneClass(threshold=-0.5, verbose=False),
                 patience: int = 5,
                 min_iter: int = 5,
                 max_iter: int = 50,
                 max_size: int = int(1e6),
                 threshold: float = -0.5,
                 threshold_decay: float = 0.01,
                 ) -> None:
        """ Joins the dataframes using a OneClass model.

        Args:
            join_validator_model (JoiningValidatorOneClass): The model to use for validating joins.
            patience (int): Number of rounds without improvement before stopping.
            min_iter (int): The minimum number of iterations to perform.
            max_iter (int): The maximum number of iterations to perform.
            max_size (int): The maximum size of the joined dataframe.
            threshold (float): The threshold parameter for the outlier model.
            threshold_decay (float): The decay rate for the threshold parameter.
        """
        self.join_validator = join_validator_model
        self.patience = patience
        self.min_iter = min_iter
        self.max_iter = max_iter
        self.max_size = max_size
        self.join_validator.threshold = threshold
        self.threshold_decay = threshold_decay
        pass

    def join(self, data: Dict[str, DataFrame]) -> DataFrame:
        """ Joins the dataframes.

        Args:
            data (Dict[str, DataFrame]): A dictionary of dataframes.

        Returns:
            DataFrame: The joined dataframe.

        Example:
            >>> from sklearn.ensemble import IsolationForest
            >>> import pandas as pd
            >>> dict_dfs = {
            ...    'df1': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 
            ...    'df2': pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
            ...    }
            >>> validator = JoiningValidatorOneClass(IsolationForest(), verbose = False)
            >>> validator.fit_classifier(dict_dfs, 
            ...                             number_of_k_fold=2, 
            ...                             random_state=42
            ...                             )
            >>> strategy = UsingOneClassValidator(validator)
            >>> result = strategy.join(dict_dfs)
            >>> isinstance(result, pd.DataFrame)
            True
        """
        while_index = 0
        df_good_joins = None

        patience_counter = 0
        while while_index < self.max_iter and len(data[list(data.keys())[0]]) > 0:
            for key, _ in data.items():
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