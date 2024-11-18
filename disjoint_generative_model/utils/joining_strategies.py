# Description: Module for holding joining strategies
# Date: 18-11-2024
# Author : Anton D. Lautrup

import pandas as pd

from pandas import DataFrame
from typing import Dict, List
from abc import ABC, abstractmethod

class Strategy(ABC):
    """ Strategy interface for joining dataframes. Declares operations common 
    to all supported algorithms.

    The JoiningModule uses this interface to call the algorithm defined by concrete strategies.
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

class Concatenating(Strategy):
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

class RandomJoining(Strategy):
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
        joined_data = pd.concat(data.values(), axis=1).reset_index(drop=True)
        return joined_data

from joining_validator import JoiningValidator
class UsingJoiningValidator(Strategy):
    def __init__(self, join_validator_model: JoiningValidator,
                 patience: int = 10, 
                 max_iter: int = 100,
                 ) -> None:
        """ Joins the dataframes randomly, in an iterative process 
        where bad joins are removed by a validator model.

        Args:
            join_validator_model (JoiningValidator): The model to use for validating joins.
            patience (int): Threshold number of items to add in each iteration before stopping.
            max_iter (int): The maximum number of iterations to perform.
        """
        self.join_validator = join_validator_model
        self.patience = patience
        self.max_iter = max_iter
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
        import numpy as np

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

            while_index += 1
            if len(df_attempt_good_joins) < self.patience:
                break
        return df_good_joins

class JoiningModule(ABC):
    """ Module for joining dataframes using different strategies.
    
    Attributes:
        strategy (Strategy): The strategy to use for joining dataframes.
    """

    def __init__(self, strategy: Strategy) -> None:
        """ Module for joining dataframes using different strategies.

        Args:
            strategy (Strategy): The strategy to use for joining dataframes.
        """

        self._strategy = strategy
        pass

    @property
    def strategy(self) -> Strategy:
        """ Get the current strategy for joining dataframes."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        """ Change the current strategy for joining dataframes."""
        self._strategy = strategy

    def perform_joining(self, data: Dict[str, DataFrame]) -> DataFrame:
        """ Perform the joining of dataframes using the current strategy.

        Args:
            data (Dict[str, DataFrame]): The data to join.
        """
        self.joined_data = self._strategy.join(data)
        return self.joined_data.copy()
    
    def evaluate_joining(self, reference_data) -> None:
        # TODO: Calculate fraction of identical rows between joined data and reference data
        # TODO: Calculate record number difference between joined data and reference data
        # TODO: Calculate some other metric
        pass


if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()

    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    # context = Context(ConcreteStrategyA())
    # print("Client: Strategy is set to normal sorting.")
    # context.do_some_business_logic()
    # print()

    # print("Client: Strategy is set to reverse sorting.")
    # context.strategy = ConcreteStrategyB()
    # context.do_some_business_logic()
