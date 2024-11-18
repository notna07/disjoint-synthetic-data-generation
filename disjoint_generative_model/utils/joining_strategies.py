# Description: Module for holding joining strategies
# Date: 18-11-2024
# Version: 0.1
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
    def join(self, data: Dict[str, DataFrame]) -> DataFrame:
        pass

class UsingJoiningValidator(Strategy):
    def __init__(self) -> None:
        pass

    def join(self, data: Dict[str, DataFrame]) -> DataFrame:
        pass


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
