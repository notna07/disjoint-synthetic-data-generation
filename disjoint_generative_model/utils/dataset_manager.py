# Description: Class for keeping track of the original dataset and for postprocessing the generated dataset
# Date: 14-11-2024
# Version: 0.1
# Author : Anton D. Lautrup

from typing import Dict, List
from pandas import DataFrame

def random_split_columns(dataset: DataFrame, split_ratios: Dict[str, float], random_state: int = None) -> Dict[str, List[str]]:
    """ Randomly split the columns of a dataset into different splits

    Example:
        >>> import pandas as pd
        >>> dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        >>> split_ratios = {'split1': 2, 'split2': 1}
        >>> result = random_split_columns(dataset, split_ratios, random_state=1)
        >>> sorted(result.keys())
        ['split1', 'split2']
        >>> sorted(result['split1'].columns)
        ['A', 'C']
        >>> sorted(result['split2'].columns)
        ['B']
    """
    # Normalise the split sizes
    divisor = sum(split_ratios.values())
    split_sizes = {split: int(ratio/divisor*dataset.shape[1]) for split, ratio in split_ratios.items()}

    # Check if the split sizes are valid
    sum_diff = sum(split_sizes.values()) - dataset.shape[1]
    if sum_diff != 0:
        split_sizes.iloc[0] += sum_diff
        raise Warning(f"Split sizes adjusted to {split_sizes}")
    
    # Randomly shuffle the columns
    dataset = dataset.sample(frac=1, axis=1, random_state=random_state)

    # Split the columns
    split_columns = {}
    for split, size in split_sizes.items():
        split_columns[split] = dataset.iloc[:, :size]
        dataset = dataset.iloc[:, size:]

    if dataset.shape[1] > 0:
        raise Warning(f"Remainder {dataset.columns} of columns were not included in any split!")
        
    return split_columns

class DataManager:
    """
    A class to manage datasets, splitting, and reverse encoding.

    Attributes:
        original_dataset (DataFrame): The original dataset.
        encoded_dataset_dict (Dict[str, DataFrame]): The dictionary of encoded datasets.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> dm = DataManager(df, prepared_splits = {'split1': ['A'], 'split2': ['B']})
        >>> dm.encoded_dataset_dict['split1'].columns.tolist()
        ['A']
        >>> dm.encoded_dataset_dict['split2'].columns.tolist()
        ['B']
    """
    def __init__(self, original_dataset: DataFrame, 
                 prepared_splits: Dict[str, List[str]] = None
                 ):
        """ Initialize the DataManager with the original dataset and optional prepared splits.

        Args:
            original_dataset (DataFrame): The original dataset.
            prepared_splits (Dict[str, List[str]], optional): A dictionary where keys are split names and values are lists of column names. Defaults to None.
        """
        self.original_dataset = original_dataset

        if prepared_splits is not None:
            self.encoded_dataset_dict = self._setup_column_splits(prepared_splits)
        else:
            self.encoded_dataset_dict = self._setup_column_splits(
                random_split_columns(original_dataset, {'split1': 1, 'split2': 1})  # Setup with a random 50/50 split
                )

        pass

    def _setup_column_splits(self, prepared_splits: Dict[str, List[str]]) -> Dict[str, DataFrame]:
        """ Setup the column splits based on the prepared splits.

        Args:
            prepared_splits (Dict[str, List[str]]): A dictionary where keys are split names and values are lists of column names.

        Returns:
            Dict[str, DataFrame]: A dictionary where keys are split names and values are the corresponding DataFrames.
        """
        return {split: self.original_dataset[columns] for split, columns in prepared_splits.items()}

    def postprocess(self, generated_dataset: DataFrame) -> DataFrame:
        """ Postprocess the generated dataset to match the original dataset's columns.

        Args:
            generated_dataset (DataFrame): The generated dataset.

        Returns:
            DataFrame: The postprocessed dataset.

        Example:
            >>> import pandas as pd
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> dm = DataManager(df, {'split1': ['A'], 'split2': ['B']})
            >>> generated_df = pd.DataFrame({'B': [7, 8], 'A': [5, 6]})
            >>> postprocessed_df = dm.postprocess(generated_df)
            >>> postprocessed_df.columns.tolist()
            ['A', 'B']
        """
        return generated_dataset[self.original_dataset.columns]


if __name__ == "__main__":
    import doctest
    doctest.testmod()