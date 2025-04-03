# Description: Class for keeping track of the original dataset and for postprocessing the generated dataset
# Date: 14-11-2024
# Author : Anton D. Lautrup

import warnings

import numpy as np

from pandas import DataFrame
from typing import Dict, List, Literal

from itertools import cycle

from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import OrdinalEncoder

def random_split_columns(dataset: DataFrame, split_ratios: Dict[str, float], random_state: int = None) -> Dict[str, List[str]]:
    """ Randomly split the columns of a dataset into different splits

    Example:
        >>> import pandas as pd
        >>> dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
        >>> split_ratios = {'split1': 2, 'split2': 1}
        >>> result = random_split_columns(dataset, split_ratios, random_state=1)
        >>> sorted(result.keys())
        ['split1', 'split2']
        >>> sorted(result['split1'])
        ['A', 'C']
        >>> sorted(result['split2'])
        ['B']
    """
    # Normalise the split sizes
    divisor = sum(split_ratios.values())
    split_sizes = {split: int(ratio/divisor*len(list(dataset.columns))) for split, ratio in split_ratios.items()}

    # Check if the split sizes are valid
    sum_diff = abs(sum(split_sizes.values()) - dataset.shape[1])
    if sum_diff != 0:
        for i in range(sum_diff):
            split_sizes[list(split_sizes.keys())[i]] += 1
        warnings.warn(f"Split sizes adjusted to {split_sizes}")
    
    # Randomly shuffle the columns
    dataset = dataset.sample(frac=1, axis=1, random_state=random_state)

    # Split the columns
    split_columns = {}
    for split, size in split_sizes.items():
        split_columns[split] = dataset.iloc[:, :size]
        dataset = dataset.iloc[:, size:]

    if dataset.shape[1] > 0:
        raise Exception(f"Remainder {dataset.columns} of columns were not included in any split!")
        
    return {key: list(frame.columns) for key,frame in split_columns.items()}

def correlated_distribute_columns(dataset: DataFrame, num_partitions: int,  min_size: int = 2, random_state: int = None):
    """ Partition the columns into sets based on correlation. Highly correlated items are (as much as possible) 
    placed into *different* partitions, to ensure that the validator models have more to go on.

    Example:
        >>> import pandas as pd
        >>> dataset = pd.DataFrame({'A': [1, 2, 3], 'B': [1, 0, 0], 'C': [2, 4, 6]})
        >>> correlated_distribute_columns(dataset, num_partitions=2, min_size=1, random_state=1)
        {'split0': ['A'], 'split1': ['B', 'C']}
    """
    corr_matrix = dataset.corr().abs()
    np.fill_diagonal(corr_matrix.values, 0)
    corr_matrix = corr_matrix.fillna(0)
    
    clustering = SpectralClustering(n_clusters=num_partitions, affinity='precomputed', assign_labels='kmeans', random_state=random_state)
    labels = clustering.fit_predict(1 - corr_matrix)
    
    clusters = {}
    for feature, cluster in zip(dataset.columns, labels):
        clusters.setdefault(cluster, []).append(feature)
    
    # Redistribute features into approximately equal-sized buckets
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)
    partitions = {f'split{i}': [] for i in range(num_partitions)}
    partition_cycle = cycle(range(num_partitions))

    for cluster in sorted_clusters:
        for feature in cluster:
            partitions[f'split{next(partition_cycle)}'].append(feature)
    return partitions

class DataManager:
    """
    A class to manage datasets, splitting, and reverse encoding.

    Attributes:
        original_dataset (DataFrame): The original dataset.
        encoded_dataset_dict (Dict[str, DataFrame]): The dictionary of encoded datasets.
        column_splits (Dict[str, List[str]]): The dictionary of the used column splits.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> dm = DataManager(df, prepared_splits = {'split1': ['A'], 'split2': ['B']})
        >>> dm.encoded_dataset_dict['split1'].columns.tolist()
        ['A']
        >>> dm.encoded_dataset_dict['split2'].columns.tolist()
        ['B']
        >>> dm2 = DataManager(df, num_random_splits=2)
    """
    def __init__(self, original_dataset: DataFrame, 
                 prepared_splits: Dict[str, List[str]] = None,
                 automated_splits: Literal['correlated', 'random'] = 'random',
                 num_automated_splits: int = 2,
                 random_state: int = None,
                 ):
        """ Initialize the DataManager with the original dataset and optional prepared splits.

        Args:
            original_dataset (DataFrame): The original dataset.
            prepared_splits (Dict[str, List[str]], optional): A dictionary where keys are split names and values are lists of column names. Defaults to None.
            num_random_splits (int, optional): The number of random splits to generate if prepared_splits is None. Defaults to 2.
        """

        # Convert categorical columns to numbers using OrdinalEncoder
        self.cats_cols = original_dataset.select_dtypes(include=['object']).columns.tolist()
        if len(self.cats_cols) > 0:
            self.cat_encoder = OrdinalEncoder().fit(original_dataset[self.cats_cols])
            original_dataset[self.cats_cols] = self.cat_encoder.transform(original_dataset[self.cats_cols])

        self.original_dataset = original_dataset

        #BUG: If prepared splits is a dictionary, it cannot have two of the same model names
        if prepared_splits is not None:
            if isinstance(list(prepared_splits.values())[0],list):
                self.column_splits = prepared_splits
            else:
                self.column_splits = random_split_columns(original_dataset, {f'split{i}': split for i, split in enumerate(prepared_splits.values())}, random_state=random_state)
        else:
            match automated_splits:
                case 'correlated':
                    self.column_splits = correlated_distribute_columns(original_dataset, num_partitions=num_automated_splits, min_size=2, random_state=random_state)
                case 'random':
                    self.column_splits = random_split_columns(original_dataset, {f'split{i}': 1 for i in range(num_automated_splits)}, random_state=random_state)
                case _:
                    raise ValueError(f"Unknown automated_splits option: {automated_splits}. Use 'correlated' or 'random'.")
                
        self.encoded_dataset_dict = self._setup_column_splits(self.column_splits)

        pass

    def _setup_column_splits(self, prepared_splits: Dict[str, List[str]]) -> Dict[str, DataFrame]:
        """ Setup the column splits based on the prepared splits.

        Args:
            prepared_splits (Dict[str, List[str]]): A dictionary where keys are split names and values are lists of column names.

        Returns:
            Dict[str, DataFrame]: A dictionary where keys are split names and values are the corresponding DataFrames.
        """
        return {str(split): self.original_dataset[columns] for split, columns in prepared_splits.items()}

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
        generated_dataset = generated_dataset[list(self.original_dataset.columns)]

        if len(self.cats_cols) > 0:
            generated_dataset[self.cats_cols] = self.cat_encoder.inverse_transform(generated_dataset[self.cats_cols])
        return generated_dataset


if __name__ == "__main__":
    import doctest
    doctest.testmod()