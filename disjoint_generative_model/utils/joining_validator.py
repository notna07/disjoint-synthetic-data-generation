# Description: Class for learning and validating joints between records
# Date: 14-11-2024
# Author : Anton D. Lautrup

import numpy as np
import pandas as pd

from typing import Dict
from pandas import DataFrame

from sklearn.model_selection import StratifiedKFold

def _setup_training_data(dictionary_of_data_chunks: Dict[str, DataFrame],
                         num_batches_of_bad_joins: int = 2,
                         random_state: int =  None,
                         ) -> tuple[DataFrame, list]:
    """Prepare the training data for the classifier
    
    Args:
        dictionary_of_data_chunks (Dict[str, DataFrame]): A dictionary of dataframes.
        num_batches_of_bad_joins (int): The number of bad joins to generate for each good join.
        random_state (int): The random state to use if reproducibility.

    Returns:
        tuple: A tuple of the training data and the labels.

    Example:
        >>> import pandas as pd
        >>> df_original = pd.read_csv('tests/dummy_train.csv')
        >>> dict_dfs = {'df1': df_original[['class','age','sex']], 'df2': df_original[['height','weight','income','education']]}
        >>> df_join_train, train_labels = _setup_training_data(dict_dfs, num_batches_of_bad_joins=2, random_state=42)
        >>> isinstance(df_join_train, pd.DataFrame)
        True
        >>> isinstance(train_labels, list)
        True
    """

    correct_joins, incorrect_joins = [], []
    for _, dataset_chunk in dictionary_of_data_chunks.items():
        correct_joins.append(dataset_chunk)
        incorrect_joins.append(dataset_chunk.sample(frac=num_batches_of_bad_joins, random_state=random_state, replace=True).reset_index(drop=True))

    correct_joins = pd.concat(correct_joins, axis=1, ignore_index=True)
    incorrect_joins = pd.concat(incorrect_joins, axis=1, ignore_index=True)

    train_labels = [1]*len(correct_joins)+[0]*len(incorrect_joins)

    df_join_train = pd.concat([correct_joins, incorrect_joins], axis=0).reset_index(drop=True)

    return df_join_train, train_labels


class JoiningValidator:
    def __init__(self, 
                 classifier_model_base: object,
                 verbose = True,
                 ):
        
        # check that the classifier model is a valid model
        if not hasattr(classifier_model_base, 'fit'):
            raise ValueError('The classifier model must have a fit method')
        if not hasattr(classifier_model_base, 'predict'):
            raise ValueError('The classifier model must have a predict method')

        self.classifier_model = classifier_model_base

        self.verbose = verbose
        pass
                             
    def fit_classifier(self,
                       dictionary_of_data_chunks: Dict[str, DataFrame],
                       number_of_stratified_k_fold: int = 5,
                       num_batches_of_bad_joins: int = 2,
                       random_state: int = None,
                       ) -> None:
        """ Perform cross-validation training and train the final model.

        Args:
            dictionary_of_data_chunks (Dict[str, DataFrame]): A dictionary of dataframes.
            number_of_stratified_k_fold (int): The number of stratified k-folds to use.
            num_batches_of_bad_joins (int): The number of bad joins to generate for each good join.
            random_state (int): The random state to use.

        Example:
            >>> from sklearn.linear_model import LogisticRegression
            >>> import pandas as pd
            >>> dict_dfs = {'df1': pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}), 'df2': pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})}
            >>> validator = JoiningValidator(LogisticRegression())
            >>> validator.fit_classifier(dict_dfs, number_of_stratified_k_fold=2, num_batches_of_bad_joins=2, random_state=42)
            Cross-validated accuracies: [0.6, 0.25]
            Mean accuracy: 0.425
            Final model trained!
        """
        
        df_join_train, train_labels = _setup_training_data(dictionary_of_data_chunks, num_batches_of_bad_joins, random_state)
        train_labels = np.array(train_labels)

        skf = StratifiedKFold(n_splits=number_of_stratified_k_fold, shuffle=True, random_state=random_state)
        
        accuracies = []
        for train_index, test_index in skf.split(df_join_train, train_labels):
            X_train, X_test = df_join_train.iloc[train_index], df_join_train.iloc[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            
            self.classifier_model.fit(X_train, y_train)
            y_pred = self.classifier_model.predict(X_test)
            accuracies.append((y_pred.round() == y_test).mean())

        if self.verbose:
            print(f'Cross-validated accuracies: {accuracies}')
            print(f'Mean accuracy: {sum(accuracies) / len(accuracies)}')
        
        self.classifier_model = self.classifier_model.fit(df_join_train, train_labels)

        if self.verbose: print('Final model trained!')
        pass
    
    def validate(self, query_data: DataFrame) -> DataFrame:
        """ Validate the given DataFrame using the trained model.

        Args:
            df_attempt (DataFrame): The DataFrame to validate.

        Returns:
            DataFrame: The rows of df_attempt that are predicted to be good joins.

        Example:
            >>> from sklearn.linear_model import LogisticRegression
            >>> import numpy as np
            >>> np.random.seed(9)
            >>> df_train = pd.DataFrame(np.random.rand(100, 5))
            >>> labels = pd.Series(np.random.randint(0, 2, size=100))
            >>> validator = JoiningValidator(LogisticRegression().fit(df_train, labels))
            >>> query_data = pd.DataFrame(np.random.rand(10, 5))
            >>> result = validator.validate(query_data)
            Predicted good joins fraction: 0.9
            >>> isinstance(result, pd.DataFrame)
            True
        """
        pred = self.classifier_model.predict(query_data)
        if self.verbose: print(f'Predicted good joins fraction: {(pred==1).mean()}')
        return query_data.loc[pred==1]

if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()