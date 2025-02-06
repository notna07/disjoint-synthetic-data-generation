# Description: Class for learning and validating joints between records
# Date: 14-11-2024
# Author : Anton D. Lautrup

import numpy as np
import pandas as pd

import copy

from typing import Dict, Literal
from pandas import DataFrame

from sklearn.metrics import f1_score, brier_score_loss

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from pyod.models.iforest import IForest

def _setup_training_data(dictionary_of_data_chunks: Dict[str, DataFrame],
                         num_batches_of_bad_joins: int = 2,
                         random_state: int =  None,
                         negative_class: int = 0,
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
        correct_joins.append(dataset_chunk.sample(frac=num_batches_of_bad_joins, random_state=0, replace=True).reset_index(drop=True))
        incorrect_joins.append(dataset_chunk.sample(frac=num_batches_of_bad_joins, random_state=random_state, replace=True).reset_index(drop=True))
        if random_state is not None: random_state += 1

    correct_joins = pd.concat(correct_joins, axis=1, ignore_index=True)
    incorrect_joins = pd.concat(incorrect_joins, axis=1, ignore_index=True)

    train_labels = [1]*len(correct_joins)+[negative_class]*len(incorrect_joins)

    df_join_train = pd.concat([correct_joins, incorrect_joins], axis=0).reset_index(drop=True)

    return df_join_train, train_labels


class JoiningValidator:
    """Class for learning and validating joints between records using a classifier model.

    Attributes:
        classifier_model (object): The classifier model to use.
        threshold (float): The threshold for the classifier.
        verbose (bool): Whether to print information.
    
    Methods:
        fit_classifier: Perform cross-validation training and train the final model.
        validate: Validate the given DataFrame using the trained model.
    """
    def __init__(self, 
                 classifier_model_base: object = RandomForestClassifier(n_estimators=100, max_depth=5),
                 model_parameter_grid: dict = None,
                 calibration_method: Literal['isotonic', 'sigmoid'] | None = 'isotonic',
                 save_proba: bool = False,
                 verbose = True,
                 ):
        
        # check that the classifier model is a valid model
        if not hasattr(classifier_model_base, 'fit'):
            raise ValueError('The classifier model must have a fit method')
        if not hasattr(classifier_model_base, 'predict'):
            raise ValueError('The classifier model must have a predict method')
        if not hasattr(classifier_model_base, 'predict_proba'):
            raise ValueError('The classifier model must have a predict_proba method')

        self.model = classifier_model_base
        self.params = model_parameter_grid
        self.calibration_method = calibration_method
        self.threshold = 0.5
        self.auto_threshold_percentage = None
        self.save_proba = save_proba
        self.verbose = verbose
        pass

    def get_standard_behavior(self) -> Dict:
        """ Get the standard parameters for the strategy. """
        dict = {
            "patience": 3,
            "min_iter": 5,
            "max_iter": 100,
            "threshold": 'auto',
            "threshold_decay": 0, 
            'auto_threshold_percentage': 0.1
        }
        return dict

    def fit_classifier(self,
                       dictionary_of_data_chunks: Dict[str, DataFrame],
                       number_of_validation_folds: int = 2,
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

        base_model = copy.copy(self.model)

        if self.params is not None:
            if self.verbose: print("Validator: Grid search for hyperparameters")
            grid_search = GridSearchCV(estimator=base_model, param_grid=self.params, n_jobs=-1, cv=number_of_validation_folds, scoring='neg_brier_score')#"balanced_accuracy")#
            grid_result = grid_search.fit(df_join_train, train_labels)

            if self.verbose: print("Validator: Best Brier score %f using %s" % (grid_result.best_score_, grid_result.best_params_))
            estimator = grid_result.best_estimator_
        else:
            if self.verbose: print("Validator: No search parameters specified. Using default configuration.")
            estimator = base_model
            estimator.fit(df_join_train, train_labels)
        
        y_pred = estimator.predict(df_join_train)

        score_pre = brier_score_loss(train_labels, y_pred)
        if self.calibration_method is not None:
            calibrated_model = CalibratedClassifierCV(estimator, cv='prefit', method=self.calibration_method)
            calibrated_model.fit(df_join_train, train_labels)
            y_pred = calibrated_model.predict(df_join_train)
            score_post = brier_score_loss(train_labels, y_pred)

            if score_post < score_pre:
                if self.verbose: print(f"Validator: Calibration improved the model from {score_pre:.4f} to {score_post:.4f}")
                fitted_model = calibrated_model
            else:
                if self.verbose: print(f"Validator: Calibration did not improve the model. Using the original model.")
                fitted_model = estimator
        else:
            fitted_model = estimator

        self.model = fitted_model
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
        pred = self.model.predict_proba(query_data.values)[:,1]
        if self.threshold == "auto":
            self.threshold = sorted(pred, reverse=True)[int(self.auto_threshold_percentage*len(query_data))]
            print("Threshold auto-set to:", self.threshold)

            if self.save_proba:
                from .plots import plot_proba_hist
                plot_proba_hist(pred, save_dir='plots')

        pred = (pred >= self.threshold).astype(int)
        if self.verbose: print(f'Predicted good joins fraction: {(pred==1).mean()}')
        return query_data.loc[pred==1]

class OneClassValidator:
    """Class for learning and validating joints between records using a one-class classifier model.

    Attributes:
        one_class (object): One class classifier model to use.
        threshold (float): The threshold for the classifier.
        verbose (bool): Whether to print information.
    
    Methods:
        fit_classifier: Perform cross-validation training and train the final model.
        validate: Validate the given DataFrame using the trained model.
    """
    def __init__(self, 
                 one_class_model: object = OneClassSVM(),
                 verbose = True,
                 ):
        
        # check that the classifier model is a valid model
        if not hasattr(one_class_model, 'fit'):
            raise ValueError('The one-class model must have a fit method')
        if not hasattr(one_class_model, 'predict'):
            raise ValueError('The one-class model must have a predict method')
        if not hasattr(one_class_model, 'score_samples'):
            raise ValueError('The one-class model must have a score_samples method')

        self.model = one_class_model
        self.threshold = 0.5
        self.auto_threshold_percentage = None
        self.verbose = verbose
        pass

    def get_standard_behavior(self) -> Dict:
        """ Get the standard parameters for the strategy. """
        dict = {
            "patience": 5,
            "min_iter": 5,
            "max_iter": 50,
            "threshold":  "auto",
            "threshold_decay":  0.01,
            'auto_threshold_percentage': 0.1
        }
        return dict

    def fit_classifier(self,
                       dictionary_of_data_chunks: Dict[str, DataFrame],
                       number_of_k_fold: int = 5,
                       random_state: int = None,
                       ) -> None:
        """ Train the outlier detection model using the given data.

        Example:
            >>> import numpy as np
            >>> import pandas as pd
            >>> dict_dfs = {'df1': pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 4, 4]}), 'df2': pd.DataFrame({'C': [2, 4, 6, 8], 'D': [2, 4, 8, 8]})}
            >>> validator = OneClassValidator()
            >>> validator.fit_classifier(dict_dfs, number_of_k_fold=2, random_state=42)
            -etc-
            Final model trained!
        """

        df_join_train, train_labels = _setup_training_data(dictionary_of_data_chunks, 1, random_state, negative_class=-1)

        train_labels = np.array(train_labels)

        kf = KFold(n_splits=number_of_k_fold, shuffle=True, random_state=random_state)

        accuracies = []
        for train_index, test_index in kf.split(df_join_train[train_labels==1]):
            temp_oc = copy.copy(self.model)
            X_train, X_test_inliers = df_join_train.iloc[train_index], df_join_train.iloc[test_index]
            X_test_outliers = df_join_train[train_labels==-1].reset_index(drop=True).iloc[test_index]
            temp_oc.fit(X_train)

            X_test = pd.concat([X_test_inliers, X_test_outliers], axis=0)

            y_test = np.array([1]*len(X_test_inliers)+[-1]*len(X_test_outliers))
            y_pred = temp_oc.predict(X_test)

            score = f1_score(y_test, y_pred)
            accuracies.append(score)

        if self.verbose:
            print(f'F1-Score (Good Joins): {accuracies}')
            print(f'Mean F1: {sum(accuracies) / len(accuracies)}')

        self.model.fit(df_join_train[train_labels==1])

        if self.verbose: print('Final model trained!')
        pass

    def validate(self, query_data: DataFrame) -> DataFrame:
        """ Validate the given DataFrame using the trained model.

        Args:
            df_attempt (DataFrame): The DataFrame to validate.

        Returns:
            DataFrame: The rows of df_attempt that are predicted to be good joins.

        Example:
            >>> import numpy as np
            >>> import pandas as pd
            >>> np.random.seed(9)
            >>> df_train = pd.DataFrame(np.random.rand(100, 5))
            >>> validator = OneClassValidator(OneClassSVM().fit(df_train))
            >>> query_data = pd.DataFrame(np.random.rand(10, 5))
            >>> result = validator.validate(query_data)
            Predicted good joins fraction: 1.0
            >>> isinstance(result, pd.DataFrame)
            True
        """
        pred = 0.5+self.model.decision_function(query_data.values)
        if self.threshold == "auto":
            self.threshold = sorted(pred, reverse=True)[int(self.auto_threshold_percentage*len(query_data))]
            print("Threshold auto-set to:", self.threshold)

        pred = (pred >= self.threshold).astype(int)
        if self.verbose: print(f'Predicted good joins fraction: {(pred==1).mean()}')
        return query_data.loc[pred==1]


class OutlierValidator:
    """Class for learning and validating joints between records using an outlier detection model.

    Attributes:
        outlier_detector (object): Outlier detection model to use.
        threshold (float): The threshold for the outlier detection model. Since PyOD is used, threshold decay is set to be negative.
        flex (float): The flexiblity of the outlier detection model in range [0, 1]. Higher values indicate a more strict behavior.
        verbose (bool): Whether to print information.
    
    Methods:
        fit_classifier: Perform cross-validation training and train the final model.
        validate: Validate the given DataFrame using the trained model.
    """
    def __init__(self, 
                 outlier_detector_model: object = IForest(),
                 verbose = True,
                 ):
        
        # check that the classifier model is a valid model
        if not hasattr(outlier_detector_model, 'fit'):
            raise ValueError('The outlier detection model must have a fit method')
        if not hasattr(outlier_detector_model, 'predict'):
            raise ValueError('The outlier detection model must have a predict method')
        if not hasattr(outlier_detector_model, 'decision_function'):
            raise ValueError('The outlier detection model must have a decision_function method')

        self.model = outlier_detector_model
        self.threshold = 0.5
        self.auto_threshold_percentage = None
        self.flex = 0.5
        self.model.contamination = self.flex / 2
        self.verbose = verbose
        pass

    def get_standard_behavior(self) -> Dict:
        """ Get the standard parameters for the strategy. """
        dict = {
            "patience": 5,
            "min_iter": 5,
            "max_iter": 50,
            "threshold":  "auto",
            "threshold_decay":  0.01,
            'auto_threshold_percentage': 0.1
        }
        return dict

    def fit_classifier(self,
                       dictionary_of_data_chunks: Dict[str, DataFrame],
                       number_of_k_fold: int = 5,
                       random_state: int = None,
                       ) -> None:
        """ Train the outlier detection model using the given data.

        Example:
            >>> import numpy as np
            >>> import pandas as pd
            >>> dict_dfs = {'df1': pd.DataFrame({'A': [1, 2, 3, 4], 'B': [1, 2, 4, 4]}), 'df2': pd.DataFrame({'C': [2, 4, 6, 8], 'D': [2, 4, 8, 8]})}
            >>> validator = OutlierValidator()
            >>> validator.fit_classifier(dict_dfs, number_of_k_fold=2, random_state=42)
            Final model trained!
        """

        df_join_train, train_labels = _setup_training_data(dictionary_of_data_chunks, 2, random_state, negative_class=-1)  
        train_labels = np.array(train_labels)
        df_join_train_inlier = df_join_train[train_labels==1]
        df_join_train_outlier = df_join_train[train_labels==-1].sample(len(df_join_train[train_labels==-1]) // 6)        

        accuracies = []
        if len(df_join_train_outlier) != 1:
            for i in range(number_of_k_fold):
                temp_od = self.model
                inlier_train, inlier_test = train_test_split(df_join_train_inlier, test_size=1/number_of_k_fold, random_state=random_state if random_state is None else random_state+i)
                outlier_train, outlier_test = train_test_split(df_join_train_outlier, test_size=1/number_of_k_fold, random_state=random_state if random_state is None else random_state+i)
                combined_train = pd.concat([inlier_train, outlier_train], ignore_index=True)
                inlier_train_labels = pd.DataFrame([0] * len(inlier_train), columns=['label'])
                outlier_train_labels = pd.DataFrame([1] * len(outlier_train), columns=['label'])
                combined_train_labels = pd.concat([inlier_train_labels, outlier_train_labels], ignore_index=True)
                combined_test = pd.concat([inlier_test, outlier_test], ignore_index=True)
                inlier_test_labels = pd.DataFrame([0] * len(inlier_test), columns=['label'])
                outlier_test_labels = pd.DataFrame([1] * len(outlier_test), columns=['label'])
                combined_test_labels = pd.concat([inlier_test_labels, outlier_test_labels], ignore_index=True)
                temp_od.fit(combined_train)
                y_pred = -temp_od.predict(combined_test) #multiply by -1 to map the scores from [inlier, outlier] => [0, 1] to [outlier, inlier] => [-1, 0] for consistency with different framework behaviors. 
                score = f1_score(combined_test_labels, y_pred, average='macro')
                accuracies.append(score)

            if self.verbose:
                print(f'F1-Score (Good Joins): {accuracies}')
                print(f'Mean F1: {sum(accuracies) / len(accuracies)}')

        self.model = self.model.fit(pd.concat([df_join_train_inlier, df_join_train_outlier], ignore_index=True))

        if self.verbose: print('Final model trained!')
        pass

    def validate(self, query_data: DataFrame) -> DataFrame:
        """ Validate the given DataFrame using the trained model.

        Args:
            df_attempt (DataFrame): The DataFrame to validate.

        Returns:
            DataFrame: The rows of df_attempt that are predicted to be good joins.

        Example:
            >>> import numpy as np
            >>> import pandas as pd
            >>> np.random.seed(9)
            >>> df_train = pd.DataFrame(np.random.rand(100, 5))
            >>> validator = OutlierValidator(IForest().fit(df_train))
            >>> query_data = pd.DataFrame(np.random.rand(10, 5))
            >>> result = validator.validate(query_data)
            Predicted good joins fraction: 1.0
            >>> isinstance(result, pd.DataFrame)
            True
        """
        pred = -self.model.decision_function(query_data.values)+1 #multiply by -1 to map the scores from [inlier, outlier] => [0, 1] to [outlier, inlier] => [0, 1] for consistency with different framework behaviors. 
        if self.threshold == "auto":
            self.threshold = sorted(pred, reverse=False)[int(self.auto_threshold_percentage*len(query_data))]
            print("Threshold auto-set to:", self.threshold)

        pred = (pred >= self.threshold).astype(int)
        if self.verbose: print(f'Predicted good joins fraction: {(pred==1).mean()}')
        return query_data.loc[pred==1]


if __name__ == "__main__":
    import doctest
    doctest.ELLIPSIS_MARKER = '-etc-'
    doctest.testmod()
