# Description: Class for learning and validating joints between records
# Date: 14-11-2024
# Version: 0.1
# Author : Anton D. Lautrup

import pandas as pd
from sklearn.model_selection import StratifiedKFold

class JoinValidator:

    def __init__(self, 
                 classifier_model_base,
                 verbose = True,
                 ):
        
        # check that the classifier model is a valid model
        if not hasattr(self.classifier_model_base, 'fit'):
            raise ValueError('The classifier model must have a fit method')
        if not hasattr(self.classifier_model_base, 'predict'):
            raise ValueError('The classifier model must have a predict method')

        self.classifier_model = classifier_model_base

        self.verbose = verbose
        pass
    
    def _setup_training_data(self,
                             dictionary_of_data_chunks,
                             num_batches_of_bad_joins = 2,
                             random_state = 42,
                                ):
        """Prepare the training data for the classifier"""

        correct_joins, incorrect_joins = [], []
        for dataset_chunk in dictionary_of_data_chunks:
            correct_joins.append(dataset_chunk)
            incorrect_joins.append(dataset_chunk.sample(frac=num_batches_of_bad_joins, random_state=random_state, replace=True))
        
        correct_joins = pd.concat(correct_joins).reset_index(drop=True)
        incorrect_joins = pd.concat(incorrect_joins).reset_index(drop=True)

        train_labels = [1]*len(correct_joins)+[0]*len(incorrect_joins)

        df_join_train = pd.concat([correct_joins, incorrect_joins], axis=0).reset_index(drop=True)

        return df_join_train, train_labels
                             

    def fit_classifier(self,
                       dictionary_of_data_chunks,
                       number_of_stratified_k_fold = 5,
                       num_batches_of_bad_joins = 2,
                       random_state = 42,
                       ):
        """Crossvalidate and train the classifier"""
        
        df_join_train, train_labels = self._setup_training_data(dictionary_of_data_chunks, num_batches_of_bad_joins, random_state)

        skf = StratifiedKFold(n_splits=number_of_stratified_k_fold, shuffle=True, random_state=random_state)
        
        accuracies = []
        for train_index, test_index in skf.split(df_join_train, train_labels):
            X_train, X_test = df_join_train.iloc[train_index], df_join_train.iloc[test_index]
            y_train, y_test = train_labels[train_index], train_labels[test_index]
            
            self.classifier_model_base.fit(X_train, y_train)
            y_pred = self.classifier_model.predict(X_test)
            accuracies.append((y_pred.round() == y_test).mean())

        if self.verbose:
            print(f'Cross-validated accuracies: {accuracies}')
            print(f'Mean accuracy: {sum(accuracies) / len(accuracies)}')
        
        self.classifier_model = self.classifier_model.fit(df_join_train, train_labels)

        if self.verbose: print('Final model trained!')
        pass
    
    def validate(self, df_attempt):
        pred = self.classifier_model.predict(df_attempt)
        if self.verbose: print(f'Predicted good joins: {(pred==1).mean()}')
        return df_attempt.loc[pred==1]