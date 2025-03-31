import copy
import numpy as np

from pandas import DataFrame

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

class ResNet(torch.nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, inputs):
        return self.module(inputs) + inputs

class Custom_MLP_model(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim, output_dim=2, learning_rate=0.001, epochs=100, weight_decay=0.01, betas=(0.9, 0.999), dropout_rate=0.3, min_delta=0.001, patience=10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.betas = betas
        self.min_delta = min_delta
        self.patience = patience
        self.epoch_early_stop = 20
        self.dropout_rate = dropout_rate
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay, betas=self.betas)
        self.scaler = StandardScaler()
    
    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(self.dropout_rate),
            ResNet(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(self.hidden_dim)
                )
            ),
            ResNet(
                nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(self.hidden_dim)
                )
            ),
            nn.Linear(self.hidden_dim, self.output_dim)
        )
    
    def fit(self, X: DataFrame, y: list):
        X = self.scaler.fit_transform(X.values)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        best_acc = - np.inf
        best_weights = None
 
        loss_history = []
        epochs_no_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)

            outputs = torch.softmax(self.model(X_tensor), dim=1)
            _, predicted = torch.max(outputs, 1)
            acc = (predicted == y_tensor).float().mean()
            acc = float(acc)
            loss.backward()
            self.optimizer.step()

            loss_history.append(loss.detach().numpy())

            if acc > best_acc:
                best_acc = acc
                best_weights = copy.deepcopy(self.model.state_dict())
            
            # Early stopping check
            if len(loss_history) > self.epoch_early_stop and abs(np.diff(loss_history[-10:])).mean() < self.min_delta:
                epochs_no_improve += 1
            else:
                epochs_no_improve = 0
                
            if epoch > self.epoch_early_stop and epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}")
                print(f"Accuracy: {acc}")
                break

        # Load best weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        self.classes_ = y_tensor.unique().numpy()
    
    def predict(self, X: DataFrame):
        self.model.eval()
        X = self.scaler.transform(X.values)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = torch.softmax(self.model(X_tensor), dim=1)
        _, predicted = torch.max(outputs, 1)
        return predicted.numpy()
    
    def predict_proba(self, X: DataFrame):
        self.model.eval()
        X = self.scaler.transform(X.values)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        outputs = torch.softmax(self.model(X_tensor), dim=1)
        return outputs.detach().numpy()
