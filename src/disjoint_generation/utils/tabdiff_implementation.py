# Description: Self-contained TabDiff implementation for portability.
#   All neural-network, diffusion, data-processing, and training logic is
#   inlined here so this single file (plus the adapter in generative_model_adapters.py) 
#   can be dropped into any repository to allow access to tabdiff.
#
# Sources (copied / adapted from the TabDiff repository, i.e., https://github.com/MinkaiXu/TabDiff):
#   tabdiff/modules/transformer.py
#   tabdiff/modules/main_modules.py
#   tabdiff/models/noise_schedule.py
#   tabdiff/models/unified_ctime_diffusion.py
#   tabdiff/trainer.py          (training loop + sample_synthetic)
#   utils_train.py              (TabDiffDataset, preprocess, make_dataset)
#   src/data.py                 (Dataset, Transformations, transform_dataset, …)
#   src/util.py                 (TaskType, get_categories, load_config)
#
# External dependencies (must be pip-installable in the target repo):
#   torch, numpy, pandas, scikit-learn, scipy, tqdm, tomli, tomli_w,
#   category_encoders, etc.

# ============================================================================
# License information
# ============================================================================

# Copyright 2024 Minkai Xu

# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the “Software”), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is furnished 
# to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
# IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# ============================================================================
# Standard + third-party imports
# ============================================================================

import abc
import glob
import json
import math
import os
import time
from collections import Counter
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from enum import Enum
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init
import tomli
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

# ============================================================================
# TaskType enum (from src/util.py)
# ============================================================================

class TaskType(Enum):
    BINCLASS = 'binclass'
    MULTICLASS = 'multiclass'
    REGRESSION = 'regression'

    def __str__(self) -> str:
        return self.value


# ============================================================================
# Config helpers (from src/util.py)
# ============================================================================

_CONFIG_NONE = '__none__'

def _replace_in_config(data, condition, value):
    if isinstance(data, dict):
        return {k: _replace_in_config(v, condition, value) for k, v in data.items()}
    elif isinstance(data, list):
        return [_replace_in_config(y, condition, value) for y in data]
    else:
        return value if condition(data) else data

def load_config(path: Union[Path, str]) -> Any:
    with open(path, 'rb') as f:
        raw = tomli.load(f)
    return _replace_in_config(raw, lambda x: x == _CONFIG_NONE, None)

def get_categories(X_train_cat) -> Optional[List[int]]:
    if X_train_cat is None:
        return None
    return [len(set(X_train_cat[:, i])) for i in range(X_train_cat.shape[1])]


# ============================================================================
# Dataset dataclass (from src/data.py)
# ============================================================================

ArrayDict = Dict[str, np.ndarray]
Normalization = Literal['standard', 'quantile', 'minmax']
NumNanPolicy = Literal['drop-rows', 'mean']
CatNanPolicy = Literal['most_frequent']
CatEncoding = Literal['one-hot', 'counter']
YPolicy = Literal['default']
DEQUANT_DIST = Literal['uniform', 'beta', 'round', 'none']

CAT_MISSING_VALUE = 'nan'
CAT_RARE_VALUE = '__rare__'


@dataclass(frozen=False)
class Dataset:
    X_num: Optional[ArrayDict]
    X_cat: Optional[ArrayDict]
    y: ArrayDict
    int_col_idx_wrt_num: list
    y_info: Dict[str, Any]
    task_type: TaskType
    n_classes: Optional[int]

    # populated by transform_dataset
    num_transform: Any = None
    int_transform: Any = None
    cat_transform: Any = None


def get_category_sizes(X: Union[torch.Tensor, np.ndarray]) -> List[int]:
    XT = X.T.cpu().tolist() if isinstance(X, torch.Tensor) else X.T.tolist()
    return [len(set(x)) for x in XT]


# ============================================================================
# Data transforms (from src/data.py)
# ============================================================================

class StandardScaler1d(StandardScaler):
    def partial_fit(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().partial_fit(X[:, None], *args, **kwargs)

    def transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().transform(X[:, None], *args, **kwargs).squeeze(1)

    def inverse_transform(self, X, *args, **kwargs):
        assert X.ndim == 1
        return super().inverse_transform(X[:, None], *args, **kwargs).squeeze(1)


def _normalize(X: ArrayDict, normalization: Normalization, seed: Optional[int], return_normalizer=False):
    X_train = X['train']
    if normalization == 'standard':
        normalizer = sklearn.preprocessing.StandardScaler()
    elif normalization == 'minmax':
        normalizer = sklearn.preprocessing.MinMaxScaler()
    elif normalization == 'quantile':
        normalizer = sklearn.preprocessing.QuantileTransformer(
            output_distribution='normal',
            n_quantiles=max(min(X_train.shape[0] // 30, 1000), 10),
            subsample=int(1e9),
            random_state=seed,
        )
    else:
        raise ValueError(f'Unknown normalization: {normalization}')
    normalizer.fit(X_train)
    if return_normalizer:
        return {k: normalizer.transform(v) for k, v in X.items()}, normalizer
    return {k: normalizer.transform(v) for k, v in X.items()}


class _Dequantizer:
    def __init__(self, dequant_dist, int_col_idx_wrt_num, int_dequant_factor):
        self.dequant_dist = dequant_dist
        self.int_col_idx_wrt_num = int_col_idx_wrt_num
        self.int_dequant_factor = int_dequant_factor

    def transform(self, X):
        X = X.copy()
        X_int = X[:, self.int_col_idx_wrt_num]
        if self.dequant_dist == 'uniform':
            X[:, self.int_col_idx_wrt_num] = X_int + np.random.uniform(size=X_int.shape) * self.int_dequant_factor
        elif self.dequant_dist == 'beta':
            X[:, self.int_col_idx_wrt_num] = X_int + np.random.beta(self.int_dequant_factor, self.int_dequant_factor, size=X_int.shape) - 0.5
        return X

    def inverse_transform(self, X):
        X = X.copy()
        X_int = X[:, self.int_col_idx_wrt_num]
        if self.dequant_dist == 'uniform':
            X[:, self.int_col_idx_wrt_num] = np.floor(X_int)
        elif self.dequant_dist in ('beta', 'round'):
            X[:, self.int_col_idx_wrt_num] = np.rint(X_int)
        return X


def _num_process_nans(dataset: Dataset, policy: Optional[NumNanPolicy]) -> Dataset:
    assert dataset.X_num is not None
    nan_masks = {k: np.isnan(v) for k, v in dataset.X_num.items()}
    if not any(x.any() for x in nan_masks.values()):
        return dataset
    assert policy is not None
    if policy == 'mean':
        new_values = np.nanmean(dataset.X_num['train'], axis=0)
        X_num = deepcopy(dataset.X_num)
        for k, v in X_num.items():
            num_nan_indices = np.where(nan_masks[k])
            v[num_nan_indices] = np.take(new_values, num_nan_indices[1])
        dataset = replace(dataset, X_num=X_num)
    else:
        raise ValueError(f'Unknown NumNanPolicy: {policy}')
    return dataset


def _cat_process_nans(X: ArrayDict, policy: Optional[CatNanPolicy]) -> ArrayDict:
    nan_masks = {k: v == CAT_MISSING_VALUE for k, v in X.items()}
    if any(x.any() for x in nan_masks.values()):
        if policy == 'most_frequent':
            imputer = SimpleImputer(missing_values=CAT_MISSING_VALUE, strategy=policy)
            imputer.fit(X['train'])
            return {k: cast(np.ndarray, imputer.transform(v)) for k, v in X.items()}
        elif policy is None:
            return X
        else:
            raise ValueError(f'Unknown CatNanPolicy: {policy}')
    return X


def _cat_encode(X: ArrayDict, encoding, y_train, seed, return_encoder=False):
    if encoding is None:
        unknown_value = np.iinfo('int64').max - 3
        oe = sklearn.preprocessing.OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=unknown_value,
            dtype='int64',
        ).fit(X['train'])
        encoder = make_pipeline(oe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        max_values = X['train'].max(axis=0)
        for part in X.keys():
            if part == 'train':
                continue
            for col in range(X[part].shape[1]):
                X[part][X[part][:, col] == unknown_value, col] = max_values[col] + 1
        if return_encoder:
            return X, False, encoder
        return X, False
    elif encoding == 'one-hot':
        ohe = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)
        encoder = make_pipeline(ohe)
        encoder.fit(X['train'])
        X = {k: encoder.transform(v) for k, v in X.items()}
        if return_encoder:
            return X, True, encoder
        return X, True
    else:
        raise ValueError(f'Unknown CatEncoding: {encoding}')


def _build_target(y: ArrayDict, policy, task_type: TaskType):
    info: Dict[str, Any] = {'policy': policy}
    if policy == 'default':
        if task_type == TaskType.REGRESSION:
            mean, std = float(y['train'].mean()), float(y['train'].std())
            y = {k: (v - mean) / std for k, v in y.items()}
            info['mean'] = mean
            info['std'] = std
    return y, info


@dataclass(frozen=True)
class Transformations:
    seed: int = 0
    normalization: Optional[Normalization] = None
    num_nan_policy: Optional[NumNanPolicy] = None
    cat_nan_policy: Optional[CatNanPolicy] = None
    cat_min_frequency: Optional[float] = None
    cat_encoding: Optional[CatEncoding] = None
    y_policy: Optional[YPolicy] = 'default'
    dequant_dist: Optional[DEQUANT_DIST] = None
    int_dequant_factor: Optional[float] = 0.0


def transform_dataset(dataset: Dataset, transformations: Transformations, cache_dir=None) -> Dataset:
    if dataset.X_num is not None:
        dataset = _num_process_nans(dataset, transformations.num_nan_policy)

    num_transform = None
    int_transform = None
    cat_transform = None
    X_num = dataset.X_num

    int_col_idx_wrt_num = dataset.int_col_idx_wrt_num
    if X_num is not None and int_col_idx_wrt_num and transformations.dequant_dist is not None:
        int_transform = _Dequantizer(transformations.dequant_dist, int_col_idx_wrt_num, transformations.int_dequant_factor)
        X_num = {k: int_transform.transform(v) for k, v in X_num.items()}

    if X_num is not None and transformations.normalization is not None:
        has_num = all(x.shape[1] > 0 for x in X_num.values())
        if has_num:
            X_num, num_transform = _normalize(X_num, transformations.normalization, transformations.seed, return_normalizer=True)

    if dataset.X_cat is None:
        X_cat = None
    else:
        has_cat = all(x.shape[1] > 0 for x in dataset.X_cat.values())
        if not has_cat:
            X_cat = dataset.X_cat
            for split in X_cat.keys():
                X_cat[split] = X_cat[split].astype(np.int64)
        else:
            X_cat = _cat_process_nans(dataset.X_cat, transformations.cat_nan_policy)
            X_cat, is_num, cat_transform = _cat_encode(X_cat, transformations.cat_encoding, dataset.y['train'], transformations.seed, return_encoder=True)
            if is_num:
                X_num = X_cat if X_num is None else {x: np.hstack([X_num[x], X_cat[x]]) for x in X_num}
                X_cat = None

    y, y_info = _build_target(dataset.y, transformations.y_policy, dataset.task_type)
    dataset = replace(dataset, X_num=X_num, X_cat=X_cat, y=y, y_info=y_info)
    dataset.num_transform = num_transform
    dataset.int_transform = int_transform
    dataset.cat_transform = cat_transform
    return dataset


def read_pure_data(path: str, split: str = 'train'):
    y = np.load(os.path.join(path, f'y_{split}.npy'), allow_pickle=True)
    X_num, X_cat = None, None
    if os.path.exists(os.path.join(path, f'X_num_{split}.npy')):
        X_num = np.load(os.path.join(path, f'X_num_{split}.npy'), allow_pickle=True)
    if os.path.exists(os.path.join(path, f'X_cat_{split}.npy')):
        X_cat = np.load(os.path.join(path, f'X_cat_{split}.npy'), allow_pickle=True)
    return X_num, X_cat, y


# ============================================================================
# Dataset builder (from utils_train.py: make_dataset + preprocess)
# ============================================================================

def _concat_y_to_X(X, y):
    if X is None:
        return y.reshape(-1, 1)
    return np.concatenate([y.reshape(-1, 1), X], axis=1)


def _make_dataset(data_path: str, T: Transformations, task_type: str, y_only: bool = False) -> Dataset:
    task = TaskType(task_type)
    X_cat: Optional[Dict] = {} if os.path.exists(os.path.join(data_path, 'X_cat_train.npy')) else None
    X_num: Optional[Dict] = {} if os.path.exists(os.path.join(data_path, 'X_num_train.npy')) else None
    y: Dict = {} if os.path.exists(os.path.join(data_path, 'y_train.npy')) else None

    for split in ['train', 'test']:
        X_num_t, X_cat_t, y_t = read_pure_data(data_path, split)
        if y_only:
            X_num_t = X_num_t[:, :0] if X_num_t is not None else X_num_t
            X_cat_t = X_cat_t[:, :0] if X_cat_t is not None else X_cat_t
        if X_num is not None:
            if task == TaskType.REGRESSION and not y_only:
                X_num[split] = _concat_y_to_X(X_num_t, y_t)
            else:
                X_num[split] = X_num_t
        if X_cat is not None:
            if task in (TaskType.BINCLASS, TaskType.MULTICLASS) and not y_only:
                X_cat[split] = _concat_y_to_X(X_cat_t, y_t)
            else:
                X_cat[split] = X_cat_t
        if y is not None:
            y[split] = y_t

    with open(os.path.join(data_path, 'info.json')) as fh:
        info_json = json.load(fh)
    int_col_idx_wrt_num = info_json.get('int_col_idx_wrt_num', [])
    if y_only:
        int_col_idx_wrt_num = []

    n_classes = info_json.get('n_classes')

    D = Dataset(X_num, X_cat, y, int_col_idx_wrt_num, {}, task, n_classes)
    return transform_dataset(D, T, None)


def _preprocess(data_path: str, y_only: bool, dequant_dist: str, int_dequant_factor: float, task_type: str, inverse: bool = False):
    T = Transformations(
        normalization='quantile',
        num_nan_policy='mean',
        cat_nan_policy=None,
        cat_min_frequency=None,
        cat_encoding=None,
        y_policy='default',
        dequant_dist=dequant_dist,
        int_dequant_factor=int_dequant_factor,
    )
    dataset = _make_dataset(data_path, T, task_type, y_only)

    X_num = dataset.X_num
    X_cat = dataset.X_cat
    X_train_num, X_test_num = X_num['train'], X_num['test']
    X_train_cat, X_test_cat = X_cat['train'], X_cat['test']

    categories = get_categories(X_train_cat)
    categories = np.array(categories) if categories is not None else np.array([])
    d_numerical = X_train_num.shape[1]

    if inverse:
        num_inverse = dataset.num_transform.inverse_transform if dataset.num_transform is not None else lambda x: x
        int_inverse = dataset.int_transform.inverse_transform if dataset.int_transform is not None else lambda x: x
        cat_inverse = dataset.cat_transform.inverse_transform if dataset.cat_transform is not None else lambda x: x
        return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical, num_inverse, int_inverse, cat_inverse
    return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical


class TabDiffDataset(TorchDataset):
    def __init__(self, data_dir: str, info: dict, isTrain: bool = True, y_only: bool = False, dequant_dist: str = 'none', int_dequant_factor: float = 0.0):
        X_num, X_cat, categories, d_numerical, num_inverse, int_inverse, cat_inverse = _preprocess(
            data_dir, y_only, dequant_dist, int_dequant_factor,
            task_type=info['task_type'], inverse=True
        )
        X_train_num = torch.tensor(X_num[0]).float()
        X_test_num = torch.tensor(X_num[1]).float()
        X_train_cat = torch.tensor(X_cat[0])
        X_test_cat = torch.tensor(X_cat[1])

        self.X = torch.cat((X_train_num, X_train_cat), dim=1) if isTrain else torch.cat((X_test_num, X_test_cat), dim=1)
        self.num_inverse = num_inverse
        self.int_inverse = int_inverse
        self.cat_inverse = cat_inverse
        self.d_numerical = d_numerical
        self.categories = categories
        self.info = info

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return self.X.shape[0]


# ============================================================================
# Noise Schedules (from tabdiff/models/noise_schedule.py)
# ============================================================================

class _Noise(abc.ABC, nn.Module):
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def total_noise(self, t): ...


class LogLinearNoise(_Noise):
    def __init__(self, eps_max=1e-3, eps_min=1e-5, **kwargs):
        super().__init__()
        self.eps_max = eps_max
        self.eps_min = eps_min

    def k(self):
        return torch.tensor(1)

    def rate_noise(self, t):
        return (1 - self.eps_max - self.eps_min) / (1 - ((1 - self.eps_max - self.eps_min) * t + self.eps_min))

    def total_noise(self, t):
        return -torch.log1p(-((1 - self.eps_max - self.eps_min) * t + self.eps_min))


class PowerMeanNoise(_Noise):
    def __init__(self, sigma_min=0.002, sigma_max=80, rho=7, **kwargs):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.raw_rho = rho

    def rho(self):
        return torch.tensor(self.raw_rho)

    def rate_noise(self, t):
        return None

    def total_noise(self, t):
        return (self.sigma_min ** (1 / self.rho()) + t * (self.sigma_max ** (1 / self.rho()) - self.sigma_min ** (1 / self.rho()))).pow(self.rho())

    def inverse_to_t(self, sigma):
        return (sigma.pow(1 / self.rho()) - self.sigma_min ** (1 / self.rho())) / (self.sigma_max ** (1 / self.rho()) - self.sigma_min ** (1 / self.rho()))


class PowerMeanNoise_PerColumn(nn.Module):
    def __init__(self, num_numerical, sigma_min=0.002, sigma_max=80, rho_init=1, rho_offset=2, **kwargs):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_numerical = num_numerical
        self.rho_offset = rho_offset
        self.rho_raw = nn.Parameter(torch.tensor([rho_init] * num_numerical, dtype=torch.float32))

    def rho(self):
        return F.softplus(self.rho_raw) + self.rho_offset

    def total_noise(self, t):
        rho = self.rho()
        return (self.sigma_min ** (1 / rho) + t * (self.sigma_max ** (1 / rho) - self.sigma_min ** (1 / rho))).pow(rho)

    def rate_noise(self, t):
        return None

    def inverse_to_t(self, sigma):
        rho = self.rho()
        return (sigma.pow(1 / rho) - self.sigma_min ** (1 / rho)) / (self.sigma_max ** (1 / rho) - self.sigma_min ** (1 / rho))


class LogLinearNoise_PerColumn(nn.Module):
    def __init__(self, num_categories, eps_max=1e-3, eps_min=1e-5, k_init=-6, k_offset=1, **kwargs):
        super().__init__()
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.num_categories = num_categories
        self.k_offset = k_offset
        self.k_raw = nn.Parameter(torch.tensor([k_init] * num_categories, dtype=torch.float32))

    def k(self):
        return F.softplus(self.k_raw) + self.k_offset

    def rate_noise(self, t, noise_fn=None):
        k = self.k()
        numerator = (1 - self.eps_max - self.eps_min) * k * t.pow(k - 1)
        denominator = 1 - ((1 - self.eps_max - self.eps_min) * t.pow(k) + self.eps_min)
        return numerator / denominator

    def total_noise(self, t, noise_fn=None):
        k = self.k()
        return -torch.log1p(-((1 - self.eps_max - self.eps_min) * t.pow(k) + self.eps_min))


# ============================================================================
# Neural network modules (from tabdiff/modules/transformer.py + main_modules.py)
# ============================================================================

class _MultiheadAttention(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0, initialization='kaiming'):
        if n_heads > 1:
            assert d % n_heads == 0
        super().__init__()
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_out = nn.Linear(d, d) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None
        for m in [self.W_q, self.W_k, self.W_v]:
            if initialization == 'xavier' and (n_heads > 1 or m is not self.W_v):
                nn_init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            nn_init.zeros_(m.bias)
        if self.W_out is not None:
            nn_init.zeros_(self.W_out.bias)

    def _reshape(self, x):
        bs, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return x.reshape(bs, n_tokens, self.n_heads, d_head).transpose(1, 2).reshape(bs * self.n_heads, n_tokens, d_head)

    def forward(self, x_q, x_kv, key_compression=None, value_compression=None):
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        bs = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]
        q, k = self._reshape(q), self._reshape(k)
        attention = F.softmax(q @ k.transpose(1, 2) / math.sqrt(d_head_key), dim=-1)
        if self.dropout is not None:
            attention = self.dropout(attention)
        x = attention @ self._reshape(v)
        x = x.reshape(bs, self.n_heads, n_q_tokens, d_head_value).transpose(1, 2).reshape(bs, n_q_tokens, self.n_heads * d_head_value)
        if self.W_out is not None:
            x = self.W_out(x)
        return x


class _Transformer(nn.Module):
    def __init__(self, n_layers, d_token, n_heads, d_out, d_ffn_factor, attention_dropout=0.0, ffn_dropout=0.0, residual_dropout=0.0, prenormalization=True, initialization='kaiming'):
        super().__init__()
        d_hidden = int(d_token * d_ffn_factor)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = nn.ModuleDict({
                'attention': _MultiheadAttention(d_token, n_heads, attention_dropout, initialization),
                'linear0': nn.Linear(d_token, d_hidden),
                'linear1': nn.Linear(d_hidden, d_token),
                'norm1': nn.LayerNorm(d_token),
            })
            if not prenormalization or i:
                layer['norm0'] = nn.LayerNorm(d_token)
            self.layers.append(layer)
        self.activation = nn.ReLU()
        self.prenormalization = prenormalization
        self.last_normalization = nn.LayerNorm(d_token) if prenormalization else None
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.head = nn.Linear(d_token, d_out)

    def _start_residual(self, x, layer, norm_idx):
        x_residual = x
        if self.prenormalization:
            norm_key = f'norm{norm_idx}'
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, x, x_residual, layer, norm_idx):
        if self.residual_dropout:
            x_residual = F.dropout(x_residual, self.residual_dropout, self.training)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f'norm{norm_idx}'](x)
        return x

    def forward(self, x):
        for layer in self.layers:
            x_residual = self._start_residual(x, layer, 0)
            x_residual = layer['attention'](x_residual, x_residual)
            x = self._end_residual(x, x_residual, layer, 0)
            x_residual = self._start_residual(x, layer, 1)
            x_residual = layer['linear0'](x_residual)
            x_residual = self.activation(x_residual)
            if self.ffn_dropout:
                x_residual = F.dropout(x_residual, self.ffn_dropout, self.training)
            x_residual = layer['linear1'](x_residual)
            x = self._end_residual(x, x_residual, layer, 1)
        return x


class _Tokenizer(nn.Module):
    def __init__(self, d_numerical, categories, d_token, bias):
        super().__init__()
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + list(categories[:-1])).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.cat_weight = nn.Parameter(Tensor(sum(categories), d_token))
            nn_init.kaiming_uniform_(self.cat_weight, a=math.sqrt(5))
        self.weight = nn.Parameter(Tensor(d_numerical + 1, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    @property
    def n_tokens(self):
        return len(self.weight) + (0 if self.category_offsets is None else len(self.category_offsets))

    def forward(self, x_num, x_cat):
        x_some = x_num if x_cat is None else x_cat
        x_num_in = torch.cat([torch.ones(len(x_some), 1, device=x_some.device)] + ([] if x_num is None else [x_num]), dim=1)
        x = self.weight[None] * x_num_in[:, :, None]
        if x_cat is not None:
            offsets_end = torch.cat([self.category_offsets[1:], torch.tensor([x_cat.shape[1]], device=x_cat.device)])
            for start, end in zip(self.category_offsets, offsets_end):
                if start < end:
                    x = torch.cat([x, x_cat[:, start:end].unsqueeze(1) @ self.cat_weight[start:end][None]], dim=1)
        if self.bias is not None:
            bias = torch.cat([torch.zeros(1, self.bias.shape[1], device=x.device), self.bias])
            x = x + bias[None]
        return x


class _Reconstructor(nn.Module):
    def __init__(self, d_numerical, categories, d_token):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.d_token = d_token
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        nn_init.xavier_uniform_(self.weight, gain=1 / math.sqrt(2))
        self.cat_recons = nn.ModuleList()
        for d in categories:
            recon = nn.Linear(d_token, d)
            nn_init.xavier_uniform_(recon.weight, gain=1 / math.sqrt(2))
            self.cat_recons.append(recon)

    def forward(self, h):
        h_num = h[:, :self.d_numerical]
        h_cat = h[:, self.d_numerical:]
        recon_x_num = torch.mul(h_num, self.weight.unsqueeze(0)).sum(-1)
        recon_x_cat = [recon(h_cat[:, i]) for i, recon in enumerate(self.cat_recons)]
        return recon_x_num, recon_x_cat


class _PositionalEmbedding(nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels // 2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        return torch.cat([x.cos(), x.sin()], dim=1)


class _MLPDiffusion(nn.Module):
    def __init__(self, d_in, dim_t=512, use_mlp=True):
        super().__init__()
        self.proj = nn.Linear(d_in, dim_t)
        self.mlp = nn.Sequential(
            nn.Linear(dim_t, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t * 2), nn.SiLU(),
            nn.Linear(dim_t * 2, dim_t), nn.SiLU(),
            nn.Linear(dim_t, d_in),
        ) if use_mlp else nn.Linear(dim_t, d_in)
        self.map_noise = _PositionalEmbedding(num_channels=dim_t)
        self.time_embed = nn.Sequential(nn.Linear(dim_t, dim_t), nn.SiLU(), nn.Linear(dim_t, dim_t))

    def forward(self, x, timesteps):
        emb = self.map_noise(timesteps)
        emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape)
        emb = self.time_embed(emb)
        x = self.proj(x) + emb
        return self.mlp(x)


class UniModMLP(nn.Module):
    def __init__(self, d_numerical, categories, num_layers, d_token, n_head=1, factor=4, bias=True, dim_t=512, use_mlp=True, **kwargs):
        super().__init__()
        self.d_numerical = d_numerical
        self.categories = categories
        self.tokenizer = _Tokenizer(d_numerical, categories, d_token, bias=bias)
        self.encoder = _Transformer(num_layers, d_token, n_head, d_token, factor)
        d_in = d_token * (d_numerical + len(categories))
        self.mlp = _MLPDiffusion(d_in, dim_t=dim_t, use_mlp=use_mlp)
        self.decoder = _Transformer(num_layers, d_token, n_head, d_token, factor)
        self.detokenizer = _Reconstructor(d_numerical, categories, d_token)
        self.model = nn.ModuleList([self.tokenizer, self.encoder, self.mlp, self.decoder, self.detokenizer])

    def forward(self, x_num, x_cat, timesteps):
        e = self.tokenizer(x_num, x_cat)
        decoder_input = e[:, 1:, :]
        y = self.encoder(decoder_input)
        pred_y = self.mlp(y.reshape(y.shape[0], -1), timesteps)
        pred_e = self.decoder(pred_y.reshape(*y.shape))
        x_num_pred, x_cat_pred = self.detokenizer(pred_e)
        x_cat_pred = torch.cat(x_cat_pred, dim=-1) if len(x_cat_pred) > 0 else torch.zeros_like(x_cat).to(x_num_pred.dtype)
        return x_num_pred, x_cat_pred


class _Precond(nn.Module):
    def __init__(self, denoise_fn, sigma_data=0.5, net_conditioning='sigma'):
        super().__init__()
        self.sigma_data = sigma_data
        self.net_conditioning = net_conditioning
        self.denoise_fn_F = denoise_fn

    def forward(self, x_num, x_cat, t, sigma):
        x_num = x_num.to(torch.float32)
        sigma = sigma.to(torch.float32)
        assert sigma.ndim == 2
        sigma_cond = (0.002 ** (1 / 7) + t * (80 ** (1 / 7) - 0.002 ** (1 / 7))).pow(7) if sigma.dim() > 1 else sigma
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma_cond.log() / 4
        x_in = c_in * x_num
        if self.net_conditioning == 'sigma':
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, c_noise.flatten())
        else:
            F_x, x_cat_pred = self.denoise_fn_F(x_in, x_cat, t)
        D_x = c_skip * x_num + c_out * F_x.to(torch.float32)
        return D_x, x_cat_pred


class Model(nn.Module):
    def __init__(self, denoise_fn, sigma_data=0.5, precond=False, net_conditioning='sigma', **kwargs):
        super().__init__()
        self.precond = precond
        self.denoise_fn_D = _Precond(denoise_fn, sigma_data=sigma_data, net_conditioning=net_conditioning) if precond else denoise_fn

    def forward(self, x_num, x_cat, t, sigma=None):
        if self.precond:
            return self.denoise_fn_D(x_num, x_cat, t, sigma)
        return self.denoise_fn_D(x_num, x_cat, t)


# ============================================================================
# Diffusion model (from tabdiff/models/unified_ctime_diffusion.py)
# ============================================================================

_S_churn = 1
_S_min = 0
_S_max = float('inf')
_S_noise = 1


class UnifiedCtimeDiffusion(nn.Module):
    def __init__(self, num_classes: np.ndarray, num_numerical_features: int, denoise_fn, y_only_model,
                 num_timesteps=1000, scheduler='power_mean', cat_scheduler='log_linear', noise_dist='uniform',
                 edm_params=None, noise_dist_params=None, noise_schedule_params=None, sampler_params=None,
                 device=torch.device('cpu'), **kwargs):
        super().__init__()
        edm_params = edm_params or {}
        noise_dist_params = noise_dist_params or {}
        noise_schedule_params = noise_schedule_params or {}
        sampler_params = sampler_params or {}

        self.num_numerical_features = num_numerical_features
        self.num_classes = num_classes
        self.num_classes_expanded = torch.from_numpy(
            np.concatenate([num_classes[i].repeat(num_classes[i]) for i in range(len(num_classes))])
        ).to(device) if len(num_classes) > 0 else torch.tensor([]).to(device).int()
        self.mask_index = torch.tensor(self.num_classes).long().to(device)
        self.neg_infinity = -1000000.0
        self.num_classes_w_mask = tuple(self.num_classes + 1)

        offsets = np.cumsum(self.num_classes)
        offsets = np.append([0], offsets)
        self.slices_for_classes = [np.arange(offsets[i - 1], offsets[i]) for i in range(1, len(offsets))]
        self.offsets = torch.from_numpy(offsets).to(device)

        offsets2 = np.cumsum(self.num_classes) + np.arange(1, len(self.num_classes) + 1)
        offsets2 = np.append([0], offsets2)
        self.slices_for_classes_with_mask = [np.arange(offsets2[i - 1], offsets2[i]) for i in range(1, len(offsets2))]

        self._denoise_fn = denoise_fn
        self.y_only_model = y_only_model
        self.num_timesteps = num_timesteps
        self.scheduler = scheduler
        self.cat_scheduler = cat_scheduler
        self.noise_dist = noise_dist
        self.edm_params = edm_params
        self.noise_dist_params = noise_dist_params
        self.sampler_params = sampler_params

        if num_numerical_features == 0:
            self.sampler_params['stochastic_sampler'] = False
            self.sampler_params['second_order_correction'] = False

        self.w_num = 0.0
        self.w_cat = 0.0
        self.num_mask_idx = []
        self.cat_mask_idx = []
        self.device = device

        if scheduler == 'power_mean':
            self.num_schedule = PowerMeanNoise(**noise_schedule_params)
        elif scheduler == 'power_mean_per_column':
            self.num_schedule = PowerMeanNoise_PerColumn(num_numerical=num_numerical_features, **noise_schedule_params)
        else:
            raise NotImplementedError(f'Scheduler {scheduler} not implemented')

        if cat_scheduler == 'log_linear':
            self.cat_schedule = LogLinearNoise(**noise_schedule_params)
        elif cat_scheduler == 'log_linear_per_column':
            self.cat_schedule = LogLinearNoise_PerColumn(num_categories=len(num_classes), **noise_schedule_params)
        else:
            raise NotImplementedError(f'Cat scheduler {cat_scheduler} not implemented')

    def mixed_loss(self, x):
        b = x.shape[0]
        device = x.device
        x_num = x[:, :self.num_numerical_features]
        x_cat = x[:, self.num_numerical_features:].long()

        if self.noise_dist == 'uniform_t':
            t = torch.rand(b, device=device, dtype=x_num.dtype)[:, None]
            sigma_num = self.num_schedule.total_noise(t)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = self.cat_schedule.rate_noise(t)
        else:
            sigma_num = self._sample_ctime_noise(x)
            t = self.num_schedule.inverse_to_t(sigma_num)
            while torch.any((t < 0) + (t > 1)):
                invalid_idx = ((t < 0) + (t > 1)).nonzero().squeeze(-1)
                sigma_num[invalid_idx] = self._sample_ctime_noise(x[:len(invalid_idx)])
                t = self.num_schedule.inverse_to_t(sigma_num)
            sigma_cat = self.cat_schedule.total_noise(t)
            dsigma_cat = None

        alpha = torch.exp(-sigma_cat)
        move_chance = -torch.expm1(-sigma_cat)

        x_num_t = x_num
        if x_num.shape[1] > 0:
            x_num_t = x_num + torch.randn_like(x_num) * sigma_num

        x_cat_t = x_cat
        x_cat_t_soft = x_cat
        if x_cat.shape[1] > 0:
            is_learnable = self.cat_scheduler == 'log_linear_per_column'
            strategy = 'soft' if is_learnable else 'hard'
            x_cat_t, x_cat_t_soft = self._q_xt(x_cat, move_chance, strategy=strategy)

        model_out_num, model_out_cat = self._denoise_fn(x_num_t, x_cat_t_soft, t.squeeze(), sigma=sigma_num)

        d_loss = torch.zeros((1,)).float()
        c_loss = torch.zeros((1,)).float()
        if x_num.shape[1] > 0:
            c_loss = self._edm_loss(model_out_num, x_num, sigma_num)
        if x_cat.shape[1] > 0:
            logits = self._subs_parameterization(model_out_cat, x_cat_t)
            d_loss = self._absorbed_closs(logits, x_cat, sigma_cat, dsigma_cat)

        return d_loss.mean(), c_loss.mean()

    @torch.no_grad()
    def sample(self, num_samples):
        b = num_samples
        device = self.device
        dtype = torch.float32

        t = torch.linspace(0, 1, self.num_timesteps, dtype=dtype, device=device)[:, None]
        sigma_num_cur = self.num_schedule.total_noise(t)
        sigma_cat_cur = self.cat_schedule.total_noise(t)
        sigma_num_next = torch.zeros_like(sigma_num_cur)
        sigma_num_next[1:] = sigma_num_cur[0:-1]
        sigma_cat_next = torch.zeros_like(sigma_cat_cur)
        sigma_cat_next[1:] = sigma_cat_cur[0:-1]

        if self.sampler_params.get('stochastic_sampler'):
            gamma = min(_S_churn / self.num_timesteps, np.sqrt(2) - 1) * (_S_min <= sigma_num_cur) * (sigma_num_cur <= _S_max)
            sigma_num_hat = sigma_num_cur + gamma * sigma_num_cur
            t_hat = self.num_schedule.inverse_to_t(sigma_num_hat)
            t_hat = torch.min(t_hat, dim=-1, keepdim=True).values
            zero_gamma = (gamma == 0).any()
            t_hat[zero_gamma] = t[zero_gamma]
            out_of_bound = (t_hat > 1).squeeze()
            sigma_num_hat[out_of_bound] = sigma_num_cur[out_of_bound]
            t_hat[out_of_bound] = t[out_of_bound]
            sigma_cat_hat = self.cat_schedule.total_noise(t_hat)
        else:
            t_hat, sigma_num_hat, sigma_cat_hat = t, sigma_num_cur, sigma_cat_cur

        z_norm = torch.randn((b, self.num_numerical_features), device=device) * sigma_num_cur[-1]
        has_cat = len(self.num_classes) > 0
        z_cat = torch.zeros((b, 0), device=device).float()
        if has_cat:
            z_cat = self._sample_masked_prior(b, len(self.num_classes))

        pbar = tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps, desc='Sampling')
        for i in pbar:
            z_norm, z_cat, _ = self._edm_update(
                z_norm, z_cat, i,
                t[i], t[i - 1] if i > 0 else None, t_hat[i],
                sigma_num_cur[i], sigma_num_next[i], sigma_num_hat[i],
                sigma_cat_cur[i], sigma_cat_next[i], sigma_cat_hat[i],
            )

        assert torch.all(z_cat < self.mask_index)
        return torch.cat([z_norm, z_cat], dim=1).cpu()

    def sample_all(self, num_samples, batch_size, keep_nan_samples=False):
        all_samples = []
        num_generated = 0
        while num_generated < num_samples:
            print(f'Samples left to generate: {num_samples - num_generated}')
            sample = self.sample(batch_size)
            mask_nan = torch.any(sample.isnan(), dim=1)
            sample = sample * (~mask_nan)[:, None] if keep_nan_samples else sample[~mask_nan]
            all_samples.append(sample)
            num_generated += sample.shape[0]
        return torch.cat(all_samples, dim=0)[:num_samples]

    def _q_xt(self, x, move_chance, strategy='hard'):
        if strategy == 'hard':
            move_indices = torch.rand(*x.shape, device=x.device) < move_chance
            xt = torch.where(move_indices, self.mask_index, x)
            return xt, self._to_one_hot(xt).to(move_chance.dtype)
        else:  # soft
            bs = x.shape[0]
            xt_soft = torch.zeros(bs, torch.sum(self.mask_index + 1), device=x.device)
            xt = torch.zeros_like(x)
            for i in range(len(self.num_classes)):
                slice_i = self.slices_for_classes_with_mask[i]
                prob_i = torch.zeros(bs, 2, device=x.device)
                prob_i[:, 0] = 1 - move_chance[:, i]
                prob_i[:, -1] = move_chance[:, i]
                soft_sample_i = F.gumbel_softmax(torch.log(prob_i), tau=0.01, hard=True)
                idx = torch.stack((x[:, i] + slice_i[0], torch.ones_like(x[:, i]) * slice_i[-1]), dim=-1)
                xt_soft[torch.arange(len(idx)).unsqueeze(1), idx] = soft_sample_i
                xt[:, i] = torch.where(soft_sample_i[:, 1] > soft_sample_i[:, 0], self.mask_index[i], x[:, i])
            return xt, xt_soft

    def _subs_parameterization(self, unormalized_prob, xt):
        unormalized_prob = self._pad(unormalized_prob, self.neg_infinity)
        unormalized_prob[:, range(unormalized_prob.shape[1]), self.mask_index] += self.neg_infinity
        logits = unormalized_prob - torch.logsumexp(unormalized_prob, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _pad(self, x, pad_value):
        splited = torch.split(x, self.num_classes_w_mask, dim=-1)
        max_K = max(self.num_classes_w_mask)
        padded_ = [torch.cat((t, pad_value * torch.ones(*t.shape[:-1], max_K - t.shape[-1], dtype=t.dtype, device=t.device)), dim=-1) for t in splited]
        return torch.stack(padded_, dim=-2)

    def _to_one_hot(self, x_cat):
        return torch.cat([F.one_hot(x_cat[:, i], num_classes=self.num_classes[i] + 1) for i in range(len(self.num_classes))], dim=-1)

    def _absorbed_closs(self, model_output, x0, sigma, dsigma):
        log_p_theta = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        alpha = torch.exp(-sigma)
        if self.cat_scheduler in ('log_linear_unified', 'log_linear_per_column'):
            elbo_weight = -dsigma / torch.expm1(sigma)
        else:
            elbo_weight = -1 / (1 - alpha)
        return elbo_weight * log_p_theta

    def _sample_masked_prior(self, *batch_dims):
        return self.mask_index[None, :] * torch.ones(*batch_dims, dtype=torch.int64, device=self.mask_index.device)

    def _mdlm_update(self, log_p_x0, x, alpha_t, alpha_s):
        move_chance_t = (1 - alpha_t).unsqueeze(-1)
        move_chance_s = (1 - alpha_s).unsqueeze(-1)
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, range(q_xs.shape[1]), self.mask_index] = move_chance_s[:, :, 0]
        dummy_mask = torch.tensor([[(1 if i <= mask_idx else 0) for i in range(max(self.mask_index + 1))] for mask_idx in self.mask_index], device=q_xs.device)
        q_xs *= torch.ones_like(q_xs) * dummy_mask
        _x = self._sample_categorical(q_xs)
        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x, q_xs

    def _sample_categorical(self, categorical_probs):
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)

    def _sample_ctime_noise(self, batch):
        if self.noise_dist == 'log_norm':
            rnd_normal = torch.randn(batch.shape[0], device=batch.device)
            return (rnd_normal * self.noise_dist_params['P_std'] + self.noise_dist_params['P_mean']).exp()
        raise NotImplementedError(f'Noise distribution {self.noise_dist} not implemented')

    def _edm_loss(self, D_yn, y, sigma):
        weight = (sigma ** 2 + self.edm_params['sigma_data'] ** 2) / (sigma * self.edm_params['sigma_data']) ** 2
        return weight * ((D_yn - y) ** 2)

    def _edm_update(self, x_num_cur, x_cat_cur, i, t_cur, t_next, t_hat,
                    sigma_num_cur, sigma_num_next, sigma_num_hat,
                    sigma_cat_cur, sigma_cat_next, sigma_cat_hat):
        b = x_num_cur.shape[0]
        has_cat = len(self.num_classes) > 0

        x_num_hat = x_num_cur + (sigma_num_hat ** 2 - sigma_num_cur ** 2).sqrt() * _S_noise * torch.randn_like(x_num_cur)
        move_chance = -torch.expm1(sigma_cat_cur - sigma_cat_hat)
        x_cat_hat, _ = self._q_xt(x_cat_cur, move_chance) if has_cat else (x_cat_cur, x_cat_cur)

        x_cat_hat_oh = self._to_one_hot(x_cat_hat).to(x_num_hat.dtype) if has_cat else x_cat_hat
        denoised, raw_logits = self._denoise_fn(
            x_num_hat.float(), x_cat_hat_oh,
            t_hat.squeeze().repeat(b), sigma=sigma_num_hat.unsqueeze(0).repeat(b, 1)
        )

        d_cur = (x_num_hat - denoised) / sigma_num_hat
        x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * d_cur

        x_cat_next = x_cat_cur
        q_xs = torch.zeros_like(x_cat_cur).float()
        if has_cat:
            logits = self._subs_parameterization(raw_logits, x_cat_hat)
            alpha_t = torch.exp(-sigma_cat_hat).unsqueeze(0).repeat(b, 1)
            alpha_s = torch.exp(-sigma_cat_next).unsqueeze(0).repeat(b, 1)
            x_cat_next, q_xs = self._mdlm_update(logits, x_cat_hat, alpha_t, alpha_s)

        if self.sampler_params.get('second_order_correction') and i > 0:
            x_cat_hat_oh = self._to_one_hot(x_cat_hat).to(x_num_next.dtype) if has_cat else x_cat_hat
            denoised, _ = self._denoise_fn(
                x_num_next.float(), x_cat_hat_oh,
                t_next.squeeze().repeat(b), sigma=sigma_num_next.unsqueeze(0).repeat(b, 1)
            )
            d_prime = (x_num_next - denoised) / sigma_num_next
            x_num_next = x_num_hat + (sigma_num_next - sigma_num_hat) * (0.5 * d_cur + 0.5 * d_prime)

        return x_num_next, x_cat_next, q_xs


# ============================================================================
# Post-processing helpers (from tabdiff/trainer.py)
# ============================================================================

@torch.no_grad()
def split_num_cat_target(syn_data, info, num_inverse, int_inverse, cat_inverse):
    task_type = info['task_type']
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    n_num_feat = len(num_col_idx)
    n_cat_feat = len(cat_col_idx)
    if task_type == 'regression':
        n_num_feat += len(target_col_idx)
    else:
        n_cat_feat += len(target_col_idx)

    syn_num = syn_data[:, :n_num_feat]
    syn_cat = syn_data[:, n_num_feat:]

    if n_num_feat > 0:
        syn_num = num_inverse(syn_num).astype(np.float32)
        syn_num = int_inverse(syn_num).astype(np.float32)
    else:
        syn_num = np.empty((syn_data.shape[0], 0), dtype=np.float32)

    syn_cat = cat_inverse(syn_cat)

    if task_type == 'regression':
        syn_target = syn_num[:, :len(target_col_idx)]
        syn_num = syn_num[:, len(target_col_idx):]
    else:
        syn_target = syn_cat[:, :len(target_col_idx)]
        syn_cat = syn_cat[:, len(target_col_idx):]

    return syn_num, syn_cat, syn_target


def recover_data(syn_num, syn_cat, syn_target, info):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    idx_mapping = {int(k): v for k, v in info['idx_mapping'].items()}

    syn_df = pd.DataFrame()
    for i in range(len(num_col_idx) + len(cat_col_idx) + len(target_col_idx)):
        if i in set(num_col_idx):
            syn_df[i] = syn_num[:, idx_mapping[i]]
        elif i in set(cat_col_idx):
            syn_df[i] = syn_cat[:, idx_mapping[i] - len(num_col_idx)]
        else:
            syn_df[i] = syn_target[:, idx_mapping[i] - len(num_col_idx) - len(cat_col_idx)]
    return syn_df


# ============================================================================
# EMA helper (from utils_train.py)
# ============================================================================

def update_ema(target_params, source_params, rate=0.999):
    for target, source in zip(target_params, source_params):
        target.detach().mul_(rate).add_(source.detach(), alpha=1 - rate)


# ============================================================================
# Trainer (from tabdiff/trainer.py — training loop + sample_synthetic only)
# ============================================================================

class _NullLogger:
    """Minimal wandb-compatible stub."""
    def log(self, *a, **kw): pass
    def define_metric(self, *a, **kw): pass


class TabDiffTrainer:
    def __init__(
        self, diffusion, train_loader, dataset, val_dataset, logger,
        lr, weight_decay, steps, batch_size, check_val_every,
        sample_batch_size, model_save_path, result_save_path,
        num_samples_to_generate=None, lr_scheduler='reduce_lr_on_plateau',
        reduce_lr_patience=100, factor=0.9, ema_decay=0.997,
        closs_weight_schedule='fixed', c_lambda=1.0, d_lambda=1.0,
        device=torch.device('cpu'), ckpt_path=None, **kwargs
    ):
        self.diffusion = diffusion
        self.ema_model = deepcopy(diffusion._denoise_fn)
        for p in self.ema_model.parameters(): p.detach_()
        self.ema_num_schedule = deepcopy(diffusion.num_schedule)
        for p in self.ema_num_schedule.parameters(): p.detach_()
        self.ema_cat_schedule = deepcopy(diffusion.cat_schedule)
        for p in self.ema_cat_schedule.parameters(): p.detach_()

        self.train_iter = train_loader
        self.dataset = dataset
        self.steps = steps
        self.init_lr = lr
        self.optimizer = torch.optim.AdamW(diffusion.parameters(), lr=lr, weight_decay=weight_decay)
        self.ema_decay = ema_decay
        self.closs_weight_schedule = closs_weight_schedule
        self.c_lambda = c_lambda
        self.d_lambda = d_lambda
        self.batch_size = batch_size
        self.sample_batch_size = sample_batch_size
        self.num_samples_to_generate = num_samples_to_generate
        self.logger = logger
        self.check_val_every = check_val_every
        self.device = device
        self.model_save_path = model_save_path
        self.result_save_path = result_save_path

        try:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=reduce_lr_patience, verbose=True)
        except TypeError:
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=factor, patience=reduce_lr_patience)

        self.curr_epoch = 0
        if ckpt_path is not None:
            state_dicts = torch.load(ckpt_path, map_location=device)
            if isinstance(state_dicts, dict) and 'denoise_fn' in state_dicts:
                diffusion._denoise_fn.load_state_dict(state_dicts['denoise_fn'])
                diffusion.num_schedule.load_state_dict(state_dicts['num_schedule'])
                diffusion.cat_schedule.load_state_dict(state_dicts['cat_schedule'])
            elif isinstance(state_dicts, dict):
                diffusion._denoise_fn.load_state_dict(state_dicts)
            else:
                raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')
            self.curr_epoch = int(os.path.basename(ckpt_path).split('_')[-1].split('.')[0])

    def _run_step(self, x, closs_weight, dloss_weight):
        x = x.to(self.device)
        self.diffusion.train()
        self.optimizer.zero_grad()
        dloss, closs = self.diffusion.mixed_loss(x)
        (dloss_weight * dloss + closs_weight * closs).backward()
        self.optimizer.step()
        return dloss, closs

    def _compute_loss(self):
        curr_dloss = curr_closs = curr_count = 0
        for batch in self.train_iter:
            x = batch.float().to(self.device)
            self.diffusion.eval()
            with torch.no_grad():
                bd, bc = self.diffusion.mixed_loss(x)
            curr_dloss += bd.item() * len(x)
            curr_closs += bc.item() * len(x)
            curr_count += len(x)
        return np.around(curr_dloss / curr_count, 4), np.around(curr_closs / curr_count, 4)

    def _to_ema(self):
        curr = self.diffusion._denoise_fn, self.diffusion.num_schedule, self.diffusion.cat_schedule
        self.diffusion._denoise_fn = self.ema_model
        self.diffusion.num_schedule = self.ema_num_schedule
        self.diffusion.cat_schedule = self.ema_cat_schedule
        return curr

    def _from_ema(self, curr):
        self.diffusion._denoise_fn, self.diffusion.num_schedule, self.diffusion.cat_schedule = curr

    def run_loop(self):
        closs_weight, dloss_weight = self.c_lambda, self.d_lambda
        best_ema_loss = np.inf
        print(f'Training TabDiff for {self.steps} epochs …')
        pbar_epochs = tqdm(range(self.curr_epoch, self.steps), total=self.steps - self.curr_epoch, desc='Training')
        for epoch in pbar_epochs:
            self.curr_epoch = epoch + 1
            curr_dloss = curr_closs = curr_count = 0
            if self.closs_weight_schedule == 'anneal':
                closs_weight = self.c_lambda * (1 - epoch / self.steps)
            for batch in self.train_iter:
                bd, bc = self._run_step(batch.float(), closs_weight, dloss_weight)
                curr_dloss += bd.item() * len(batch)
                curr_closs += bc.item() * len(batch)
                curr_count += len(batch)
            total_loss = np.around((curr_dloss + curr_closs) / curr_count, 4)
            self.scheduler.step(total_loss)

            update_ema(self.ema_model.parameters(), self.diffusion._denoise_fn.parameters(), self.ema_decay)
            update_ema(self.ema_num_schedule.parameters(), self.diffusion.num_schedule.parameters(), self.ema_decay)
            update_ema(self.ema_cat_schedule.parameters(), self.diffusion.cat_schedule.parameters(), self.ema_decay)

            # Track best EMA loss and optionally save checkpoint
            curr = self._to_ema()
            ema_d, ema_c = self._compute_loss()
            self._from_ema(curr)
            ema_total = ema_d + ema_c
            pbar_epochs.set_postfix({'loss': f'{total_loss:.4f}', 'ema_loss': f'{ema_total:.4f}'})
            if ema_total < best_ema_loss and self.curr_epoch > 500 and self.model_save_path:
                best_ema_loss = ema_total
                old = glob.glob(os.path.join(self.model_save_path, 'best_ema_model_*'))
                if old:
                    os.remove(old[0])
                state = {
                    'denoise_fn': self.ema_model.state_dict(),
                    'num_schedule': self.ema_num_schedule.state_dict(),
                    'cat_schedule': self.ema_cat_schedule.state_dict(),
                }
                torch.save(state, os.path.join(self.model_save_path, f'best_ema_model_{np.round(ema_total,4)}_{epoch+1}.pt'))

    def sample_synthetic(self, num_samples: int, ema: bool = False) -> pd.DataFrame:
        if ema:
            curr = self._to_ema()
        self.diffusion.eval()
        info = self.dataset.info
        syn_data = self.diffusion.sample_all(num_samples, self.sample_batch_size, keep_nan_samples=True)

        num_inverse = self.dataset.num_inverse
        int_inverse = self.dataset.int_inverse
        cat_inverse = self.dataset.cat_inverse

        syn_num, syn_cat, syn_target = split_num_cat_target(syn_data, info, num_inverse, int_inverse, cat_inverse)
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)
        idx_name_mapping = {int(k): v for k, v in info['idx_name_mapping'].items()}
        syn_df.rename(columns=idx_name_mapping, inplace=True)

        if ema:
            self._from_ema(curr)
        return syn_df


# ============================================================================
# Public helper used by the adapter
# ============================================================================

# Default config baked in (mirrors tabdiff/configs/tabdiff_configs.toml)
_DEFAULT_CONFIG = {
    'data': {'dequant_dist': 'none', 'int_dequant_factor': 0},
    'unimodmlp_params': {'num_layers': 2, 'd_token': 4, 'n_head': 1, 'factor': 32, 'bias': True, 'dim_t': 1024, 'use_mlp': True},
    'diffusion_params': {
        'num_timesteps': 50,
        'scheduler': 'power_mean',
        'cat_scheduler': 'log_linear',
        'noise_dist': 'uniform_t',
        'sampler_params': {'stochastic_sampler': True, 'second_order_correction': True},
        'edm_params': {'precond': True, 'sigma_data': 1.0, 'net_conditioning': 'sigma'},
        'noise_dist_params': {'P_mean': -1.2, 'P_std': 1.2},
        'noise_schedule_params': {
            'sigma_min': 0.002, 'sigma_max': 80, 'rho': 7,
            'eps_max': 1e-3, 'eps_min': 1e-5,
            'rho_init': 7.0, 'rho_offset': 5.0, 'k_init': -6.0, 'k_offset': 1.0,
        },
    },
    'train': {'main': {'steps': 2500, 'lr': 0.001, 'weight_decay': 0, 'ema_decay': 0.997, 'batch_size': 4096, 'check_val_every': 500}},
    'sample': {'batch_size': 10000},
}


def build_and_train(
    data_dir: str,
    info: dict,
    num_samples: int,
    device: torch.device,
    steps: int,
    learnable_schedule: bool,
    sample_batch_size: int,
    model_save_path: Optional[str],
    result_save_path: Optional[str],
) -> pd.DataFrame:
    """Build, train, and sample from a TabDiff model.  Returns a synthetic DataFrame."""
    import copy
    cfg = copy.deepcopy(_DEFAULT_CONFIG)
    cfg['train']['main']['steps'] = steps

    dataset_kwargs = dict(y_only=False, dequant_dist=cfg['data']['dequant_dist'], int_dequant_factor=cfg['data']['int_dequant_factor'])
    train_dataset = TabDiffDataset(data_dir, info, isTrain=True, **dataset_kwargs)
    val_dataset = TabDiffDataset(data_dir, info, isTrain=False, **dataset_kwargs)

    train_loader = DataLoader(train_dataset, batch_size=cfg['train']['main']['batch_size'], shuffle=True, num_workers=0)

    d_numerical = train_dataset.d_numerical
    categories = train_dataset.categories

    cfg['unimodmlp_params']['d_numerical'] = d_numerical
    cfg['unimodmlp_params']['categories'] = (categories + 1).tolist()

    backbone = UniModMLP(**cfg['unimodmlp_params'])
    model = Model(backbone, **cfg['diffusion_params']['edm_params'])
    model.to(device)

    diff_cfg = copy.deepcopy(cfg['diffusion_params'])
    if learnable_schedule:
        diff_cfg['scheduler'] = 'power_mean_per_column'
        diff_cfg['cat_scheduler'] = 'log_linear_per_column'

    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **diff_cfg,
        device=device,
    )
    diffusion.to(device)

    trainer = TabDiffTrainer(
        diffusion=diffusion,
        train_loader=train_loader,
        dataset=train_dataset,
        val_dataset=val_dataset,
        logger=_NullLogger(),
        num_samples_to_generate=num_samples,
        model_save_path=model_save_path,
        result_save_path=result_save_path,
        device=device,
        ckpt_path=None,
        sample_batch_size=sample_batch_size,
        **cfg['train']['main'],
    )
    trainer.run_loop()
    diffusion.eval()
    return trainer.sample_synthetic(num_samples, ema=True)


# ---------------------------------------------------------------------------
# TabDiff helpers (self-contained; do not depend on being inside the TabDiff repo)
# ---------------------------------------------------------------------------

def _tabdiff_find_target_column(df: pd.DataFrame) -> List[int]:
    keywords = {'target', 'label', 'class', 'outcome', 'result', 'diagnosis'}
    for i, col in enumerate(df.columns):
        if col.lower() in keywords:
            return [i]
    return [len(df.columns) - 1]


def _tabdiff_detect_column_types(df: pd.DataFrame, target_col_idx: List[int]):
    num_col_idx, cat_col_idx = [], []
    for i, col in enumerate(df.columns):
        if i in target_col_idx:
            continue
        try:
            pd.to_numeric(df[col])
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False
        if is_numeric:
            n_unique = df[col].nunique()
            if n_unique / len(df) < 0.05 and n_unique <= 20:
                cat_col_idx.append(i)
            else:
                num_col_idx.append(i)
        else:
            cat_col_idx.append(i)
    return num_col_idx, cat_col_idx


def _tabdiff_column_name_mapping(column_names, num_col_idx, cat_col_idx, target_col_idx):
    """Build idx_mapping / inverse_idx_mapping / idx_name_mapping.

    TabDiff reorders columns internally as [numerical | categorical | target].
    """
    idx_mapping = {}
    curr_num = 0
    curr_cat = len(num_col_idx)
    curr_tgt = curr_cat + len(cat_col_idx)
    for idx in range(len(column_names)):
        if idx in num_col_idx:
            idx_mapping[int(idx)] = curr_num; curr_num += 1
        elif idx in cat_col_idx:
            idx_mapping[int(idx)] = curr_cat; curr_cat += 1
        else:
            idx_mapping[int(idx)] = curr_tgt; curr_tgt += 1
    inverse_idx_mapping = {int(v): int(k) for k, v in idx_mapping.items()}
    idx_name_mapping = {int(i): str(column_names[i]) for i in range(len(column_names))}
    return idx_mapping, inverse_idx_mapping, idx_name_mapping


def _tabdiff_build_info(df_train, df_test, num_col_idx, cat_col_idx,
                        target_col_idx, task_type, data_dir, name):
    column_names = df_train.columns.tolist()
    complete_df = pd.concat([df_train, df_test], ignore_index=True)

    int_columns, int_col_idx, int_col_idx_wrt_num = [], [], []
    for pos, col_idx in enumerate(num_col_idx):
        col = column_names[col_idx]
        col_data = pd.to_numeric(complete_df[col], errors='coerce').dropna()
        if (col_data % 1 == 0).all():
            int_columns.append(col)
            int_col_idx.append(col_idx)
            int_col_idx_wrt_num.append(pos)

    idx_mapping, inverse_idx_mapping, idx_name_mapping = _tabdiff_column_name_mapping(
        column_names, num_col_idx, cat_col_idx, target_col_idx
    )

    col_info: dict = {}
    for col_idx in num_col_idx:
        col = column_names[col_idx]
        col_info[col_idx] = {}
        col_info['type'] = 'numerical'
        col_info['max'] = float(df_train[col].max())
        col_info['min'] = float(df_train[col].min())
    for col_idx in cat_col_idx:
        col_info[col_idx] = {}
        col_info['type'] = 'categorical'
        col_info['categorizes'] = list(set(df_train[column_names[col_idx]].tolist()))
    for col_idx in target_col_idx:
        col = column_names[col_idx]
        if task_type == 'regression':
            col_info[col_idx] = {}
            col_info['type'] = 'numerical'
            col_info['max'] = float(df_train[col].max())
            col_info['min'] = float(df_train[col].min())
        else:
            col_info[col_idx] = {}
            col_info['type'] = 'categorical'
            col_info['categorizes'] = list(set(df_train[col].tolist()))

    n_classes = None
    if task_type in ('binclass', 'multiclass'):
        n_classes = int(df_train[column_names[target_col_idx[0]]].nunique())

    return {
        "name": name,
        "task_type": task_type,
        "header": "infer",
        "column_names": column_names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": os.path.join(data_dir, f"{name}.csv"),
        "test_path": None,
        "int_col_idx": int_col_idx,
        "int_columns": int_columns,
        "int_col_idx_wrt_num": int_col_idx_wrt_num,
        "column_info": col_info,
        "train_num": len(df_train),
        "test_num": len(df_test),
        "val_num": 0,
        "n_classes": n_classes,
        "idx_mapping": {str(k): int(v) for k, v in idx_mapping.items()},
        "inverse_idx_mapping": {str(k): int(v) for k, v in inverse_idx_mapping.items()},
        "idx_name_mapping": {str(k): v for k, v in idx_name_mapping.items()},
    }


def _tabdiff_write_npy_files(df_train, df_test, num_col_idx, cat_col_idx,
                              target_col_idx, data_dir):
    column_names = df_train.columns.tolist()
    num_columns = [column_names[i] for i in num_col_idx]
    cat_columns = [column_names[i] for i in cat_col_idx]
    target_columns = [column_names[i] for i in target_col_idx]
    for split_df, split in ((df_train, 'train'), (df_test, 'test')):
        np.save(os.path.join(data_dir, f'X_num_{split}.npy'),
                split_df[num_columns].to_numpy().astype(np.float32))
        np.save(os.path.join(data_dir, f'X_cat_{split}.npy'),
                split_df[cat_columns].to_numpy())
        np.save(os.path.join(data_dir, f'y_{split}.npy'),
                split_df[target_columns].to_numpy())

def _tabdiff_restore_column_types(syn_df: pd.DataFrame, original_df: pd.DataFrame, original_dtypes) -> pd.DataFrame:
    """Restore original column types and values in synthetic data generated by TabDiff.
    
    TabDiff converts all data to numeric internally, so this function restores
    the original object types (intervals, categorical values, etc.) by mapping
    synthetic numeric values back to training data categories.
    
    Args:
        syn_df: Synthetic DataFrame generated by TabDiff
        original_df: Original training DataFrame with correct types
        original_dtypes: Original column dtypes from training data
        
    Returns:
        DataFrame: Synthetic data with restored column types
    """
    for col in syn_df.columns:
        if col in original_dtypes and col in original_df.columns:
            original_dtype = original_dtypes[col]
            
            # If original column had object dtype (intervals, special categories), restore values
            if pd.api.types.is_object_dtype(original_dtype):
                # Get unique values from training data to map back
                unique_train_vals = original_df[col].unique()
                
                # If synthetic column is numeric and training was object, 
                # map rounded values back to original categories
                if pd.api.types.is_numeric_dtype(syn_df[col]):
                    try:
                        indices = np.round(syn_df[col].values).astype(int)
                        indices = np.clip(indices, 0, len(unique_train_vals) - 1)
                        syn_df[col] = [unique_train_vals[i] for i in indices]
                    except (TypeError, ValueError, IndexError):
                        pass  # Keep as-is if mapping fails
            else:
                # For non-object types, try direct dtype conversion
                try:
                    syn_df[col] = syn_df[col].astype(original_dtype)
                except (TypeError, ValueError):
                    pass
    
    return syn_df