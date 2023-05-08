import os
import json
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from typing import Union, List, Tuple

from copy import deepcopy


class TabDataPair(Dataset):
    _input_types = Union[np.ndarray, torch.Tensor]

    def __init__(
        self,
        X: _input_types,
        name: str = "",
        Y: _input_types = None,
        normalize: bool = False,
        normalization_params: Tuple[List, List] = None,
        shuffle: bool = False,
        add_existence_cols: bool = False,
        fill_na: bool = False,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    ):
        assert Y is None or Y.shape[0] in X.shape

        self.device = device
        self.X = self._transform_types(X, float).to(self.device)
        self.Y = self._transform_types(Y, int).to(self.device) if Y is not None else None

        self.normalized = False
        self.norm_params = None

        if normalize:
            self.zscore(normalization_params, inplace=True, return_params=False)

        if fill_na:
            self.fill_na(inplace=True)

        if shuffle:
            self._shuffle()

        self.existence_cols = []

        if add_existence_cols:
            self.add_existence_cols(inplace=True)

        self.name = name
        self.train = self.Y is not None

    @staticmethod
    def _transform_types(
        data: _input_types, _naive_type: Union[int, float] = float
    ) -> torch.Tensor:
        if type(data) == np.ndarray:
            data = torch.from_numpy(data)

        data = data.long() if _naive_type == int else data.float()

        return data

    def _shuffle(self, inplace: bool = True):
        idx = torch.randperm(self.X.shape[0])

        if inplace:
            if self.Y is not None:
                self.X, self.Y = self.X[idx], self.Y[idx]

            else:
                self.X = self.X[idx]

            return self

        else:
            if self.Y is not None:
                return self.X[idx], self.Y[idx]

            return self.X[idx]

    def get_features(self):
        return self.X

    def get_labels(self):
        return self.Y

    def get_num_features(self):
        return self.X.shape[1]

    def get_num_samples(self):
        return self.X.shape[0]

    def get_num_classes(self):
        return self.Y.max().item() + 1 if self.Y is not None else -1

    def zscore(
        self,
        params: Tuple[List, List] = None,
        inplace: bool = True,
        return_params: bool = False,
        one_hot_features: List[int] = [],
    ):
        if self.normalized:
            print("Data is already normalize!")
            return self.X if not inplace else self

        if params is not None:
            mu, sigma = params
            assert len(mu) == len(sigma) == self.X.shape[1]

        else:
            mu = torch.nanmean(self.X, axis=0).to(self.device)
            sigma = torch.from_numpy(np.nanstd(self.X.cpu().detach().numpy(), axis=0)).to(self.device)

        if one_hot_features is not None and len(one_hot_features) > 0:
            mu[one_hot_features] = 0
            sigma[one_hot_features] = 1

        if inplace:
            self.normalized = True
            self.norm_params = (mu, sigma)

            self.X = (self.X - mu) / sigma

            if return_params:
                return self, mu, sigma

            return self

        if return_params:
            return (self.X - mu) / sigma, mu, sigma

        return (self.X - mu) / sigma

    def denormalize(self, inplace: bool = True):
        if not self.normalized:
            return self.X if not inplace else self

        mu, sigma = self.norm_params
        denorm_ = self.X * sigma + mu

        if inplace:
            self.normalized = False
            self.norm_params = None

            self.X = denorm_
            return self

        return denorm_

    def fill_na(self, inplace: bool = True):
        if inplace:
            self.X = torch.nan_to_num(self.X)
            return self

        return torch.nan_to_num(self.X)

    def __getitem__(self, idx):
        if self.train:
            return self.X[idx], self.Y[idx]

        return self.X[idx]

    def __len__(self):
        return self.X.shape[0]

    def __str__(self):
        return f'Dataset "{self.name}" contains {self.X.shape[0]} samples and {self.X.shape[1]} features, in {"train_graph" if self.train else "test_graph"} mode '

    def get_feat_corr(self, abs_: bool = False, fill_na_first: bool = False):
        df = pd.DataFrame(self.X.cpu().detach().numpy())

        if fill_na_first:
            df = df.fillna(df.mean())

        corr = df.corr().fillna(0)

        if abs_:
            corr = abs(corr)

        return corr.values

    def add_existence_cols(self, inplace: bool = True):
        assert len(self.existence_cols) == 0, "Existence columns were already added!"
        existence_cols = 1 - torch.isnan(self.X).float()
        existence_cols = existence_cols.to(self.device)

        if inplace:
            self.X = torch.cat((self.X, existence_cols), dim=1)
            self.existence_cols = list(range(int(self.X.shape[1] / 2), self.X.shape[1]))
            return self

        return torch.cat((self.X, existence_cols), dim=1)

    def drop_existence_cols(self, inplace: bool = True):
        assert len(self.existence_cols) > 0, "Existence columns were not added!"
        if inplace:
            self.X = self.X[:, : -len(self.existence_cols)]
            self.existence_cols = []
            return self

        return self.X[:, : -len(self.existence_cols)]

    def to_loader(self, **kwargs):
        return DataLoader(self, **kwargs)

    @classmethod
    def load(
        cls,
        data_dir: str,
        data_file_name: str,
        include_config: bool = True,
        name: str = None,
        target_col: str = None,
        **kwargs,
    ):
        assert os.path.isdir(data_dir), "Directory doesn't exist!"
        assert os.path.isfile(
            os.path.join(data_dir, data_file_name)
        ), "Data file doesn't exist!"
        assert data_file_name.endswith(".csv"), "Only csv files are supported"
        assert include_config or target_col is not None

        all_data = pd.read_csv(os.path.join(data_dir, data_file_name), index_col=0)

        if include_config:
            with open(os.path.join(data_dir, "config.json"), "r") as f:
                config = json.load(f)

            target_col = config["target_col"]
            name = config.get("name", "")

        Y = all_data[target_col].values
        X = all_data.drop(target_col, axis=1).values

        return cls(
            X=X,
            Y=Y,
            name=f'{name} - {"".join(data_file_name.split(".")[:-1])}',
            **kwargs,
        )

    def set_X(self, X):
        self.X = X

    def set_Y(self, Y):
        self.Y = Y

    def __add__(self, other):
        self_ = deepcopy(self)
        other_ = deepcopy(other)

        normalized, norm_params = self_.normalized, self_.norm_params
        self_.denormalize(inplace=True)
        self_.X = torch.cat((self_.X, other_.X), dim=0)

        if self_.Y is not None:
            self_.Y = torch.cat((self_.Y, other_.Y), dim=0)

        if normalized:
            self_.zscore(params=norm_params, inplace=True)

        del other_, norm_params, normalized
        return self_
