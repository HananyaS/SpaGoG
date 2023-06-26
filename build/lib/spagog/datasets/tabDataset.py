import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

from typing import Union
from .tabDataPair import TabDataPair

from sklearn.feature_selection import mutual_info_classif
from scipy.sparse import csr_matrix

from sklearn.model_selection import train_test_split


class TabDataset:
    _input_types = Union[torch.Tensor, np.ndarray]

    def __init__(
            self,
            name: str,
            train: TabDataPair,
            test: TabDataPair = None,
            val: TabDataPair = None,
            normalize: bool = False,
            feature_selection: int = None,
    ):
        self.name = name
        self.normalized = False

        self.train = train
        self.test_exists = test is not None
        self.val_exists = val is not None

        if self.test_exists:
            assert (
                    test.get_num_features() == train.get_num_features()
            ), "Test doesn't have the same number of features as in train"
            self.test = test
            self.test.denormalize(inplace=True)

        if self.val_exists:
            assert (
                    val.get_num_features() == train.get_num_features()
            ), "Validation doesn't have the same number of features as in train"
            self.val = val
            self.val.denormalize(inplace=True)

        unique_vals = self.remove_feats()

        is_one_hot = lambda i: len(unique_vals[i]) == 2 and 0 in unique_vals[i] and 1 in unique_vals[i]
        one_hot_feats = list(filter(is_one_hot, range(len(unique_vals))))

        self.one_hot_feats = one_hot_feats

        if normalize:
            self.zscore()

        if (
                feature_selection is not None
                and 1 < int(feature_selection) < self.train.get_num_features()
        ):
            print(f"Selecting best {int(feature_selection)} features!")
            X_train_np = self.train.X.numpy()
            row, col = np.where(1 - np.isnan(X_train_np))
            data = X_train_np[row, col]

            X_train_np = csr_matrix((data, (row, col)), shape=X_train_np.shape)

            Y_train_np = self.train.Y.detach().cpu().numpy()

            mi_scores = mutual_info_classif(X_train_np, Y_train_np)
            idx2remain = np.argpartition(mi_scores, kth=-int(feature_selection))[
                         -int(feature_selection):
                         ]

            self.train.X = self.train.X[:, idx2remain]
            self.val.X = self.val.X[:, idx2remain]
            self.test.X = self.test.X[:, idx2remain]

            new_one_hot_feats = []

            for i, n in enumerate(idx2remain):
                if n in self.one_hot_feats:
                    new_one_hot_feats.append(i)

            self.one_hot_feats = new_one_hot_feats

    @classmethod
    def from_attributes(
            cls,
            name: str,
            train_X: _input_types,
            train_Y: _input_types = None,
            test_X: _input_types = None,
            test_Y: _input_types = None,
            val_X: _input_types = None,
            val_Y: _input_types = None,
            shuffle: bool = False,
            add_existence_cols: bool = False,
            normalize: bool = True,
            **kwargs,
    ):
        assert test_X is not None or test_Y is None
        assert val_X is not None or val_Y is None

        train = TabDataPair(
            X=train_X,
            Y=train_Y,
            name=f"{name} - train",
            normalize=False,
            shuffle=shuffle,
            add_existence_cols=add_existence_cols,
        )

        if test_X is not None:
            test = TabDataPair(
                X=test_X,
                Y=test_Y,
                name=f"{name} - test",
                normalize=False,
                shuffle=shuffle,
                add_existence_cols=add_existence_cols,
            )
        else:
            test = None

        if val_X is not None:
            val = TabDataPair(
                X=val_X,
                Y=val_Y,
                name=f"{name} - val",
                normalize=False,
                shuffle=shuffle,
                add_existence_cols=add_existence_cols,
            )

        else:
            val = None

        return cls(
            name=name, train=train, test=test, val=val, normalize=normalize, **kwargs
        )

    def remove_random_data(self, percentage: float):
        assert 0 < percentage < 1

        num_entries = self.train.X.shape[0] * self.train.X.shape[1]
        num_entries_to_remove = int(num_entries * percentage)

        idxs = np.random.choice(num_entries, num_entries_to_remove, replace=False)

        r = idxs // self.train.X.shape[1]
        c = idxs % self.train.X.shape[1]

        self.train.X[r, c] = np.nan

        if self.val_exists is not None:
            num_entries = self.val.X.shape[0] * self.val.X.shape[1]
            num_entries_to_remove = int(num_entries * percentage)

            idxs = np.random.choice(num_entries, num_entries_to_remove, replace=False)

            r = idxs // self.val.X.shape[1]
            c = idxs % self.val.X.shape[1]

            self.val.X[r, c] = np.nan

        if self.test_exists is not None:
            num_entries = self.test.X.shape[0] * self.test.X.shape[1]
            num_entries_to_remove = int(num_entries * percentage)

            idxs = np.random.choice(num_entries, num_entries_to_remove, replace=False)

            r = idxs // self.test.X.shape[1]
            c = idxs % self.test.X.shape[1]

            self.test.X[r, c] = np.nan

    def remove_feats(self):
        is_nan = lambda x: x == x
        find_unique_vals = lambda col: list(filter(is_nan, np.unique(col)))

        unique_vals = list(map(find_unique_vals, self.train.X.T))

        feats2remain = list(filter(lambda i: len(unique_vals[i]) > 1, range(len(unique_vals))))

        self.train.X = self.train.X[:, feats2remain]
        self.test.X = self.test.X[:, feats2remain]
        self.val.X = self.val.X[:, feats2remain]
        unique_vals = [unique_vals[i] for i in feats2remain]

        if self.train.normalized:
            self.train.norm_params = self.train.norm_params[0][feats2remain], self.train.norm_params[1][feats2remain]

        if self.test.normalized:
            self.test.norm_params = self.test.norm_params[0][feats2remain], self.test.norm_params[1][feats2remain]

        if self.val.normalized:
            self.val.norm_params = self.val.norm_params[0][feats2remain], self.val.norm_params[1][feats2remain]

        return unique_vals

    def __str__(self):
        sets = ["train"]

        if self.test_exists:
            sets.append("test")

        if self.val_exists:
            sets.append("val")

        return f'Dataset "{self.name}" contains {self.train.X.shape[1]} features, including {", ".join(sets)}'

    def zscore(self):
        if self.normalized:
            return self

        if self.train.normalized:
            self.train.denormalize(inplace=True)

        _, mu, sigma = self.train.zscore(inplace=True, return_params=True, one_hot_features=self.one_hot_feats)

        mu[self.one_hot_feats] = 0
        sigma[self.one_hot_feats] = 1

        if self.test_exists:
            if self.test.normalized:
                self.test.denormalize(inplace=True)

            self.test.zscore(inplace=True, params=(mu, sigma))

        if self.val_exists:
            if self.val.normalized:
                self.val.denormalize(inplace=True)

            self.val.zscore(inplace=True, params=(mu, sigma))

        self.normalized = True

        return self

    def denormalize(self):
        if not self.normalized:
            return self

        self.train.denormalize(inplace=True)

        if self.test_exists:
            self.test.denormalize(inplace=True)

        if self.val_exists:
            self.val.denormalize(inplace=True)

        self.normalized = False

        return self

    def get_train_data(self, as_loader=False, **kwargs):
        if as_loader:
            return DataLoader(self.train, **kwargs)

        return self.train

    def get_test_data(self, as_loader=False, **kwargs):
        if self.test_exists:
            if as_loader:
                return DataLoader(self.test, **kwargs)

            return self.test

        raise ValueError("No test data available")

    def get_val_data(self, as_loader=False, **kwargs):
        if self.val_exists:
            if as_loader:
                return DataLoader(self.val, **kwargs)

            return self.val

        raise ValueError("No val data available")

    def get_num_features(self):
        return self.train.X.shape[1]

    def get_train_corr(self, **kwargs):
        return self.train.get_feat_corr(**kwargs)

    def add_existence_cols(self):
        self.train.add_existence_cols(inplace=True)

        if self.test_exists:
            self.test.add_existence_cols(inplace=True)

        if self.val_exists:
            self.val.add_existence_cols(inplace=True)

        self.one_hot_feats.extend(self.train.existence_cols)

    def drop_existence_cols(self):
        self.train.drop_existence_cols(inplace=True)

        if self.test_exists:
            self.test.drop_existence_cols(inplace=True)

        if self.val_exists:
            self.val.drop_existence_cols(inplace=True)

    @classmethod
    def load(
            cls,
            train_X: pd.DataFrame = None,
            train_Y: pd.DataFrame = None,
            val_X: pd.DataFrame = None,
            val_Y: pd.DataFrame = None,
            test_X: pd.DataFrame = None,
            test_Y: pd.DataFrame = None,
            name: str = "",
            **kwargs,
    ):
        if train_X is not None and train_Y is not None and test_X is not None:
            if val_X is None or val_Y is None:
                train_X, val_X, train_Y, val_Y = train_test_split(train_X, train_Y, test_size=0.2, random_state=42)

            train_X = train_X.reset_index(drop=True)
            train_Y = train_Y.reset_index(drop=True)
            val_X = val_X.reset_index(drop=True)
            val_Y = val_Y.reset_index(drop=True)

            train_X = train_X.values
            train_Y = train_Y.values
            val_X = val_X.values
            val_Y = val_Y.values
            test_X = test_X.values

            if test_Y is not None:
                test_Y = test_Y.values

            return cls.from_attributes(
                train_X=train_X,
                train_Y=train_Y,
                val_X=val_X,
                val_Y=val_Y,
                test_X=test_X,
                test_Y=test_Y,
                name=name,
                **kwargs,
            )

    def get_all_data(self):
        all_data_X = self.get_train_data(as_loader=False).X
        all_data_Y = self.get_train_data(as_loader=False).Y
        train_mask = list(range(all_data_X.shape[0]))

        next_idx = all_data_X.shape[0]

        if self.val_exists:
            val_data_X = self.get_val_data(as_loader=False).X
            all_data_X = torch.cat((all_data_X, val_data_X))
            all_data_Y = torch.cat((all_data_Y, self.get_val_data(as_loader=False).Y))
            val_mask = list(range(next_idx, next_idx + val_data_X.shape[0]))
            next_idx = all_data_X.shape[0]

        else:
            val_mask = None

        if self.test_exists:
            test_data_X = self.get_test_data(as_loader=False).X
            all_data_X = torch.cat((all_data_X, test_data_X))
            if self.test.Y is not None:  # added if
                all_data_Y = torch.cat((all_data_Y, self.get_test_data(as_loader=False).Y))
            test_mask = list(range(next_idx, next_idx + test_data_X.shape[0]))

        else:
            test_mask = None

        return all_data_X, all_data_Y, train_mask, val_mask, test_mask
