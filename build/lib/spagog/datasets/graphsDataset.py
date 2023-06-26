import copy

import torch
import numpy as np
from typing import List, Union

from .graphsDataPair import GraphsDataPair
from .tabDataset import TabDataset

from ..utils.data.tab2graph import tab2graphs

from operator import itemgetter


class GraphsDataset:
    _input_types = Union[torch.Tensor, np.ndarray]

    def __init__(
        self,
        name: str,
        train: GraphsDataPair,
        test: GraphsDataPair = None,
        val: GraphsDataPair = None,
        normalize: bool = True,
    ):
        self.name = name
        self.normalized = False

        self.train = train
        self.train.denormalize(inplace=True)

        self.test_exists = test is not None
        self.val_exists = val is not None

        if self.test_exists:
            assert (
                test.num_features == train.num_features
            ), "Test doesn't have the same number of features as in train"
            self.test = test
            self.test.denormalize(inplace=True)

        if self.val_exists:
            assert (
                val.num_features == train.num_features
            ), "Validation doesn't have the same number of features as in train"
            self.val = val
            self.val.denormalize(inplace=True)

        if normalize:
            self.zscore()

    @classmethod
    def from_attributes(
        cls,
        train_X_attributes: List[_input_types],
        train_edges: List[_input_types],
        train_Y: List[_input_types] = None,
        test_X_attributes: List[_input_types] = None,
        test_edges: List[_input_types] = None,
        test_Y: List[_input_types] = None,
        val_X_attributes: List[_input_types] = None,
        val_edges: List[_input_types] = None,
        val_Y: List[_input_types] = None,
        name: str = "",
        normalize: bool = False,
        **kwargs,
    ):
        test_exists = test_X_attributes is not None
        val_exists = val_X_attributes is not None

        train = GraphsDataPair(
            X_list=train_X_attributes,
            edges_list=train_edges,
            Y_list=train_Y,
            name=f"{name} - train",
            normalize=False,
            **kwargs,
        )

        if test_exists:
            test = GraphsDataPair(
                X_list=test_X_attributes,
                edges_list=test_edges,
                Y_list=test_Y,
                name=f"{name} - test",
                normalize=False,
                **kwargs,
            )
        else:
            test = None

        if val_exists:
            val = GraphsDataPair(
                X_list=val_X_attributes,
                edges_list=val_edges,
                Y_list=val_Y,
                name=f"{name} - val",
                normalize=False,
                **kwargs,
            )
        else:
            val = None

        return cls(name=name, train=train, val=val, test=test, normalize=normalize)

    @classmethod
    def from_tab(
            cls, tab_data: TabDataset, inter_sample_edges: torch.Tensor = None, calc_intra_edges: bool = True, **kwargs
    ):
        is_graph = inter_sample_edges is not None
        test_y_given = tab_data.test.Y is not None

        if is_graph:
            (
                all_data_X,
                all_data_Y,
                train_mask,
                val_mask,
                test_mask,
            ) = tab_data.get_all_data()

            (graph_kwargs, *_), _ = tab2graphs(
                tab_data=all_data_X,
                val_mask=val_mask,
                test_mask=test_mask,
                include_edge_weights=True,
                inter_sample_edges=inter_sample_edges,
                name=tab_data.name.split('-')[0].lower(),
                calc_intra_edges=calc_intra_edges,
                **kwargs,
            )

            all_imputed_X = graph_kwargs["X_list"]
            all_imputed_X = torch.stack(all_imputed_X)

            if (
                    tab_data.normalized
                    and tab_data.train.existence_cols is not None
                    and len(tab_data.train.existence_cols) > 0
            ):
                test_val_ind = list(set(range(all_imputed_X.shape[0])) - set(train_mask))

                train_imputed_X = all_imputed_X[train_mask]
                test_val_imputed_X = all_imputed_X[test_val_ind]

                test_val_data_orig_feats = test_val_imputed_X
                train_data_orig_feats = train_imputed_X
                train_mean, train_std = train_data_orig_feats.mean(0), train_data_orig_feats.std(0)

                train_mean[tab_data.one_hot_feats] = 0
                train_std[tab_data.one_hot_feats] = 1

                test_val_data_orig_feats = (
                                                   test_val_data_orig_feats - train_mean
                                           ) / train_std
                train_data_orig_feats = (train_data_orig_feats - train_mean) / train_std

                all_imputed_X[test_val_ind] = test_val_data_orig_feats
                all_imputed_X[train_mask] = train_data_orig_feats

            all_imputed_X = list(all_imputed_X)

            all_intra_edges = graph_kwargs["edges_list"]

            graphs_dataset = cls.from_attributes(
                train_X_attributes=list(itemgetter(*train_mask)(all_imputed_X)),
                train_edges=list(itemgetter(*train_mask)(all_intra_edges)) if calc_intra_edges else None,
                train_Y=list(all_data_Y[train_mask]),
                val_X_attributes=list(itemgetter(*val_mask)(all_imputed_X)),
                val_edges=list(itemgetter(*val_mask)(all_intra_edges)) if calc_intra_edges else None,
                val_Y=list(all_data_Y[val_mask]),
                test_X_attributes=list(itemgetter(*test_mask)(all_imputed_X)),
                test_edges=list(itemgetter(*test_mask)(all_intra_edges)) if calc_intra_edges else None,
                test_Y=list(all_data_Y[test_mask]) if test_y_given else None,
                normalize=False,
                name=f"{tab_data.name} - graphs",
                given_as_adj=True,
                include_edge_weights=True,
            )

            graphs_dataset.normalized = True

            return (
                graphs_dataset,
                inter_sample_edges,
                [
                    train_mask,
                    val_mask,
                    test_mask,
                ],
            )

        else:
            (
                all_data_X,
                all_data_Y,
                train_mask,
                val_mask,
                test_mask,
            ) = tab_data.get_all_data()

            if inter_sample_edges is not None:
                inter_sample_edges_train = torch.stack(
                    list(filter(lambda e: e[0] <= train_mask[-1], inter_sample_edges))
                )

            else:
                inter_sample_edges_train = None

            train_graph_kwargs, *_ = tab2graphs(
                tab_data=tab_data.get_train_data(as_loader=False),
                val_mask=None,
                test_mask=None,
                include_edge_weights=True,
                inter_sample_edges=inter_sample_edges_train,
                name=tab_data.name.split('-')[0].lower(),
                **kwargs,
            )

            train_edges_list = train_graph_kwargs["inter_sample_edges"]

            all_graph_kwargs, *_ = tab2graphs(
                tab_data=all_data_X,
                val_mask=val_mask,
                test_mask=test_mask,
                include_edge_weights=True,
                inter_sample_edges=inter_sample_edges,
                name=tab_data.name.split('-')[0].lower(),
                **kwargs,
            )

            all_imputed_X = all_graph_kwargs["X_list"]
            all_imputed_X = torch.stack(all_imputed_X)

            if (
                    tab_data.normalized
                    and tab_data.train.existence_cols is not None
                    and len(tab_data.train.existence_cols) > 0
            ):
                test_val_ind = list(set(range(all_imputed_X.shape[0])) - set(train_mask))

                train_imputed_X = all_imputed_X[train_mask]
                test_val_imputed_X = all_imputed_X[test_val_ind]

                test_val_data_orig_feats = test_val_imputed_X
                train_data_orig_feats = train_imputed_X
                train_mean, train_std = train_data_orig_feats.mean(
                    0
                ), train_data_orig_feats.std(0)

                train_mean[tab_data.one_hot_feats] = 0
                train_std[tab_data.one_hot_feats] = 1

                test_val_data_orig_feats = (
                                                   test_val_data_orig_feats - train_mean
                                           ) / train_std
                train_data_orig_feats = (train_data_orig_feats - train_mean) / train_std

                all_imputed_X[test_val_ind] = test_val_data_orig_feats
                all_imputed_X[train_mask] = train_data_orig_feats

            all_imputed_X = list(all_imputed_X)

            all_intra_edges = all_graph_kwargs["edges_list"]

            graphs_dataset = cls.from_attributes(
                train_X_attributes=list(itemgetter(*train_mask)(all_imputed_X)),
                train_edges=list(itemgetter(*train_mask)(all_intra_edges)),
                train_Y=list(all_data_Y[train_mask]),
                val_X_attributes=list(itemgetter(*val_mask)(all_imputed_X)),
                val_edges=list(itemgetter(*val_mask)(all_intra_edges)),
                val_Y=list(all_data_Y[val_mask]),
                test_X_attributes=list(itemgetter(*test_mask)(all_imputed_X)),
                test_edges=list(itemgetter(*test_mask)(all_intra_edges)),
                test_Y=list(all_data_Y[test_mask]) if test_y_given else None,
                normalize=False,
                name=f"{tab_data.name} - graphs",
                given_as_adj=True,
                include_edge_weights=True,
            )

            graphs_dataset.normalized = True

            all_inter_edges = all_graph_kwargs["inter_sample_edges"]
            all_inter_edges[all_inter_edges[:, 0] <= train_mask[-1], :] = train_edges_list

            return (
                graphs_dataset,
                all_inter_edges,
                [
                    train_mask,
                    val_mask,
                    test_mask,
                ],
            )

    def zscore(self):
        _, mu, sigma = self.train.zscore(return_params=True, inplace=True)
        if self.test_exists:
            self.test.zscore(normalization_params=(mu, sigma), inplace=True)
        if self.val_exists:
            self.val.zscore(normalization_params=(mu, sigma), inplace=True)

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

    def get_train_data(self, as_loader: bool = False, **kwargs):
        train = self.train
        if as_loader:
            train = train.to_loader(**kwargs)

        return train

    def get_test_data(self, as_loader: bool = False, **kwargs):
        assert self.test_exists, "Test data is not available"

        test = self.test
        if as_loader:
            test = test.to_loader(**kwargs)

        return test

    def get_val_data(self, as_loader: bool = False, **kwargs):
        assert self.val_exists, "Validation data is not available"

        val = self.val

        if as_loader:
            val = val.to_loader(**kwargs)

        return val

    def get_all_data_loader(self, **kwargs):
        train_data = self.get_train_data(as_loader=False)
        val_data = self.get_val_data(as_loader=False)
        test_data = self.get_test_data(as_loader=False)

        all_data = copy.deepcopy(train_data)
        all_data = all_data + val_data + test_data
        all_data.name = "all_data_graphs"

        return all_data.to_loader(**kwargs)

    def __str__(self):
        return f'Dataset "{self.name}" contains {self.train.X.shape[1]} features, including train {f", test" if self.test_exists else ""}, {f"val" if self.val_exists else ""}'

    def __repr__(self):
        return self.__str__()

    @property
    def num_features(self):
        return self.train.num_features

    @property
    def num_classes(self):
        return self.train.num_classes

    @property
    def train_len(self):
        return len(self.train)

    @property
    def test_len(self):
        if not self.test_exists:
            print("Test data is not available")
            return None

        return len(self.test)

    @property
    def val_len(self):
        if not self.val_exists:
            print("Validation data is not available")
            return None

        return len(self.val)

    def __len__(self):
        l = self.train_len
        if self.val_exists:
            l += self.val_len
        if self.test_exists:
            l += self.test_len

        return l
