import copy

import numpy as np
import torch
from ..datasets.tabDataPair import TabDataPair

from copy import deepcopy

from typing import Type


class KNN:
    _input_types = [np.ndarray, torch.Tensor, TabDataPair]

    def __init__(
            self, distance: str = "heur_dist", k: int = 3
    ):
        self.dist_name = distance
        self.k = k

    def get_edges(
            self,
            data: _input_types,
            return_type: [Type[torch.LongTensor], Type[np.ndarray]] = torch.LongTensor,
            as_adj: bool = True,
            val_mask: list = None,
            test_mask: list = None,
    ):
        val_mask = val_mask if val_mask is not None else []
        test_mask = test_mask if test_mask is not None else []
        train_mask = list(set(range(data.shape[0])) - set(val_mask) - set(test_mask))

        data_ = deepcopy(data)

        if not isinstance(data_, np.ndarray):
            if isinstance(data_, TabDataPair):
                data_ = data_.X

            data_ = data_.cpu().detach().numpy()

        knn = self._calc_knn_obj(data_, train_mask=train_mask)
        return KNN._parse_knn(knn[0], return_type=return_type, as_adj=as_adj), knn[1]

    def _calc_knn_obj(self, data: np.ndarray, train_mask: list):
        if self.dist_name == "euclidian":
            data_ = np.nan_to_num(data)

            data_all = copy.copy(data_)

            data_train = data_[train_mask]

            existence_train = data_train[:, data_train.shape[1] // 2:]

            data_all = data_all[:, : data_all.shape[1] // 2]
            data_train = data_train[:, : data_train.shape[1] // 2]

            f = lambda r: np.sqrt(np.power(data_train - r, 2).sum(1))

            dists = np.apply_along_axis(f, 1, data_all)
            edges = np.argsort(dists, 1)[:, 1: self.k + 1]

            conv2true_idx = np.vectorize(lambda i: train_mask[i])
            edges = conv2true_idx(edges)

            edges = torch.Tensor(
                [[i, j] for i in range(edges.shape[0]) for j in edges[i]]
            ).T.long()

            return edges, dists

        if self.dist_name == "cosine":
            data_ = np.nan_to_num(data)

            data_all = copy.copy(data_)
            data_train = data_[train_mask]

            train_norms = np.linalg.norm(data_train, axis=1)
            all_norms = np.linalg.norm(data_all, axis=1)

            dists = np.dot(data_all, data_train.T)
            dists = ((dists / (train_norms)).T / all_norms).T

            dists = np.nan_to_num(dists)
            edges = np.argpartition(dists, self.k, axis=1)[:, - self.k - 1:-1]

            conv2true_idx = np.vectorize(lambda i: train_mask[i])
            edges = conv2true_idx(edges)

            edges = torch.Tensor(
                [[i, j] for i in range(edges.shape[0]) for j in edges[i]]
            ).T.long()

            return edges, dists

        if self.dist_name == "heur_dist":
            data_ = np.nan_to_num(data)

            data_all = copy.copy(data_)

            data_train = data_[train_mask]

            existence_all = data_all[:, data_all.shape[1] // 2:]
            existence_train = data_train[:, data_train.shape[1] // 2:]

            data_all = data_all[:, : data_all.shape[1] // 2]
            data_train = data_train[:, : data_train.shape[1] // 2]

            f = lambda r: np.sqrt(np.power(data_train - r, 2).sum(1))

            dists = np.apply_along_axis(f, 1, data_all)

            nac = lambda r: (existence_train + r).sum(1)

            nulls_counter = np.apply_along_axis(nac, 1, existence_all)
            nulls_counter = existence_all.shape[1] * 2 - nulls_counter

            dists = np.sqrt(dists ** 2 + nulls_counter)

            edges = np.argsort(dists, 1)[:, : self.k]

            conv2true_idx = np.vectorize(lambda i: train_mask[i])
            edges = conv2true_idx(edges)

            edges = torch.Tensor(
                [[i, j] for i in range(edges.shape[0]) for j in edges[i]]
            ).T.long()

            return edges, dists

        if self.dist_name == "l1":
            data_ = np.nan_to_num(data)

            data_all = copy.copy(data_)

            data_train = data_[train_mask]

            existence_train = data_train[:, data_train.shape[1] // 2:]

            data_all = data_all[:, : data_all.shape[1] // 2]
            data_train = data_train[:, : data_train.shape[1] // 2]

            f = lambda r: np.abs(data_train - r).sum(1)

            dists = np.apply_along_axis(f, 1, data_all)

            edges = np.argsort(dists, 1)[:, 1: 1 + self.k]

            conv2true_idx = np.vectorize(lambda i: train_mask[i])
            edges = conv2true_idx(edges)

            edges = torch.Tensor(
                [[i, j] for i in range(edges.shape[0]) for j in edges[i]]
            ).T.long()

            return edges, dists

        if self.dist_name == "l_inf":
            data_ = np.nan_to_num(data)

            data_all = copy.copy(data_)

            data_train = data_[train_mask]

            existence_train = data_train[:, data_train.shape[1] // 2:]

            data_all = data_all[:, : data_all.shape[1] // 2]
            data_train = data_train[:, : data_train.shape[1] // 2]

            f = lambda r: np.abs(data_train - r).max(1)

            dists = np.apply_along_axis(f, 1, data_all)

            edges = np.argsort(dists, 1)[:, 1: 1 + self.k]

            conv2true_idx = np.vectorize(lambda i: train_mask[i])
            edges = conv2true_idx(edges)

            edges = torch.Tensor(
                [[i, j] for i in range(edges.shape[0]) for j in edges[i]]
            ).T.long()

            return edges, dists

        raise NotImplementedError(
            f"Distance metric isn't supported. Available metrics: {','.join(list(self._dist_dict.keys()))}"
        )

    @staticmethod
    def _parse_knn(
            edge_list: torch.Tensor,
            return_type: [Type[torch.LongTensor], Type[np.ndarray]] = torch.LongTensor,
            as_adj: bool = True,
    ):
        if as_adj:
            n_samples = int(edge_list.max().item()) + 1
            edges = torch.zeros((n_samples, n_samples))
            edges[edge_list.long().T.tolist()] = 1
            edge_list = edges

        if return_type == np.ndarray:
            edge_list = edge_list.detach().numpy()

        return edge_list.T.long()
