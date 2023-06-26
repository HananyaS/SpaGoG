import torch
import numpy as np
from .tabDataPair import TabDataPair
from typing import Type, Union
from torch.utils.data import DataLoader, Dataset

from copy import deepcopy

from ..utils.data.tab2graph import fill_data_gfp


class GraphDataPair(TabDataPair):
    _input_types = Union[np.ndarray, torch.Tensor]

    def __init__(
        self,
        X: _input_types,
        edges: _input_types,
        Y: _input_types = None,
        given_as_adj: bool = False,
        store_as_adj: bool = False,
        include_edge_weights: bool = False,
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        **kwargs,
    ):
        Y_ = Y.to(device) if Y is not None else None
        super(GraphDataPair, self).__init__(X=X, Y=Y_, device=device, **kwargs)

        self.Y = Y
        self.edges = self._transform_types(
            edges, Type[float] if include_edge_weights else Type[int]
        )

        self.edges = self._transform_edge_format(
            self.edges,
            given_as_adj=given_as_adj,
            store_as_adj=store_as_adj,
            weights=include_edge_weights,
        )

        self.include_edge_weights = include_edge_weights
        self.as_adj = store_as_adj

    @classmethod
    def from_tab(
        cls,
        tab_data: TabDataPair,
        knn_kwargs: dict = {"distance": "euclidian", "k": 30},
        gfp_kwargs: dict = {},
        **kwargs,
    ):
        imputed_data_, inter_samples_edges = fill_data_gfp(
            tab_data_=tab_data, knn_kwargs=knn_kwargs, gfp_kwargs=gfp_kwargs
        )

        return cls(
            X=imputed_data_,
            Y=tab_data.Y,
            edges=inter_samples_edges,
            **kwargs,
        )

    @staticmethod
    def _transform_edge_format(
        edges: _input_types,
        given_as_adj: bool,
        store_as_adj: bool = False,
        weights: bool = False,
    ) -> torch.Tensor:
        required_edge_dim = 2 if not weights else 3
        assert len(edges.shape) == 2, "edges_list must be a 2D array"
        assert given_as_adj or required_edge_dim in edges.shape
        assert not given_as_adj or edges.shape[0] == edges.shape[1]

        if not given_as_adj:
            if edges.shape[0] == required_edge_dim:
                edges = edges.T

        if given_as_adj == store_as_adj:
            return edges

        if not store_as_adj:
            edge_list = torch.nonzero(edges).view(-1, 2)

            if not weights:
                return edge_list.long()

            return torch.cat(
                (edge_list, edges[edge_list[:, 0], edge_list[:, 1]].view(-1, 1)), dim=1
            )

        adj = torch.zeros(edges.shape[0], edges.shape[0])

        if weights:
            adj[edges[:, 0], edges[:, 1]] = edges[:, 2]

        else:
            adj[edges[:, 0], edges[:, 1]] = 1

        return adj

    def __len__(self):
        return len(self.X)

    def num_nodes(self):
        return self.X.shape[0]

    def num_edges(self):
        return self.edges.shape[0]

    def get_nodes(self):
        return self.X

    def get_edges(self):
        return self.edges

    def get_edge_weights(self):
        if self.include_edge_weights:
            return self.edges_to_adj(inplace=False)[:, 2]

        raise Exception("Edges don't have weights!")

    def to_loader(self, **kwargs):
        class DS(Dataset):
            def __init__(self, gdp: GraphDataPair):
                self.gdp = gdp

            def __getitem__(self, idx):
                if self.gdp.include_edge_weights:
                    res = (
                        self.gdp.X,
                        self.gdp.edges_to_lst(inplace=False)[:, :1],
                        self.gdp.get_edge_weights(),
                    )

                else:
                    res = (
                        self.gdp.X,
                        self.gdp.edges_to_lst(inplace=False),
                    )

                if self.gdp.Y is not None:
                    res = *res, self.gdp.Y

                return res

            def __len__(self):
                return len(self.gdp)

        class DL(DataLoader):
            def __init__(self, ds: DS):
                super(DL, self).__init__(ds, batch_size=len(ds))

            def __iter__(self):
                if self.dataset.gdp.include_edge_weights:
                    res = (
                        self.dataset.gdp.X,
                        self.dataset.gdp.edges_to_lst(inplace=False)[:, :1],
                        self.dataset.gdp.get_edge_weights(),
                    )

                else:
                    res = (
                        self.dataset.gdp.X,
                        self.dataset.gdp.edges_to_lst(inplace=False),
                    )

                if self.dataset.gdp.Y is not None:
                    res = *res, self.dataset.gdp.Y

        ds = DS(self)
        return DataLoader(ds, batch_size=len(ds), **kwargs)

    def edges_to_adj(self, inplace: bool = True):
        if self.as_adj:
            return self if inplace else self.edges

        adj = torch.zeros(self.X.shape[0], self.X.shape[0])
        adj[self.edges[:, 0].long(), self.edges[:, 1].long()] = (
            (self.edges[:, 2]) if self.include_edge_weights else 1
        )
        if inplace:
            self.edges = adj
            return self

        return adj

    def edges_to_lst(self, inplace: bool = True):
        if not self.as_adj:
            return self if inplace else self.edges

        edges = torch.nonzero(self.edges).view(-1, 2)
        if self.include_edge_weights:
            edges = torch.cat(
                (edges, self.edges[edges[:, 0], edges[:, 1]].view(-1, 1)), dim=1
            )

        if inplace:
            self.edges = edges
            return self

        return edges

    def _shuffle(self, inplace: bool = True):
        idx = torch.randperm(self.X.shape[0])

        convertor = self.edges_to_adj if self.as_adj else self.edges_to_lst
        edges = self.edges_to_adj(inplace=False)[idx][:, idx]

        if inplace:
            if self.Y is not None:
                self.X, self.Y, self.edges = self.X[idx], self.Y[idx], convertor(edges)

            else:
                self.X = self.X[idx]

            return self

        else:
            if self.Y is not None:
                return self.X[idx], edges, self.Y[idx]

            return self.X[idx], edges

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
        raise Exception("Cannot load graph data from a file")

    def __add__(self, other):
        assert self.include_edge_weights == other.include_edge_weights

        n_self = self.get_num_samples()
        as_adj = self.as_adj

        self_ = deepcopy(self)
        other_ = deepcopy(other)

        self_.edges_to_lst(inplace=True)
        other_.edges_to_lst(inplace=True)

        self_ = super(GraphDataPair, self_).__add__(other_)

        other_edges = other_.edges + n_self
        self_.edges = torch.cat((self_.edges, other_edges), dim=0)

        if as_adj:
            self_.edges_to_adj(inplace=True)

        del other_
        return self_
