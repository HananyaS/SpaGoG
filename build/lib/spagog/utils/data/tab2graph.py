import torch
import numpy as np
import pandas as pd

from ...datasets.tabDataPair import TabDataPair

from ..gfp import GFP
from ..knn import KNN

from copy import deepcopy
from typing import Union, Type

from itertools import combinations


def tab2graphs(
        tab_data,
        store_as_adj: bool = True,
        name: str = None,
        include_edge_weights: bool = False,
        edge_weights_method: str = "corr",
        fill_data_method: str = "gfp",
        knn_kwargs: dict = {"distance": "euclidian"},
        gfp_kwargs: dict = {},
        val_mask: list = None,
        test_mask: list = None,
        inter_sample_edges: torch.Tensor = None,
        calc_intra_edges: bool = True,
        f2m: bool = False,
):
    assert not include_edge_weights or edge_weights_method in ["corr", None]
    assert fill_data_method in ["gfp", "zeros"]

    if isinstance(tab_data, TabDataPair):
        tab_data_ = deepcopy(tab_data.X)

    else:
        tab_data_ = deepcopy(tab_data)

    if fill_data_method == "gfp":
        imputed_data_, knn_inter_sample_edges, mean_dist_all, mean_dist_neigh, num_nulls, dists = fill_data_gfp(
            tab_data_,
            knn_kwargs=knn_kwargs,
            gfp_kwargs=gfp_kwargs,
            val_mask=val_mask if name not in ["cora", "citeseer", "pubmed"] else None,
            test_mask=test_mask if name not in ["cora", "citeseer", "pubmed"] else None,
            inter_sample_edges=inter_sample_edges,
            calc_knn=inter_sample_edges is None,
        )

    elif fill_data_method == "zeros":
        imputed_data_ = torch.nan_to_num(tab_data_)
        inter_sample_edges = None

    else:
        raise NotImplementedError(f"Method {fill_data_method} is not supported!")

    if name not in ["cora", "citeseer", "pubmed"]:
        inter_sample_edges = knn_inter_sample_edges

    if include_edge_weights and name not in ["cora", "citeseer", "pubmed"]:
        if edge_weights_method == "corr":
            edge_weights = abs(
                pd.DataFrame(tab_data_.cpu().detach().numpy()).corr().fillna(0).values
            )

        else:
            raise NotImplementedError

    elif include_edge_weights:
        edge_weights = np.ones((tab_data_.shape[1], tab_data_.shape[1]))

    X_list = list(imputed_data_)

    if calc_intra_edges:
        if f2m:
            edge_list = (
                torch.from_numpy(edge_weights)
                .unsqueeze(0)
                .repeat([tab_data_.shape[0], 1, 1])
            )

            masks = torch.repeat_interleave(
                ~torch.isnan(tab_data_).unsqueeze(-1), tab_data_.shape[1], dim=2
            ).long()
            edge_list = edge_list * masks
            edge_list = list(edge_list)

        else:
            edge_list = list(
                map(
                    lambda s: torch.from_numpy(
                        lst_to_mat(
                            np.array(
                                list(combinations(np.where(s.cpu() == s.cpu())[0], r=2))
                            ),
                            len(s),
                            edge_weights,
                        )
                    ),
                    list(tab_data_),
                )
            )

    else:
        edge_list = None

    if isinstance(tab_data, TabDataPair) and tab_data.Y is not None:
        Y_list = list(tab_data.Y)

    else:
        Y_list = None

    if name is None:
        name = f"{tab_data_.name} - Graph"

    kwargs = {
        "X_list": X_list,
        "edges_list": edge_list,
        "Y_list": Y_list,
        "name": name,
        "include_edge_weights": include_edge_weights,
        "given_as_adj": True,
        "store_as_adj": store_as_adj,
        "normalize": False,
        "normalization_params": None,
        "shuffle": False,
        "add_existence_cols": False,
        "inter_sample_edges": inter_sample_edges,
        "edge_weights": edge_weights,
    }

    if isinstance(tab_data, TabDataPair):
        to_return = (
            kwargs,
            tab_data.normalized,
            tab_data.norm_params,
            (tab_data.existence_cols is not None)
            and (len(tab_data.existence_cols) > 0),
        )

    else:
        to_return = (
            kwargs,
            False,
            None,
            True,
        )

    del tab_data_

    return to_return


def lst_to_mat(z, n, edge_weights=None, numpy=True):
    if numpy:
        if edge_weights is None:
            m = np.zeros(shape=(n, n))
            if 0 not in z.shape:
                z = np.concatenate((z, np.flip(z)))
                m[z[:, 0], z[:, 1]] = 1
            return m

        m = np.zeros(shape=(n, n))
        if 0 not in z.shape:
            z = np.concatenate((z, np.flip(z)))
            m[z[:, 0], z[:, 1]] = edge_weights[z[:, 0], z[:, 1]]
        return m

    m = torch.zeros((n, n))
    if 0 not in z.shape:
        m[z[0, :], z[1, :]] = edge_weights[z[0, :], z[1, :]]
    return m


def get_knn_adj(
        tab_data: torch.Tensor,
        val_mask: list = None,
        test_mask: list = None,
        **kwargs,
):
    knn = KNN(**kwargs)
    edges = knn.get_edges(
        tab_data, as_adj=False, val_mask=val_mask, test_mask=test_mask
    )

    return edges


def edges_from_sample(
        sample: Union[Type[np.ndarray], Type[torch.Tensor]],
        edge_weights: Union[Type[torch.Tensor], Type[np.ndarray]] = None,
):
    edge_weights = (
        edge_weights
        if edge_weights is not None
        else np.ones((sample.shape[1], sample.shape[1]))
    )

    adj = torch.zeros(len(sample), len(sample), dtype=torch.float)

    features_existence = np.where((sample == sample).cpu())[0]

    for i, j in combinations(features_existence, r=2):
        adj[i, j] = edge_weights[i, j]
        adj[j, i] = edge_weights[i, j]

    return adj


def fill_data_gfp(
        tab_data_,
        knn_kwargs: dict,
        gfp_kwargs: dict = {},
        val_mask: list = None,
        test_mask: list = None,
        inter_sample_edges: torch.Tensor = None,
        calc_knn: bool = True,
):
    assert calc_knn or inter_sample_edges is not None

    if calc_knn:
        knn_inter_sample_edges, dists = get_knn_adj(
            tab_data_, val_mask=val_mask, test_mask=test_mask, **knn_kwargs
        )

        mean_dist_all = np.mean(dists, 1)
        mean_dist_neigh = np.sort(dists, 1)[:, 1: knn_kwargs["k"] + 1].mean(1)
        num_nulls = tab_data_[:, :tab_data_.shape[1] // 2].isnan().sum(1)

    else:
        mean_dist_all = None
        mean_dist_neigh = None
        num_nulls = None
        dists = None
        knn_inter_sample_edges = None

    if inter_sample_edges is None:
        inter_sample_edges = knn_inter_sample_edges

    gfp = GFP(**gfp_kwargs)

    imputed_data_ = gfp.prop(tab_data_, inter_sample_edges)

    return imputed_data_, knn_inter_sample_edges, mean_dist_all, mean_dist_neigh, num_nulls, dists
