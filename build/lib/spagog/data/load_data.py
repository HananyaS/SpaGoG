import os
import json

import numpy as np
import pandas as pd
from pkg_resources import resource_filename
from sklearn.model_selection import train_test_split


def load_data(dataset_name: str):
    valid_tab_datasets = os.listdir(resource_filename(__name__, "Tabular"))
    valid_graph_datasets = os.listdir(resource_filename(__name__, "Graph"))

    assert dataset_name in valid_tab_datasets + valid_graph_datasets, f"Dataset {dataset_name} not found." \
                                                                      f" Valid datasets are" \
                                                                      f" {valid_tab_datasets + valid_graph_datasets}"

    if dataset_name in valid_tab_datasets:
        return load_tabular_data(dataset_name)

    return load_graph_data(dataset_name)


def load_tabular_data(dataset_name: str):
    train = pd.read_csv(resource_filename(__name__, f"Tabular/{dataset_name}/processed/train.csv"), index_col=0)
    test = pd.read_csv(resource_filename(__name__, f"Tabular/{dataset_name}/processed/test.csv"), index_col=0)
    val = pd.read_csv(resource_filename(__name__, f"Tabular/{dataset_name}/processed/val.csv"), index_col=0)

    with open(resource_filename(__name__, f"Tabular/{dataset_name}/processed/config.json"), "r") as f:
        conf = json.load(f)
        target_col = conf["target_col"]

    return train, val, test, target_col, "tabular"


def load_graph_data(dataset_name: str):
    train = pd.read_csv(resource_filename(__name__, f"Graph/{dataset_name}/processed/train.csv"), index_col=0)
    test = pd.read_csv(resource_filename(__name__, f"Graph/{dataset_name}/processed/test.csv"), index_col=0)
    val = pd.read_csv(resource_filename(__name__, f"Graph/{dataset_name}/processed/val.csv"), index_col=0)
    edges = pd.read_csv(resource_filename(__name__, f"Graph/{dataset_name}/processed/edge_index.csv"), index_col=0)

    with open(resource_filename(__name__, f"Graph/{dataset_name}/processed/config.json"), "r") as f:
        conf = json.load(f)
        target_col = conf["target_col"]

    return train, val, test, edges, target_col, "graph"


def split_X_y(data, target_col):
    Y = data[target_col]
    X = data.drop(target_col, axis=1)

    return X, Y


def get_folds(data, target_col, kfolds):
    X, y = split_X_y(data, target_col)

    splits = []

    for f in range(kfolds):
        all_idx = np.arange(X.shape[0])

        train_idx, test_idx, *_ = train_test_split(
            all_idx,
            y,
            test_size=1/kfolds,
            random_state=f,
            shuffle=True,
        )

        try:
            train_idx, val_idx, *_ = train_test_split(
                train_idx,
                y.iloc[train_idx],
                test_size=0.2,
                random_state=f,
                shuffle=True,
                stratify=y.iloc[train_idx],
            )

        except Exception:
            train_idx, val_idx, *_ = train_test_split(
                train_idx,
                y.iloc[train_idx],
                test_size=0.2,
                random_state=f,
                shuffle=True,
            )

            # train_X, val_X, train_Y, val_Y = train_test_split(
            #     train_X,
            #     train_Y,
            #     test_size=0.2,
            #     random_state=f,
            #     shuffle=True,
            # )

        splits.append((train_idx, val_idx, test_idx))

    return splits
