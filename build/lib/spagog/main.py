import os
import time
import json
import warnings

import torch
import pandas as pd
import numpy as np

from .experiments import run_gc, run_gnc, run_gc_nc, get_default_params_file, get_tab_data

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)


def gog_model(
        model: str,
        train_X: pd.DataFrame,
        train_Y: pd.DataFrame,
        test_X: pd.DataFrame,
        test_Y: pd.DataFrame = None,
        val_X: pd.DataFrame = None,
        val_Y: pd.DataFrame = None,
        evaluate_metrics: bool = True,
        dataset_name: str = "",
        feature_selection: int = 100,
        edges: pd.DataFrame = None,
        probs: bool = False,
        to_numpy: bool = False,
        verbosity: int = 0,
        **spec_params
):
    assert not evaluate_metrics or test_Y is not None, "Please provide test_Y to evaluate metrics"
    assert model in ["gc", "gnc", "gc+nc"], "Please provide a valid model {gc, gnc, gc+nc}}"
    assert verbosity in [0, 1, 2], "Please provide a valid verbosity level {0, 1, 2}"

    if verbosity > 0:
        _st = time.time()

    is_graph = edges is not None

    params = get_default_params_file(model)

    with open(params, "r") as f:
        params = json.load(f)

    if model == "gnc" and params.get("gc_pretrain", False):
        gc_default_params_file = get_default_params_file("gc")
        with open(gc_default_params_file, "r") as f:
            gc_params = json.load(f)

        params["gc_params"] = gc_params

    for k, v in spec_params.items():
        if k in params.keys():
            params[k] = v

        elif "gc_params" in params.keys() and k in params["gc_params"].keys():
            params["gc_params"][k] = v

    assert "embedding_layer" not in params.keys() or params["embedding_layer"] in ['one_before_last', 'first', 'mid']
    assert "gc_params" not in params.keys() or "embedding_layer" not in params["gc_params"].keys() or \
           params["gc_params"]["embedding_layer"] in ['one_before_last', 'first', 'mid']
    assert "clf_from" not in params.keys() or params["clf_from"] in ['gc', 'nc']

    tab_dataset = get_tab_data(
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        test_Y=test_Y,
        val_X=val_X,
        val_Y=val_Y,
        name=dataset_name,
        use_existence_cols=params["use_existence_cols"],
        feature_selection=feature_selection,
    )

    inter_sample_edges = torch.from_numpy(edges.values).long() if is_graph else None

    if model == "gc":
        run_func = run_gc
    elif model == "gnc":
        run_func = run_gnc
    else:
        run_func = run_gc_nc

    y_test, res_cache = run_func(
        tab_dataset,
        params,
        inter_sample_edges=inter_sample_edges,
        verbose=verbosity == 2,
        evaluate_metrics=evaluate_metrics,
        probs=probs,
        to_numpy=to_numpy,
    )

    if verbosity == 2:
        print()

    if verbosity > 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Results on {dataset_name.upper() if dataset_name != '' else 'dataset'} with {model.upper()}:")
        print(f"\tN epochs:\t{res_cache['learning_epochs']}")
        print("\tAccuracy:")
        print(
            f"\t\tThreshold:\t{res_cache['Acc Threshold']}\n"
            f"\t\tTrain:\t{res_cache['Train Acc']}\n"
            f"\t\tVal:\t{res_cache['Val Acc']}"
        )

        if evaluate_metrics:
            print(
                f"\t\tTest:\t{res_cache['Test Acc']}"
            )

        print("\tF1:")
        print(
            f"\t\tThreshold:\t{res_cache['F1 Threshold']}\n"
            f"\t\tTrain:\t{res_cache['Train F1']}\n"
            f"\t\tVal:\t{res_cache['Val F1']}"
        )

        if evaluate_metrics:
            print(
                f"\t\tTest:\t{res_cache['Test F1']}"
            )

        print("\tAUC:")
        print(
            f"\t\tTrain:\t{res_cache['Train AUC']}\n"
            f"\t\tVal:\t{res_cache['Val AUC']}"
        )

        if evaluate_metrics:
            print(
                f"\t\tTest:\t{res_cache['Test AUC']}"
            )

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    if verbosity > 0:
        _et = time.time()
        diff = _et - _st

        _sec = np.round(diff % 60, 2)
        _min = int(diff / 60)

        print(f"All program last {_min} minutes and {_sec} seconds.")

    if evaluate_metrics:
        return y_test, res_cache

    return y_test


if __name__ == "__main__":
    train_all = pd.read_csv("data/Tabular/Ecoli/processed/train.csv")
    val_all = pd.read_csv("data/Tabular/Ecoli/processed/val.csv")
    test_all = pd.read_csv("data/Tabular/Ecoli/processed/test.csv")

    target_col = "class"

    train_Y = train_all[target_col]
    train_X = train_all.drop(target_col, axis=1)

    test_Y = test_all[target_col]
    test_X = test_all.drop(target_col, axis=1)

    val_Y = val_all[target_col]
    val_X = val_all.drop(target_col, axis=1)

    results = gog_model(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y, val_X=val_X,
                        val_Y=val_Y, model="gc", evaluate_metrics=test_Y is not None,
                        verbosity=1)
