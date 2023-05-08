import os
import time
import json
import warnings
import argparse

import torch
import pandas as pd
import numpy as np

from experiments import run_gc, run_gnc, run_gc_nc, get_default_params_file, get_tab_data

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
str2bool = lambda x: x.lower() in ["true", "t", 1]

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", type=str, default="SSCFLOW-iris")
parser.add_argument("--model", type=str, default="gc", help="gc / gnc / gc+nc")
parser.add_argument("--use_existence_cols", type=str2bool, default=True)
parser.add_argument("--dist", type=str, default="heur_dist")
parser.add_argument("--k", type=str, default=3)
parser.add_argument("--n_feats", type=int, default=30)
parser.add_argument("--verbose", type=int, default=1)

args = parser.parse_args()

assert args.embedding_layer in ['one_before_last', 'first', 'mid']
assert args.clf_from in ['gc', 'nc']
assert args.model in ["gc", "gnc", "gc+nc"]
assert args.verbose in [0, 1, 2]

assert args.dataset is not None, "Please set a dataset"

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
        **spec_params
):
    assert not evaluate_metrics or test_Y is not None, "Please provide test_Y to evaluate metrics"
    assert model in ["gc", "gnc", "gc+nc"], "Please provide a valid model {gc, gnc, gc+nc}}"

    is_graph = edges is not None

    params = get_default_params_file(model)

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

    tab_dataset = get_tab_data(
        train_X=train_X,
        train_Y=train_Y,
        test_X=test_X,
        test_Y=test_Y,
        val_X=val_X,
        val_Y=val_Y,
        name=dataset_name,
        use_existence_cols=params.get("use_existence_cols", args.use_existence_cols),
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
        verbose=args.verbose == 2,
        evaluate_metrics=evaluate_metrics,
        probs=probs,
        to_numpy=to_numpy,
    )

    if args.verbose == 2:
        print()

    if args.verbose > 0:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Results on {args.dataset.upper()} with {args.model.upper()}:")
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

    if evaluate_metrics:
        return y_test, res_cache

    return y_test


if __name__ == "__main__":
    train_all = pd.read_csv("data/PAPERS/SSCFLOW/Iris/processed/train.csv")
    test_all = pd.read_csv("data/PAPERS/SSCFLOW/Iris/processed/test.csv")
    val_all = pd.read_csv("data/PAPERS/SSCFLOW/Iris/processed/val.csv")

    target_col = "tag"

    train_Y = train_all[target_col]
    train_X = train_all.drop(target_col, axis=1)

    test_Y = test_all[target_col]
    test_X = test_all.drop(target_col, axis=1)

    val_Y = val_all[target_col]
    val_X = val_all.drop(target_col, axis=1)

    if args.verbose > 0:
        _st = time.time()

    default_params_file = get_default_params_file(args.model)
    args.params = default_params_file

    with open(args.params, "r") as f:
        params = json.load(f)

    if args.model == "gnc" and args.gc_pretrain:
        gc_default_params_file = get_default_params_file("gc")
        with open(gc_default_params_file, "r") as f:
            gc_params = json.load(f)

        params["gc_params"] = gc_params

    results = gog_model(params=params, train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y, val_X=val_X,
                        val_Y=val_Y, model=args.model, evaluate_metrics=test_Y is not None,
                        feature_selection=args.n_feats, )

    if args.verbose > 0:
        _et = time.time()
        diff = _et - _st

        _sec = np.round(diff % 60, 2)
        _min = int(diff / 60)

        print(f"All program last {_min} minutes and {_sec} seconds.")
