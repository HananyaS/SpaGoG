import os
import time
import argparse
import warnings

import torch
import pandas as pd
import numpy as np

from experiments import run_gc, run_gnc, run_gc_nc, get_tab_data
from default_params.load_params import load_params
from data.load_data import load_data, split_X_y, get_folds

warnings.filterwarnings("ignore")

PROJECT_DIR = "."
os.chdir(PROJECT_DIR)

parser = argparse.ArgumentParser()

parser.add_argument("--model", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--feature_selection", type=int, default=None)
parser.add_argument("--verbosity", type=int, default=1)
parser.add_argument("--kfolds", type=int, default=None)

args = parser.parse_args()

assert args.kfolds is None or args.kfolds > 1
args.model = args.model.lower()


def gog_model(
        model: str,
        train_X: pd.DataFrame,
        train_y: pd.DataFrame,
        test_X: pd.DataFrame,
        test_y: pd.DataFrame = None,
        val_X: pd.DataFrame = None,
        val_y: pd.DataFrame = None,
        evaluate_metrics: bool = True,
        dataset_name: str = "",
        feature_selection: int = 100,
        edges: pd.DataFrame = None,
        probs: bool = False,
        to_numpy: bool = False,
        verbosity: int = 0,
        **spec_params
):
    assert not evaluate_metrics or test_y is not None, "Please provide test_Y to evaluate metrics"
    assert model in ["gc", "gnc", "gc+nc"], "Please provide a valid model {gc, gnc, gc+nc}}"
    assert verbosity in [0, 1, 2], "Please provide a valid verbosity level {0, 1, 2}"

    if verbosity > 0:
        _st = time.time()

    is_graph = edges is not None

    params = load_params(model)

    if model == "gnc" and params.get("gc_pretrain", False):
        gc_params = load_params("gc")

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
        train_Y=train_y,
        test_X=test_X,
        test_Y=test_y,
        val_X=val_X,
        val_Y=val_y,
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

    if verbosity > 0 and args.kfolds is None:
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f"Results on {dataset_name.upper() if dataset_name != '' else 'dataset'} with {model.upper()}:")

        print("\tAccuracy:")
        print(
            f"\t\tThreshold:\t{round(res_cache['Acc Threshold'], 3)}\n"
            f"\t\tTrain:\t{round(res_cache['Train Acc'], 3)}\n"
            f"\t\tVal:\t{round(res_cache['Val Acc'], 3)}"
        )

        if evaluate_metrics:
            print(
                f"\t\tTest:\t{round(res_cache['Test Acc'], 3)}"
            )

        print("\tF1:")
        print(
            f"\t\tThreshold:\t{round(res_cache['F1 Threshold'], 3)}\n"
            f"\t\tTrain:\t{round(res_cache['Train F1'], 3)}\n"
            f"\t\tVal:\t{round(res_cache['Val F1'], 3)}"
        )

        if evaluate_metrics:
            print(
                f"\t\tTest:\t{round(res_cache['Test F1'], 3)}"
            )

        print("\tAUC:")
        print(
            f"\t\tTrain:\t{round(res_cache['Train AUC'], 3)}\n"
            f"\t\tVal:\t{round(res_cache['Val AUC'], 3)}"
        )

        if evaluate_metrics:
            print(
                f"\t\tTest:\t{round(res_cache['Test AUC'], 3)}"
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
    data = load_data(args.dataset)

    if data[-1] == "tabular":
        train, val, test, target_col = data[:-1]
        edges = None

    else:
        train, val, test, edges, target_col = data[:-1]

    if args.kfolds is not None and args.kfolds > 1:
        all_data = pd.concat([train, val, test])
        splits_idx = get_folds(all_data, target_col, args.kfolds)

        for fold, (train_idx, test_idx, val_idx) in enumerate(splits_idx):
            if args.verbosity > 0:
                print(f"Running fold {fold + 1} out of {args.kfolds}...")

            train = all_data.iloc[train_idx]
            test = all_data.iloc[test_idx]
            val = all_data.iloc[val_idx]

            train_X, train_Y = split_X_y(train, target_col)
            test_X, test_y = split_X_y(test, target_col)
            val_X, val_y = split_X_y(val, target_col)

            _, fold_results = gog_model(train_X=train_X, train_y=train_Y, test_X=test_X, test_y=test_y, val_X=val_X,
                                        val_y=val_y, model=args.model, evaluate_metrics=test_y is not None,
                                        verbosity=args.verbosity, feature_selection=args.feature_selection, edges=edges)

            if fold == 0:
                results_all_folds = {k: [v] for k, v in fold_results.items()}

            else:
                for k, v in fold_results.items():
                    results_all_folds[k].append(v)

        for k, v in results_all_folds.items():
            results_all_folds[k] = np.mean(v), np.std(v)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        print(
            f"Results on {args.dataset.upper() if args.dataset != '' else 'dataset'} with {args.model.upper()} ({args.kfolds} cross-validation):")

        print("\tAccuracy:")
        print(
            f"\t\tTrain:\t{round(results_all_folds['Train Acc'][0], 3)} ± {round(results_all_folds['Train Acc'][1], 3)}\n"
            f"\t\tVal:\t{round(results_all_folds['Val Acc'][0], 3)} ± {round(results_all_folds['Val Acc'][1], 3)}\n"
            f"\t\tTest:\t{round(results_all_folds['Test Acc'][0], 3)} ± {round(results_all_folds['Test Acc'][1], 3)}"
        )

        print("\tF1:")
        print(
            f"\t\tTrain:\t{round(results_all_folds['Train F1'][0], 3)} ± {round(results_all_folds['Train F1'][1], 3)}\n"
            f"\t\tVal:\t{round(results_all_folds['Val F1'][0], 3)} ± {round(results_all_folds['Val F1'][1], 3)}\n"
            f"\t\tTest:\t{round(results_all_folds['Test F1'][0], 3)} ± {round(results_all_folds['Test F1'][1], 3)}"
        )

        print("\tAUC:")
        print(
            f"\t\tTrain:\t{round(results_all_folds['Train AUC'][0], 3)} ± {round(results_all_folds['Train AUC'][1], 3)}\n"
            f"\t\tVal:\t{round(results_all_folds['Val AUC'][0], 3)} ± {round(results_all_folds['Val AUC'][1], 3)}\n"
            f"\t\tTest:\t{round(results_all_folds['Test AUC'][0], 3)} ± {round(results_all_folds['Test AUC'][1], 3)}"
        )

    else:
        train_X, train_Y = split_X_y(train, target_col)
        test_X, test_y = split_X_y(test, target_col)
        val_X, val_y = split_X_y(val, target_col)

        results = gog_model(train_X=train_X, train_y=train_Y, test_X=test_X, test_y=test_y, val_X=val_X,
                            val_y=val_y, model=args.model, evaluate_metrics=test_y is not None,
                            verbosity=args.verbosity, feature_selection=args.feature_selection, edges=edges)
