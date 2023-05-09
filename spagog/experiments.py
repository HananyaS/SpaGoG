import pandas as pd
import torch

from .datasets.tabDataset import TabDataset
from .datasets.graphsDataset import GraphsDataset

from .models.graphClassification import ValuesAndGraphStructure as GC
from .models.graphNodeClassification import GraphNodeClassification as GNC
from .models.nodeClassification import NodeClassification

PROJECT_DIR = "."


def run_gc(
        tab_dataset: TabDataset,
        params: dict,
        early_stopping: int = 30,
        inter_sample_edges: torch.Tensor = None,
        verbose: bool = True,
        evaluate_metrics: bool = True,
        probs: bool = False,
        to_numpy: bool = True,
):
    graphs_dataset, *_ = GraphsDataset.from_tab(
        tab_data=tab_dataset,
        inter_sample_edges=inter_sample_edges,
        knn_kwargs={
            "distance": params.get("distance", "heur_dist"),
            "k": params.get("k", 3),
        },
        calc_intra_edges=True,
    )

    train_loader = graphs_dataset.train.to_loader(batch_size=params["batch_size"])
    val_loader = graphs_dataset.val.to_loader(batch_size=params["batch_size"])
    test_loader = graphs_dataset.test.to_loader(batch_size=params["batch_size"])

    RECEIVED_PARAMS = {
        "preweight": params["preweight"],
        "layer_1": params["layer_1"],
        "layer_2": params["layer_2"],
        "activation": params["activation"],
        "dropout": params["dropout"],
    }

    model = GC(
        input_example=graphs_dataset,
        RECEIVED_PARAMS=RECEIVED_PARAMS,
        init_weights=params.get("init", "xavier_normal"),
        embedding_layer="first",
    )

    cache = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        lr=params.get("lr", 0.01),
        early_stopping_patience=early_stopping,
        n_epochs=600,
        verbose=verbose,
        weight_decay=params.get("weight_decay", 0.001),
        dataset_name=tab_dataset.name,
    )

    y_test = model.predict(test_loader, probs=False, to_numpy=to_numpy)

    if evaluate_metrics and verbose == 1:
        print(f"Test accuracy: {(test_loader.dataset.gdp.Y.flatten() == y_test).float().mean():.4f}")

    if probs:
        y_test = model.predict(test_loader, probs=True, to_numpy=to_numpy)

    return y_test, cache


def run_gnc(
        tab_dataset: TabDataset,
        params: dict,
        early_stopping: int = 30,
        inter_sample_edges: torch.Tensor = None,
        verbose: bool = True,
        evaluate_metrics: bool = True,
        probs: bool = False,
        to_numpy: bool = True,
):
    graphs_dataset, inter_samples_edges, masks = GraphsDataset.from_tab(
        tab_data=tab_dataset,
        knn_kwargs={
            "distance": params.get("distance", "heur_dist"),
            "k": params.get("k", 30),
        },
        inter_sample_edges=inter_sample_edges,
        calc_intra_edges=True
    )

    train_mask, val_mask, test_mask = masks

    nc_h_layers = [int(params["nc_layer_1"]), int(params["nc_layer_2"])]
    nc_dropouts = [float(params["nc_dropout_1"]), float(params["nc_dropout_2"])]

    nc_activations = [
        str(params["nc_activation_1"]),
        str(params["nc_activation_2"]),
    ]

    # if gc_pretrain:
    if params["gc_pretrain"]:
        gc_params = params["gc_params"]

        train_loader = graphs_dataset.train.to_loader(
            batch_size=gc_params["batch_size"]
        )
        val_loader = graphs_dataset.val.to_loader(batch_size=gc_params["batch_size"])
        test_loader = graphs_dataset.test.to_loader(batch_size=gc_params["batch_size"])

        RECEIVED_PARAMS = {
            "preweight": gc_params["preweight"],
            "layer_1": gc_params["layer_1"],
            "layer_2": gc_params["layer_2"],
            "activation": gc_params["activation"],
            "dropout": gc_params["dropout"],
        }

        gc_model = GC(
            input_example=graphs_dataset,
            RECEIVED_PARAMS=RECEIVED_PARAMS,
            init_weights=gc_params.get("init", "xavier_normal"),
            embedding_layer=params["embedding_layer"]  # embedding_layer,
        )

        _ = gc_model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            lr=gc_params.get("lr", 0.01),
            early_stopping_patience=30,
            n_epochs=600,
            verbose=verbose,
            weight_decay=gc_params.get("weight_decay", 0.001),
            dataset_name=tab_dataset.name,
        )

    else:
        gc_model = None

    model = GNC(
        gc_kwargs={
            "input_example": graphs_dataset,
            "RECEIVED_PARAMS": {
                "preweight": params["gc_preweight"],
                "layer_1": params["gc_layer_1"],
                "layer_2": params["gc_layer_2"],
                "activation": params["gc_activation"],
                "dropout": params["gc_dropout"],
            },
            "embedding_layer": params["embedding_layer"]  # embedding_layer,
        },
        nc_kwargs={
            "h_layers": nc_h_layers,
            "dropouts": nc_dropouts,
            "activations": nc_activations,
        },
        gc_model=gc_model,
    )

    cache, all_graphs_loader = model.fit(
        graphs_dataset,
        inter_samples_edges,
        train_mask,
        val_mask,
        test_mask,
        n_epochs=10000,
        gc_lr=params["gc_lr"],
        nc_lr=params["nc_lr"],
        gc_weight_decay=params["gc_weight_decay"],
        nc_weight_decay=params["nc_weight_decay"],
        alpha=params["alpha"],
        batch_size=params.get("batch_size", 10),
        early_stopping_patience=early_stopping,
        verbose=verbose,
        clf_from=params["clf_from"]  # clf_from
    )

    y_test = model.predict(all_graphs_loader, inter_samples_edges.T, test_mask, clf_from=params["clf_from"], probs=False,
                           to_numpy=to_numpy)

    if evaluate_metrics and verbose == 1:
        print(f"Test accuracy: {(test_loader.dataset.gdp.Y.flatten() == y_test).float().mean():.4f}")

    if probs:
        y_test = model.predict(all_graphs_loader, inter_samples_edges.T, test_mask, clf_from=params["clf_from"], probs=True,
                               to_numpy=to_numpy)

    return y_test, cache


def run_gc_nc(
        tab_dataset: TabDataset,
        params: dict,
        early_stopping: int = 30,
        inter_sample_edges: torch.Tensor = None,
        verbose: bool = True,
        evaluate_metrics: bool = True,
        probs: bool = False,
        to_numpy: bool = True,
):
    graphs_dataset, inter_sample_edges, masks = GraphsDataset.from_tab(
        tab_data=tab_dataset,
        knn_kwargs={
            "distance": params.get("distance", "heur_dist"),
            "k": params.get("k", 30),
        },
        inter_sample_edges=inter_sample_edges,
        calc_intra_edges=True
    )

    train_graphs_loader = graphs_dataset.train.to_loader(
        batch_size=params["batch_size"]
    )

    val_graphs_loader = graphs_dataset.val.to_loader(batch_size=params["batch_size"])
    test_graphs_loader = graphs_dataset.test.to_loader(batch_size=params["batch_size"])

    GC_RECEIVED_PARAMS = {
        "preweight": params["gc_preweight"],
        "layer_1": params["gc_layer_1"],
        "layer_2": params["gc_layer_2"],
        "activation": params["gc_activation"],
        "dropout": params["gc_dropout"],
    }

    gc_model = GC(
        input_example=graphs_dataset,
        RECEIVED_PARAMS=GC_RECEIVED_PARAMS,
        init_weights=params.get("init", "xavier_normal"),
        embedding_layer=params["embedding_layer"]  # embedding_layer
    )

    _ = gc_model.fit(
        train_loader=train_graphs_loader,
        val_loader=val_graphs_loader,
        test_loader=test_graphs_loader,
        lr=params["gc_lr"],
        early_stopping_patience=early_stopping,
        n_epochs=600,
        verbose=verbose,
        weight_decay=params.get("gc_weight_decay", 0),
        dataset_name=tab_dataset.name,
    )

    def extract_embeddings(graphs_loader):
        first_batch = True

        for graphs_data in graphs_loader:
            input_data, _ = gc_model.transform_input(graphs_data)
            output_ = gc_model.forward_one_before_last_layer(*input_data)

            if first_batch:
                attr_embeddings = output_
                gc_output_ = gc_model.forward_last_layer(output_)
                gc_output = gc_output_
                first_batch = False

            else:
                attr_embeddings = torch.cat((attr_embeddings, output_), 0)
                gc_output_ = gc_model.forward_last_layer(output_)
                gc_output = torch.cat((gc_output, gc_output_), 0)

        return attr_embeddings

    train_embeddings = extract_embeddings(train_graphs_loader)
    val_embeddings = extract_embeddings(val_graphs_loader)
    test_embeddings = extract_embeddings(test_graphs_loader)

    nc_h_layers = [int(params["nc_layer_1"]), int(params["nc_layer_2"])]
    nc_dropouts = [float(params["nc_dropout_1"]), float(params["nc_dropout_2"])]

    nc_activations = [
        str(params["nc_activation_1"]),
        str(params["nc_activation_2"]),
    ]

    nc_model = NodeClassification(
        h_layers=nc_h_layers,
        dropouts=nc_dropouts,
        activations=nc_activations,
        n_classes=gc_model.num_classes,
        n_features=train_embeddings.shape[1],
        device=torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu"),
    )

    inter_samples_edges_train = []

    for i, j in inter_sample_edges:
        if i.item() <= masks[0][-1] and j.item() <= masks[0][-1]:
            inter_samples_edges_train.append([i, j])

    inter_samples_edges_train = torch.LongTensor(inter_samples_edges_train)

    all_embeddings = torch.cat([train_embeddings, val_embeddings, test_embeddings])

    train_graph = [
        (
            (train_embeddings, inter_samples_edges_train.T, masks[0]),
            tab_dataset.train.Y,
        )
    ]

    val_graph = [
        (
            (all_embeddings, inter_sample_edges.T, masks[1]),
            tab_dataset.val.Y,
        )
    ]

    test_graph = [
        (
            (all_embeddings, inter_sample_edges.T, masks[2]),
            tab_dataset.test.Y,
        )
    ]

    cache = nc_model.fit(
        train_loader=train_graph,
        val_loader=val_graph,
        test_loader=test_graph,
        lr=params["nc_lr"],
        n_epochs=600,
        early_stopping_patience=early_stopping,
        weight_decay=params.get("nc_weight_decay", 0),
        verbose=verbose,
        dataset_name=graphs_dataset.name.replace("graphs", "graph"),
        labels_from_loader=lambda x: x[0][1],
    )

    y_test = nc_model.predict(test_graph, probs=False, to_numpy=to_numpy)

    if evaluate_metrics and verbose == 1:
        print(f"Test accuracy: {(test_graph[0][1] == y_test).float().mean():.4f}")

    if probs:
        y_test = nc_model.predict(test_graph, probs=True, to_numpy=to_numpy)

    return y_test, cache


def get_default_params_file(model):
    return f"default_params/{model}.json"


def get_tab_data(
        train_X: pd.DataFrame,
        train_Y: pd.DataFrame,
        test_X: pd.DataFrame,
        test_Y: pd.DataFrame = None,
        val_X: pd.DataFrame = None,
        val_Y: pd.DataFrame = None,
        name: str = "",
        use_existence_cols: bool = True,
        *args,
        **kwargs
):
    tab_dataset = TabDataset.load(train_X=train_X, train_Y=train_Y, test_X=test_X, test_Y=test_Y, val_X=val_X,
                                  val_Y=val_Y, name=name,
                                  *args,
                                  **kwargs)

    if use_existence_cols:
        tab_dataset.add_existence_cols()

    return tab_dataset
