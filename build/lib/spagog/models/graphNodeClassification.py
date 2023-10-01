import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim

from ..datasets.graphsDataset import GraphsDataset

from ..utils.metrics.metrics import find_best_metrics_bin

MODEL = "gnc"
PROJECT_DIR = "."
sys.path.append(PROJECT_DIR)

from .abstractModel import AbstractModel
from .nodeClassification import NodeClassification
from .graphClassification import ValuesAndGraphStructure

from typing import Dict

from sklearn.metrics import roc_auc_score
from torch.nn.functional import one_hot

from ..utils.metrics.metrics import find_best_metrics_multi


class GraphNodeClassification(nn.Module, AbstractModel):
    def predict(
        self,
        graphs_loader,
        inter_samples_edges,
        mask,
        clf_from,
        probs,
        to_numpy=True,
        *args,
        **kwargs,
    ):
        final_output, gc_output = self(
            graphs_loader,
            inter_samples_edges,
            mask,
        )

        if clf_from == "gc":
            output = gc_output

        else:
            output = final_output

        if probs:
            output = torch.softmax(output, dim=1)

        else:
            output = torch.argmax(output, dim=1)

        if to_numpy:
            output = output.cpu().detach().numpy()

        return output

    def evaluate(self, *args, **kwargs):
        pass

    def __init__(
        self,
        gc_kwargs: Dict = {},
        nc_kwargs: Dict = {},
        gc_model: ValuesAndGraphStructure = None,
    ):
        super(GraphNodeClassification, self).__init__()
        if gc_model is not None:
            self.gc_model = gc_model
        else:
            self.gc_model = ValuesAndGraphStructure(**gc_kwargs)

        nc_kwargs["n_classes"] = self.gc_model.num_classes
        self.nc_model = NodeClassification(**nc_kwargs)
        self.bin = self.gc_model.num_classes == 2

        if gc_kwargs["embedding_layer"] == "first":
            self.batch_norm = nn.BatchNorm1d(num_features=self.gc_model.fc1.in_features)

        elif gc_kwargs["embedding_layer"] == "mid":
            self.batch_norm = nn.BatchNorm1d(num_features=self.gc_model.fc2.in_features)

        else:
            self.batch_norm = nn.BatchNorm1d(num_features=self.gc_model.fc3.in_features)

        assert self.gc_model.device == self.nc_model.device
        self.device = self.nc_model.device

    def __str__(self):
        return "Graph Node Classification"

    def forward(
        self,
        graphs_loader,
        inter_sample_edges,
        mask: list = None,
    ):
        first_batch = True

        for graphs_data in graphs_loader:
            [x, adj], _ = self.gc_model.transform_input(graphs_data)
            output_ = self.gc_model.forward_one_before_last_layer(x, adj)

            if first_batch:
                attr_embeddings = output_
                gc_output_ = self.gc_model.forward_last_layer(output_, adj)
                gc_output = gc_output_
                first_batch = False

            else:
                attr_embeddings = torch.cat((attr_embeddings, output_), 0)
                gc_output_ = self.gc_model.forward_last_layer(output_, adj)
                gc_output = torch.cat((gc_output, gc_output_), 0)

        gc_output = gc_output[mask]

        attr_embeddings = self.batch_norm(attr_embeddings)

        final_output = self.nc_model(
            attr_embeddings.cpu(), inter_sample_edges, mask
        ).to(self.device)

        return final_output, gc_output

    @staticmethod
    def _extract_from_tab(
        graphs_dataset: GraphsDataset,
        batch_size: int = 32,
    ):
        graphs_train_loader = graphs_dataset.get_train_data(
            as_loader=True, batch_size=batch_size
        )

        graphs_val_loader = graphs_dataset.get_val_data(
            as_loader=True, batch_size=batch_size
        )

        graphs_test_loader = graphs_dataset.get_test_data(
            as_loader=True, batch_size=batch_size
        )

        return (
            graphs_train_loader,
            graphs_val_loader,
            graphs_test_loader,
        )

    def fit(
        self,
        graphs_dataset: GraphsDataset,
        inter_samples_edges: torch.Tensor,
        train_mask: list,
        val_mask: list,
        test_mask: list,
        n_epochs: int = 100,
        batch_size: int = 32,
        optimizer: torch.optim = optim.Adam,
        alpha: float = 2,
        early_stopping: int = -1,
        verbose: bool = False,
        clf_from: str = "nc",
        gc_lr: float = 0.001,
        gc_weight_decay: float = 0,
        nc_lr: float = 0.001,
        nc_weight_decay: float = 0,
        find_beta: bool = True,
    ):

        given_test_y = graphs_dataset.test.Y is not None

        if val_mask is None and early_stopping > 0:
            print("Early stopping is off because val_loader is None")
            early_stopping = -1

        elif early_stopping > 0:
            c = 0
            min_val_loss = np.inf

        (
            graphs_train_loader,
            graphs_val_loader,
            graphs_test_loader,
        ) = self._extract_from_tab(graphs_dataset, batch_size)

        all_graphs_loader = graphs_dataset.get_all_data_loader(batch_size=batch_size)

        inter_samples_edges_train = []

        for ind, (i, j) in enumerate(inter_samples_edges):
            if i.item() <= train_mask[-1] and j.item() <= train_mask[-1]:
                inter_samples_edges_train.append([i, j])

        inter_samples_edges_train = torch.LongTensor(inter_samples_edges_train)

        inter_samples_edges = inter_samples_edges.T
        inter_samples_edges_train = inter_samples_edges_train.T

        self.gc_model.train()
        self.nc_model.train()

        train_final_losses, val_final_losses = [], []
        train_gc_losses, val_gc_losses = [], []
        train_total_losses, val_total_losses = [], []
        train_final_aucs, val_final_aucs = [], []
        train_gc_aucs, val_gc_aucs = [], []

        train_gc_final_loss_rel, val_final_gc_loss_rel = [], []

        gc_optimizer = optimizer(
            self.gc_model.parameters(), lr=gc_lr, weight_decay=gc_weight_decay
        )

        nc_optimizer = optimizer(
            list(self.nc_model.parameters()) + list(self.batch_norm.parameters()),
            lr=nc_lr,
            weight_decay=nc_weight_decay,
        )

        n_classes = all_graphs_loader.dataset.gdp.num_classes

        train_labels = all_graphs_loader.dataset.gdp.Y[train_mask]
        val_labels = all_graphs_loader.dataset.gdp.Y[val_mask]
        test_labels = all_graphs_loader.dataset.gdp.Y[test_mask]

        for epoch in range(1, n_epochs + 1):
            self.train()

            gc_optimizer.zero_grad()
            nc_optimizer.zero_grad()

            final_output, gc_output = self(
                graphs_train_loader,
                inter_samples_edges_train,
                train_mask,
            )

            train_loss_gc = nn.CrossEntropyLoss()(
                gc_output, train_labels.long().view(-1)
            )

            train_loss_final = nn.CrossEntropyLoss()(
                final_output, train_labels.long().view(-1)
            )

            train_loss_total = (1 - alpha) * train_loss_gc + alpha * train_loss_final
            train_loss_total.backward()

            train_gc_final_loss_rel.append(
                (1 - alpha) * train_loss_gc / (alpha * train_loss_final)
            )

            gc_optimizer.step()
            nc_optimizer.step()

            train_gc_losses.append(train_loss_gc.item())
            train_final_losses.append(train_loss_final.item())
            train_total_losses.append(train_loss_total.item())

            train_gc_auc = roc_auc_score(
                one_hot(train_labels.view(-1).long(), n_classes).cpu().detach().numpy(),
                gc_output.cpu().detach().numpy(),
            )

            train_final_auc = roc_auc_score(
                one_hot(train_labels.view(-1).long(), n_classes).cpu().detach().numpy(),
                final_output.cpu().detach().numpy(),
            )

            train_gc_aucs.append(train_gc_auc)
            train_final_aucs.append(train_final_auc)

            if graphs_val_loader is not None:
                with torch.no_grad():
                    self.gc_model.eval()
                    self.nc_model.eval()

                    final_output, gc_output = self(
                        all_graphs_loader,
                        inter_samples_edges,
                        val_mask,
                    )

                    val_loss_gc = nn.CrossEntropyLoss()(
                        gc_output, val_labels.long().view(-1)
                    )

                    val_loss_final = nn.CrossEntropyLoss()(
                        final_output, val_labels.long().view(-1)
                    )

                    val_loss_total = (1 - alpha) * val_loss_gc + alpha * val_loss_final

                    val_final_gc_loss_rel.append(
                        (1 - alpha) * val_loss_gc / (alpha * val_loss_final)
                    )

                    val_gc_losses.append(val_loss_gc.item())
                    val_final_losses.append(val_loss_final.item())
                    val_total_losses.append(val_loss_total.item())

                    val_gc_auc = roc_auc_score(
                        one_hot(val_labels.view(-1).long(), n_classes)
                        .cpu()
                        .detach()
                        .numpy(),
                        gc_output.cpu().detach().numpy(),
                    )

                    val_final_auc = roc_auc_score(
                        one_hot(val_labels.view(-1).long(), n_classes)
                        .cpu()
                        .detach()
                        .numpy(),
                        final_output.cpu().detach().numpy(),
                    )

                    val_gc_aucs.append(val_gc_auc)
                    val_final_aucs.append(val_final_auc)

            if verbose:
                print(
                    f"Epoch {epoch}:\n" f"\tTrain GC loss:\t{train_loss_gc}\n",
                    f"\tTrain GC loss:\t{train_loss_gc}\n",
                    f"\tTrain final loss:\t{train_loss_final}\n",
                    f"\tTrain total loss:\t{train_loss_total}\n",
                    f"\tTrain GC AUC:\t{train_gc_auc}\n",
                    f"\tTrain final AUC:\t{train_final_auc}" + "\n"
                    if graphs_val_loader is not None
                    else "",
                )

                if graphs_val_loader is not None:
                    print(
                        f"\tVal GC loss:\t{val_loss_gc}\n",
                        f"\tVal final loss:\t{val_loss_final}\n",
                        f"\tVal total loss:\t{val_loss_total}\n",
                        f"\tVal GC AUC:\t{val_gc_auc}\n",
                        f"\tVal final AUC:\t{val_final_auc}" + "\n"
                        if graphs_val_loader is not None
                        else "",
                    )

            if early_stopping > 0:
                if val_loss_total < min_val_loss:
                    min_val_loss = val_loss_total
                    c = 0

                else:
                    c += 1

                if c == early_stopping:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch}.\n")
                    break

        if graphs_val_loader is not None:
            best_epoch, best_val_total_loss = np.argmin(val_total_losses) + 1, np.min(
                val_total_losses
            )

            best_val_final_auc = val_final_aucs[best_epoch - 1]

            if verbose:
                print(
                    f"Best epoch is {best_epoch} with total loss of {np.round(best_val_total_loss, 4)} on val and AUC of {np.round(best_val_final_auc, 4)} on validation (final)"
                )

                print(
                    f"Max AUC on validation obtained is {np.max(val_final_aucs)} in epoch {np.argmax(val_final_aucs) + 1}"
                )

        final_output_train, gc_output_train = self(
            all_graphs_loader,
            inter_samples_edges_train,
            train_mask,
        )

        if graphs_val_loader is not None:
            final_output_val, gc_output_val = self(
                all_graphs_loader,
                inter_samples_edges,
                val_mask,
            )

        if given_test_y:
            final_output_test, gc_output_test = self(
                all_graphs_loader,
                inter_samples_edges,
                test_mask,
            )

        if find_beta:
            best_beta, best_loss = -1, np.inf

            for beta in tqdm.tqdm(
                np.array(range(0, 101)) / 100, desc="Searching for best beta"
            ):
                output = beta * final_output_train + (1 - beta) * gc_output_train
                train_loss = nn.CrossEntropyLoss()(
                    output, train_labels.long().view(-1)
                ).item()

                if train_loss < best_loss:
                    best_loss = train_loss
                    best_beta = beta

            if verbose:
                print(f"Best beta is {best_beta} with loss of {best_loss} on training")

        else:
            best_beta = int(clf_from == "nc")

        mixed_output_train = (
            best_beta * final_output_train + (1 - best_beta) * gc_output_train
        )

        if graphs_val_loader is not None:
            mixed_output_val = (
                best_beta * final_output_val + (1 - best_beta) * gc_output_val
            )

        if given_test_y:
            mixed_output_test = (
                best_beta * final_output_test + (1 - best_beta) * gc_output_test
            )

        try:
            train_auc = roc_auc_score(
                one_hot(train_labels.view(-1).long(), n_classes).cpu().detach().numpy(),
                mixed_output_train.cpu().detach().numpy(),
            )

        except:
            train_auc = None

        try:
            if graphs_val_loader is not None:
                val_auc = roc_auc_score(
                    one_hot(val_labels.view(-1).long(), n_classes).cpu().detach().numpy(),
                    (final_output_val if clf_from == "nc" else gc_output_val)
                    .cpu()
                    .detach()
                    .numpy(),
                )

        except:
            val_auc = None

        if given_test_y:
            try:
                test_auc = roc_auc_score(
                    one_hot(test_labels.view(-1).long(), n_classes)
                    .cpu()
                    .detach()
                    .numpy(),
                    mixed_output_test.cpu().detach().numpy(),
                )

            except:
                test_auc = None

        # train_auc = roc_auc_score(
        #     one_hot(train_labels.view(-1).long(), n_classes).cpu().detach().numpy(),
        #     (final_output_train if clf_from == "nc" else gc_output_train)
        #     .cpu()
        #     .detach()
        #     .numpy(),
        # )

        # if graphs_val_loader is not None:
        #     val_auc = roc_auc_score(
        #         one_hot(val_labels.view(-1).long(), n_classes).cpu().detach().numpy(),
        #         (final_output_val if clf_from == "nc" else gc_output_val)
        #         .cpu()
        #         .detach()
        #         .numpy(),
        #     )

        # if given_test_y:
        #     test_auc = roc_auc_score(
        #         one_hot(test_labels.view(-1).long(), n_classes).cpu().detach().numpy(),
        #         (final_output_test if clf_from == "nc" else gc_output_test)
        #         .cpu()
        #         .detach()
        #         .numpy(),
        #     )

        if self.bin:
            pos_output_train = mixed_output_train[:, 1]

            (
                best_train_acc,
                best_train_f1,
                best_acc_threshold,
                best_f1_threshold,
            ) = find_best_metrics_bin(
                pos_output_train,
                train_labels,
                threshold=None,
            )

            if graphs_val_loader is not None:
                pos_output_val = mixed_output_val[:, 1]

                best_val_acc, *_ = find_best_metrics_bin(
                    pos_output_val,
                    val_labels,
                    threshold=best_acc_threshold,
                )

                _, best_val_f1, *_ = find_best_metrics_bin(
                    pos_output_val,
                    val_labels,
                    threshold=best_f1_threshold,
                )

            else:
                best_val_acc, best_val_f1 = None, None

            if given_test_y:
                pos_output_test = mixed_output_test[:, 1]

                best_test_acc, *_ = find_best_metrics_bin(
                    pos_output_test,
                    test_labels,
                    threshold=best_acc_threshold,
                )

                _, best_test_f1, *_ = find_best_metrics_bin(
                    pos_output_test,
                    test_labels,
                    threshold=best_f1_threshold,
                )

        else:
            best_train_acc, best_train_f1 = find_best_metrics_multi(
                mixed_output_train, train_labels
            )

            if graphs_val_loader is not None:
                best_val_acc, best_val_f1 = find_best_metrics_multi(
                    mixed_output_val, val_labels
                )

            if given_test_y:
                best_test_acc, best_test_f1 = find_best_metrics_multi(
                    mixed_output_test, test_labels
                )

            best_acc_threshold, best_f1_threshold = -1, -1

        if graphs_val_loader is None:
            val_auc = None

        if verbose:
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(
                f"Best Accuracy on train:\t{np.round(best_train_acc, 4)} for threshold {best_acc_threshold}"
            )
            print(
                f"Best F1 on train:\t{np.round(best_train_f1, 4)} for threshold {best_f1_threshold}"
            )

            if graphs_val_loader is not None:
                print(f"Best Accuracy on val:\t{np.round(best_val_acc, 4)}")
                print(f"Best F1 on val:\t{np.round(best_val_f1, 4)}")

        cache = {
            "Train Acc": best_train_acc,
            # "Val Acc": best_val_acc,
            "Train F1": best_train_f1,
            # "Val F1": best_val_f1,
            "Train AUC": train_auc,
            # "Val AUC": val_auc,
            "Acc Threshold": best_acc_threshold,
            "F1 Threshold": best_f1_threshold,
            "learning_epochs": epoch,
        }

        if graphs_val_loader is not None:
            cache["Val Acc"] = best_val_acc
            cache["Val F1"] = best_val_f1
            cache["Val AUC"] = val_auc


        if given_test_y:
            cache["Test Acc"] = best_test_acc
            cache["Test F1"] = best_test_f1
            cache["Test AUC"] = test_auc

        return cache, all_graphs_loader

    @staticmethod
    def plot(train, val, content, method, path):
        assert len(train) == len(val)
        plt.clf()
        plt.plot(range(1, 1 + len(train)), train, label=f"{content} - Train")
        plt.plot(range(1, 1 + len(val)), val, label=f"{content} - Val")

        plt.title(f"{method} - {content}")
        plt.legend()
        plt.savefig(path)
        plt.show()
