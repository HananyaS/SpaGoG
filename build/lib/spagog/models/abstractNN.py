import time
import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
from abc import abstractmethod
from torch.utils.data import DataLoader

from typing import Callable, List
from sklearn.metrics import roc_auc_score

from .abstractModel import AbstractModel

from ..utils.metrics.metrics import find_best_metrics_bin, find_best_metrics_multi


class AbstractNN(nn.Module, AbstractModel):
    def __init__(
        self,
        device: torch.device = torch.device("cpu")
        if not torch.cuda.is_available()
        else torch.device("cuda"),
    ):
        super(AbstractNN, self).__init__()
        self.device = device

    def init_weights(self, method="xavier_normal"):
        assert method in [
            "xavier_normal",
            "xavier_uniform",
        ], f'Unknown method:\t{f"{method}"}'

        for k, v in self._modules.items():
            if isinstance(v, nn.Linear):
                if method == "xavier_normal":
                    nn.init.xavier_normal(v.weight)
                    v.bias.data.fill_(0.01)

                if method == "xavier_uniform":
                    nn.init.xavier_uniform(v.weight)
                    v.bias.data.fill_(0.01)

    @abstractmethod
    def _forward_one_before_last_layer(self, *args, **kwargs):
        raise NotImplementedError

    def forward_one_before_last_layer(self, *args, **kwargs):
        return self._forward_one_before_last_layer(*args, **kwargs)

    @abstractmethod
    def _forward_last_layer(self, *args, **kwargs):
        raise NotImplementedError

    def forward_last_layer(self, *args, **kwargs):
        return self._forward_last_layer(*args, **kwargs)

    @abstractmethod
    def _transform_output(self, *args, **kwargs):
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        x = self.forward_one_before_last_layer(*args, **kwargs)
        x = self.forward_last_layer(x)
        return self._transform_output(x)

    @abstractmethod
    def _transform_input(self, *args, **kwargs):
        raise NotImplementedError

    def transform_input(self, data: torch.Tensor):
        return self._transform_input(data)

    @abstractmethod
    def _eval_loss(
        self,
        output: torch.Tensor,
        labels: torch.Tensor,
        loss_func: torch.nn.modules.loss,
        n_classes: int = 2,
    ) -> torch.nn.modules.loss:
        raise NotImplementedError

    @staticmethod
    def _plot_results(
        train_losses: List,
        val_losses: List,
        train_aucs: List,
        val_aucs: List,
        save_results: bool,
        auc_plot_path: str,
        loss_plot_path: str,
        show_results: bool,
    ):
        plt.clf()
        plt.plot(range(1, 1 + len(train_losses)), train_losses, label="Train")
        plt.plot(range(1, 1 + len(val_losses)), val_losses, label="Val")
        plt.legend()
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if save_results:
            plt.savefig(loss_plot_path)

        if show_results:
            plt.show()

        plt.clf()
        plt.plot(range(1, 1 + len(train_aucs)), train_aucs, label="Train")
        plt.plot(range(1, 1 + len(val_aucs)), val_aucs, label="Val")
        plt.legend()
        plt.title("AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")

        if save_results:
            plt.savefig(auc_plot_path)

        if show_results:
            plt.show()

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        dataset_name: str,
        test_loader: DataLoader = None,
        n_epochs: int = 10,
        lr: float = 0.001,
        weight_decay: float = 0,
        optimizer: torch.optim = optim.Adam,
        verbose: bool = False,
        criterion: torch.nn.modules.loss = nn.CrossEntropyLoss(reduction="mean"),
        labels_from_loader: Callable = lambda loader: loader.dataset.gdp.Y,
        metric: str = "auc",
        save_model: bool = False,
        early_stopping: int = -1,
    ):
        assert metric in ["auc", "accuracy"]

        if early_stopping > 0 and val_loader is None:
            print("Early stopping is off because val_loader is None")
            early_stopping = -1

        self.train()
        train_losses, val_losses = [], []
        train_aucs, val_aucs = [], []

        optimizer = optimizer(self.parameters(), lr=lr, weight_decay=weight_decay)

        if early_stopping > 0:
            c = 0
            max_val_auc = np.inf

        t_per_epoch = []
        t_per_train_epoch = []
        t_per_val_epoch = []
        t_per_auc_epoch = []

        t_only_pass_train = []

        for epoch in range(1, n_epochs + 1):
            t_start = time.time()
            total_train_loss = 0

            t_train = time.time()

            for b_num_train, data in enumerate(train_loader):
                t_opt = time.time()
                optimizer.zero_grad()

                input_data, labels = self._transform_input(data)
                output = self(*input_data)

                t_only_pass_train.append(time.time() - t_opt)


                loss = self._eval_loss(
                    output,
                    labels,
                    criterion,
                    n_classes=self.get_num_classes(),
                )

                loss.backward(retain_graph=True)
                optimizer.step()

                total_train_loss += loss.item() * (
                    input_data[0].size(0)
                    / (1 if criterion.reduction == "sum" else input_data[0].size(0))
                )

            t_per_train_epoch.append(time.time() - t_train)
            total_train_loss = total_train_loss / (len(train_loader.dataset) if isinstance(train_loader, DataLoader) else train_loader[0][1].shape[0])

            train_auc = self.evaluate(
                loader=train_loader,
                metric=metric,
                labels_from_loader=labels_from_loader,
                to_numpy=False
            )

            if val_loader is not None:
                t_val = time.time()

                with torch.no_grad():
                    total_val_loss = 0

                    for b_num_val, data in enumerate(val_loader):
                        input_data, labels = self._transform_input(data)
                        output = self(*input_data)

                        loss = self._eval_loss(
                            output,
                            labels,
                            criterion,
                            n_classes=self.get_num_classes(),
                        )

                        total_val_loss += loss.item() * (
                            input_data[0].size(0)
                            / (1 if criterion.reduction == "sum" else input_data[0].size(0))
                        )

                t_per_val_epoch.append(time.time() - t_val)

                total_val_loss = total_val_loss / (len(val_loader.dataset) if isinstance(val_loader, DataLoader) else val_loader[0][1].shape[0])

                val_auc = self.evaluate(
                    loader=val_loader, metric=metric, labels_from_loader=labels_from_loader, to_numpy=False
                )

                val_losses.append(total_val_loss)
                val_aucs.append(val_auc)


            if verbose:
                print(
                    f"Epoch {epoch}/{n_epochs}:\n"
                    f"\tTrain loss: {total_train_loss:.4f}\n"
                    f"\tTrain AUC: {train_auc:.4f}" + ('\n' if val_loader is None else "")
                )

                if val_loader is not None:
                    print(
                        f"\tVal loss: {total_val_loss:.4f}\n"
                        f"\tVal AUC: {val_auc:.4f}\n"
                    )

            if early_stopping > 0 and val_loader is not None:
                if val_auc > max_val_auc:
                    max_val_auc = val_auc
                    c = 0

                else:
                    c += 1

                if c == early_stopping:
                    if verbose:
                        print(f"\nEarly stopping triggered at epoch {epoch}\n")
                    break

            t_per_epoch.append(time.time() - t_start)

        results_cache = self.evaluate(
            metric="acc+f1",
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            labels_from_loader=labels_from_loader,
            to_numpy=False,
        )

        results_cache["learning_epochs"] = epoch
        results_cache["Train AUC"] = train_auc
        results_cache["Val AUC"] = None if val_loader is None else val_auc

        if test_loader is not None and labels_from_loader(test_loader) is not None:
            test_auc = self.evaluate(
                loader=test_loader, metric="auc", labels_from_loader=labels_from_loader, to_numpy=False
            )

            results_cache["Test AUC"] = test_auc

        return results_cache

    def _save_model(self, path: str = "model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        return self

    def predict(
        self,
        loader: DataLoader,
        probs: bool = False,
        pred_from_output: Callable = lambda output: torch.argmax(output, 1),
        to_numpy: bool = True,
    ) -> torch.Tensor:
        self.eval()

        preds = None

        with torch.no_grad():
            for data in loader:
                input_data, _ = self._transform_input(data)
                output = self(*input_data)

                if probs:
                    if preds is None:
                        preds = output

                    else:
                        preds = torch.cat((preds, output), 0)

                else:
                    if preds is None:
                        preds = torch.argmax(output, dim=1).view(-1)

                    else:
                        preds = torch.cat(
                            (preds, torch.argmax(output, dim=1).view(-1)), 0
                        )

            if probs:
                preds = torch.softmax(preds, dim=1)

        return preds

    def evaluate(self, metric: str = "auc", **kwargs) -> float:
        if metric == "auc":
            return self._eval_auc(**kwargs)

        if metric == "acc+f1":
            return self._eval_acc_f1(**kwargs)

        raise NotImplementedError

    def _eval_acc_f1(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        labels_from_loader: Callable,
        test_loader: DataLoader = None,
        **kwargs
    ):
        given_test_y = test_loader is not None and labels_from_loader(test_loader) is not None

        train_labels = labels_from_loader(train_loader).view(-1, 1).long()
        final_output_train = self.predict(loader=train_loader, probs=True, **kwargs)

        if val_loader is not None:
            val_labels = labels_from_loader(val_loader).view(-1, 1)
            final_output_val = self.predict(loader=val_loader, probs=True, **kwargs)

        if given_test_y:
            test_labels = labels_from_loader(test_loader).view(-1, 1)
            final_output_test = self.predict(loader=test_loader, probs=True, **kwargs)

        if len(train_labels.unique()) == 2:
            pos_output_train = nn.Sigmoid()(final_output_train[:, 1])

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

            if val_loader is not None:
                pos_output_val = nn.Sigmoid()(final_output_val[:, 1])

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
                best_val_acc = best_val_f1 = None

            if given_test_y:
                pos_output_test = nn.Sigmoid()(final_output_test[:, 1])

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

            cache = {
                "Train Acc": best_train_acc,
                "Val Acc": best_val_acc,
                "Train F1": best_train_f1,
                "Val F1": best_val_f1,
                "Acc Threshold": best_acc_threshold,
                "F1 Threshold": best_f1_threshold,
            }

            if given_test_y:
                cache = {
                    **cache,
                    "Test Acc": best_test_acc,
                    "Test F1": best_test_f1,
                }

            return cache

        train_acc, train_f1 = find_best_metrics_multi(final_output_train, train_labels)

        if val_loader is not None:
            val_acc, val_f1 = find_best_metrics_multi(final_output_val, val_labels)

        else:
            val_acc, val_f1 = None, None

        cache = {
            "Train Acc": train_acc,
            "Val Acc": val_acc,
            "Train F1": train_f1,
            "Val F1": val_f1,
            "Acc Threshold": -1,
            "F1 Threshold": -1,
        }

        if given_test_y:
            test_acc, test_f1 = find_best_metrics_multi(final_output_test, test_labels)

            cache = {
                **cache,
                "Test Acc": test_acc,
                "Test F1": test_f1,
            }

        return cache

    def _eval_auc(
        self,
        loader: DataLoader,
        labels_from_loader: Callable = lambda loader: loader.dataset.gdp.Y,
        **kwargs
    ) -> float:

        if loader is None:
            return None

        self.eval()
        preds = self.predict(loader, probs=True, **kwargs)
        labels = labels_from_loader(loader).view(-1, 1).long()

        if labels is None:
            return None

        labels_ = torch.zeros_like(preds)
        labels_[range(labels.shape[0]), labels[:, 0]] = 1

        try:
            return roc_auc_score(labels_.cpu(), preds.cpu())

        except ValueError:
            return -1

    def __str__(self):
        return "BaseModel"

    @abstractmethod
    def get_num_classes(self):
        raise NotImplementedError
