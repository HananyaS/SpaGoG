import numpy as np
import torch

from sklearn.metrics import f1_score


def find_best_metrics_bin(
    pos_output: torch.Tensor, labels: torch.Tensor, threshold: float = None
):
    def calc_metrics(pos_output, threshold):
        labels_ = labels.view(-1)
        pred = pos_output > threshold

        acc = (pred == labels_).float().mean().item()

        conf_vector = pred / labels_

        tp = (conf_vector == 1).float().sum().item()
        fp = (conf_vector == float("inf")).float().sum().item()
        fn = (conf_vector == 0).float().sum().item()

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn)

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall != 0
            else 0
        )

        return acc, f1

    if threshold is None:
        thresholds = np.linspace(0, 1, 101)
        max_acc, max_f1 = 0, 0
        max_acc_t, max_f1_t = -1, -1

        for t in thresholds:
            acc, f1 = calc_metrics(pos_output, t)

            if acc > max_acc:
                max_acc = acc
                max_acc_t = t

            if f1 > max_f1:
                max_f1 = f1
                max_f1_t = t

        best_acc, best_f1, best_acc_t, best_f1_t = (
            max_acc,
            max_f1,
            max_acc_t,
            max_f1_t,
        )

    else:
        best_acc, best_f1 = calc_metrics(pos_output, threshold)
        best_acc_t = best_f1_t = threshold

    return best_acc, best_f1, best_acc_t, best_f1_t


def find_best_metrics_multi(pred_proba: torch.Tensor, labels: torch.Tensor):
    preds = pred_proba.argmax(dim=1)

    acc = (preds == labels.view(-1)).float().mean().item()
    f1 = f1_score(labels.view(-1).cpu(), preds.cpu(), average='macro')

    return acc, f1