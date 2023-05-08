import sys

PROJECT_DIR = '.'
sys.path.append(PROJECT_DIR)

import torch
from torch import nn
from torch.nn.functional import one_hot

from torch_geometric.nn import GCNConv, Sequential
from .abstractNN import AbstractNN
from typing import List, Union, Type


class NodeClassification(AbstractNN):
    _activation_dict = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "tanh": nn.Tanh(),
    }

    def __init__(
            self,
            device: torch.device = torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu"),
            n_features: int = -1,
            n_classes: int = 2,
            h_layers: List[int] = [10, 5],
            dropouts: List[float] = None,
            activations: List[str] = ["elu", "elu"],
    ):
        assert dropouts is None or len(dropouts) == len(h_layers)
        assert activations is None or len(activations) == len(h_layers)
        assert set(activations) <= set(self._activation_dict.keys())
        assert dropouts is None or all([0 <= d < 1 for d in dropouts])

        super(NodeClassification, self).__init__(device)
        self.n_classes = n_classes
        self.h_layers = h_layers
        self.dropouts = dropouts
        self.activations = activations
        self.n_features = n_features

        self.one_before_last_layer, self.classifier = self._get_layers(
            n_features=n_features,
            n_classes=n_classes,
            h_layers=h_layers,
            dropouts=dropouts,
            activations=activations,
        )

    def _get_layers(self, n_features, n_classes, h_layers, dropouts, activations):
        start_layers = []
        end_layers = []

        for i in range(len(h_layers)):
            if i == 0:
                gcn_layer = GCNConv(n_features, h_layers[i]).cpu()
            else:
                gcn_layer = GCNConv(h_layers[i - 1], h_layers[i]).cpu()

            start_layers.append((gcn_layer, "x, edge_index -> x"))

            if dropouts is not None:
                if i < len(dropouts) - 1:
                    start_layers.append(
                        (nn.Dropout(dropouts[i], inplace=False).cpu(), "x -> x")
                    )

                else:
                    end_layers.append(
                        (nn.Dropout(dropouts[i], inplace=False).cpu(), "x -> x")
                    )

            if activations:
                if i < len(activations) - 1:
                    start_layers.append(
                        (self._activation_dict[activations[i]], "x -> x")
                    )

                else:
                    end_layers.append((self._activation_dict[activations[i]], "x -> x"))

        end_layers.append((nn.Linear(h_layers[-1], n_classes).cpu(), "x -> x"))

        start_layers = Sequential("x, edge_index", start_layers).cpu()
        end_layers = Sequential("x", end_layers).cpu()

        return start_layers, end_layers

    def _forward_one_before_last_layer(self, *args, **kwargs):
        mask = args[-1]
        args = args[:-1]
        return (self.one_before_last_layer(*args, **kwargs)), mask

    def _forward_last_layer(self, *args, **kwargs):
        args, mask = args[0]
        args = tuple([args])
        return self.classifier(*args, **kwargs), mask

    def _transform_input(self, data: Union[Type[torch.Tensor], List]):
        return data

    def _transform_output(self, output):
        output, mask = output
        return output.to(self.device)[mask]

    def get_num_classes(self):
        return self.n_classes

    def _eval_loss(
            self,
            output: torch.Tensor,
            labels: torch.Tensor,
            loss_func: torch.nn.modules.loss,
            n_classes: int = 2,
    ) -> torch.nn.modules.loss:
        labels = one_hot(labels.long(), num_classes=n_classes).float()
        loss = loss_func(output, labels.squeeze(1))
        return loss

    def __str__(self):
        return "Node Classification model"
