import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from .abstractNN import AbstractNN
from typing import Union, Type, List
from torch.utils.data import DataLoader
from ..datasets.graphsDataset import GraphsDataset


class ValuesAndGraphStructure(AbstractNN):
    def __init__(
            self,
            nodes_number: int = None,
            feature_size: int = None,
            RECEIVED_PARAMS: dict = {
                "preweight": 5,
                "layer_1": 9,
                "layer_2": 7,
                "activation": "elu",
                "dropout": 0,
            },
            num_classes: int = 1,
            input_example: Union[DataLoader, GraphsDataset] = None,
            init_weights="xavier_normal",
            embedding_layer: str = 'one_before_last',
            n_gcn_layers: int = 1,
    ):
        super(ValuesAndGraphStructure, self).__init__()

        if input_example is not None:
            self.init_attributes_by_example(input_example)

        else:
            assert (
                    nodes_number is not None
                    and feature_size is not None
                    and num_classes is not None
            )

            self.feature_size = feature_size
            self.nodes_number = nodes_number
            self.num_classes = num_classes

        self.num_classes = max([self.num_classes, 2])
        self.RECEIVED_PARAMS = RECEIVED_PARAMS

        self.first_pre_weighting = nn.Linear(1, int(self.RECEIVED_PARAMS["preweight"])).to(
            self.device
        )

        self.next_pre_weighting = [nn.Linear(
            int(self.RECEIVED_PARAMS["preweight"]), int(self.RECEIVED_PARAMS["preweight"])
        ).to(self.device) for _ in range(n_gcn_layers - 1)]

        self.fc1 = nn.Linear(
            int(self.RECEIVED_PARAMS["preweight"]) * self.nodes_number,
            int(self.RECEIVED_PARAMS["layer_1"]),
        ).to(
            self.device
        )

        self.fc2 = nn.Linear(
            int(self.RECEIVED_PARAMS["layer_1"]), int(self.RECEIVED_PARAMS["layer_2"])
        ).to(self.device)
        self.fc3 = nn.Linear(int(self.RECEIVED_PARAMS["layer_2"]), self.num_classes).to(
            self.device
        )
        self.activation_func = self.RECEIVED_PARAMS["activation"]
        self.dropout = nn.Dropout(p=self.RECEIVED_PARAMS["dropout"]).to(self.device)

        self.alpha = nn.Parameter(torch.rand(1, requires_grad=True, device=self.device))

        self.activation_func_dict = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
        }

        if self.feature_size > 1:
            self.transform_mat_to_vec = nn.Linear(self.feature_size, 1)

        if init_weights is not None:
            self.init_weights(init_weights)

        self.first_gcn_layer = nn.Sequential(
            self.first_pre_weighting, self.activation_func_dict[self.activation_func]
        ).to(self.device)

        self.next_gcn_layers = nn.ModuleList([nn.Sequential(
            next_pre_weighting, self.activation_func_dict[self.activation_func]
        ).to(self.device) for next_pre_weighting in self.next_pre_weighting])

        self.embedding_layer = embedding_layer

    def init_attributes_by_example(
            self,
            input_example: Union[DataLoader, GraphsDataset],
    ):
        if isinstance(input_example, DataLoader):
            input_example = input_example.dataset.gdp

        self.feature_size = input_example.num_features
        self.nodes_number = input_example.num_features
        self.num_classes = input_example.num_classes

    def _forward_one_before_last_layer(self, x, adjacency_matrix, *args):
        x = x.to(self.device)

        a, b, c = adjacency_matrix.shape
        I = torch.eye(b).to(self.device)
        alpha_I = I * self.alpha.expand_as(I)
        normalized_adjacency_matrix = self.calculate_adjacency_matrix(
            adjacency_matrix
        ).to(
            self.device
        )
        alpha_I_plus_A = alpha_I + normalized_adjacency_matrix
        x = torch.einsum("ijk, ik->ij", alpha_I_plus_A.float(), x).unsqueeze(
            -1
        )

        x = self.first_gcn_layer(x)

        for next_gcn_layer in self.next_gcn_layers:
            x = next_gcn_layer(x)

        x = torch.flatten(x, start_dim=1)

        if self.embedding_layer != 'first':
            x = self.activation_func_dict['relu'](x)
            x = self.fc1(x)

            if self.embedding_layer != 'mid':
                x = self.activation_func_dict['relu'](x)
                x = nn.Dropout(self.RECEIVED_PARAMS["dropout"])(x)
                x = self.fc2(x)

        return x

    def _forward_last_layer(self, x, *args):
        if self.embedding_layer == 'first':
            x = self.activation_func_dict['relu'](x)
            x = self.fc1(x)

        if self.embedding_layer != 'one_before_last':
            x = nn.ReLU()(x)
            x = nn.Dropout(self.RECEIVED_PARAMS["dropout"])(x)
            x = self.fc2(x)

        x = nn.ELU()(x)
        x = self.fc3(x)

        return x

    def calculate_adjacency_matrix(self, batched_adjacency_matrix):
        def calc_d_minus_root_sqr(batched_adjacency_matrix):
            r = []
            for adjacency_matrix in batched_adjacency_matrix:
                sum_of_each_row = adjacency_matrix.sum(1).to(self.device)
                sum_of_each_row_plus_one = torch.where(
                    sum_of_each_row != 0,
                    sum_of_each_row,
                    1.0,
                )
                r.append(torch.diag(torch.pow(sum_of_each_row_plus_one, -0.5)))
            s = torch.stack(r)
            if torch.isnan(s).any():
                print("Alpha when stuck", self.alpha.item())
                print(
                    "batched_adjacency_matrix",
                    torch.isnan(batched_adjacency_matrix).any(),
                )
                print("The model is stuck", torch.isnan(s).any())
            return s

        batched_adjacency_matrix = batched_adjacency_matrix.to(self.device)
        D__minus_sqrt = calc_d_minus_root_sqr(batched_adjacency_matrix).to(self.device)
        normalized_adjacency = torch.matmul(
            torch.matmul(D__minus_sqrt, batched_adjacency_matrix), D__minus_sqrt
        )
        return normalized_adjacency

    def _transform_input(self, data: Union[Type[torch.Tensor], List]):
        if len(data) == 2:
            xs, adjs = data
            labels = None
        else:
            xs, adjs, labels = data

        xs = xs.to(self.device)
        adjs = adjs.to(self.device)

        if len(xs.shape) == 3 and xs.shape[-1] == 1:
            xs = torch.squeeze(xs, -1)

        return [xs, adjs], labels

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

    def get_num_classes(self):
        return self.num_classes + 1 if self.num_classes == 1 else self.num_classes

    def _transform_output(self, output):
        return output
