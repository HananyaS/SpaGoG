import numpy as np
import torch
from typing import Callable, Union

torchify = (
    lambda x, device: x if type(x) == torch.Tensor else torch.from_numpy(x).to(device)
)
rmse = lambda mat1, mat2: torch.sqrt(torch.mean(torch.pow(mat1 - mat2, 2))).float()
gap = lambda ary: float("inf") if len(ary) < 2 else ary[-2] - ary[-1]


class GFP:
    _types = Union[torch.Tensor, np.ndarray]

    def __init__(
            self,
            index: Union[bool, torch.Tensor] = False,
            iters: int = 50,
            eps: float = 1e-3,
            delta_func: Callable = rmse,
            early_stop: bool = False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.index = (
            torchify(index, self.device).bool() if type(index) != bool else index
        )

        self.early = early_stop
        self.iters = iters
        self.A_hat = None
        self.delta = []
        self.stop = False
        self.eps = eps
        self.f = delta_func

    def build_a_hat(self, nodes_num: int, edges: torch.Tensor):
        edges = edges if edges.size()[0] == 2 else edges.T

        eye = torch.arange(nodes_num)
        row = torch.cat((edges[0, :], eye))
        col = torch.cat((edges[1, :], eye))

        edges = torch.stack((row, col)).to(self.device)

        _, D = torch.unique(edges[0, :], return_counts=True)
        D_inv = torch.pow(D, -0.5)
        weight = D_inv[row] * D_inv[col]

        self.A_hat = torch.sparse.FloatTensor(edges, weight).to(self.device)

        del weight, D, D_inv, row, col, edges

        return None

    def define_index(self, X):
        self.index = (
            (X != X).long().bool()
            if type(X) == torch.Tensor
            else torch.from_numpy(X != X).bool()
        ).to(self.device)
        return None

    def prop(self, X: _types, edges: _types):
        input_type = torch if isinstance(X, torch.Tensor) else np

        X = torchify(X, self.device).float()
        edges = torchify(edges, self.device)

        if 2 not in edges.size():
            edges = torch.nonzero(edges, as_tuple=False)

        self.define_index(X)
        X = torch.nan_to_num(X).to(self.device)
        self.build_a_hat(X.size()[0], edges)

        old_mat = X

        for i in range(self.iters):
            AX = torch.sparse.mm(self.A_hat, X)
            X = torch.where(self.index, AX, old_mat)

            self.delta.append(
                self.f(AX[self.index == False], old_mat[self.index == False])
            )

            if self.early and gap(self.delta) < self.eps:
                print(f"Early stop at iteration {i + 1}")
                self.stop = True

                if input_type == torch:
                    return X

                return X.detach().cpu().numpy()

        if input_type == torch:
            return X

        return X.detach().cpu().numpy()
