import torch
from dataclasses import dataclass
from pde.graph_grid.graph_utils import gen_perim
from enum import Enum, auto


class P_Types(Enum):
    NORMAL = auto()
    BOUNDARY = auto()
    GHOST = auto()

@dataclass
class Point:
    point_type: P_Types
    X: torch.Tensor
    value: float
    direction: str = None

    """
    Value:  If normal, value = initial value. Standard PDE enforced on this point.
            If boundary, value = boundary value. Dirichlet BC enforced on this point.
            If ghost, value = derivative value. Direction is direction of normal derivative boundary. Neuman BCs enforced on this point.
    """
    def __repr__(self):
        if self.value is None:
            return f'\033[33mPoint:\n     X={self.X}, \n     Type={self.point_type})\n\033[0m'
        else:
            return f'\033[33mPoint:\n     X={self.X}, \n     Type={self.point_type}, \n     Value={self.value})\n\033[0m'

@dataclass
class DerivGraph:
    edge_idx: torch.Tensor
    weights: torch.Tensor
    neighbors: dict[int, torch.Tensor]

    device: str = "cpu"
    def cuda(self):
        self.edge_idx = self.edge_idx.cuda(non_blocking=True)
        self.weights = self.weights.cuda(non_blocking=True)
        self.device = "cuda"


def main():
    torch.set_printoptions(precision=3, sci_mode=False)
    torch.random.manual_seed(2)
    points_bc = gen_perim(1, 1, 0.2)
    points_bc = [Point(X, P_Boundary, 0.) for X in points_bc]
    points_main = torch.rand([100, 2])
    points_main = [Point(X, P_Normal, 0.) for X in points_main]

    points_all = points_bc + points_main
    points_dict = {i: X for i, X in enumerate(points_all)}
    # print(f'{points_dict = }')

    edge_idx, weights, neighbors = calc_coeff(points_dict, 3, (2, 0))
    #
    # show_graph(edge_idx, points, torch.zeros_like(points[:, 0]))


if __name__ == "__main__":
    main()