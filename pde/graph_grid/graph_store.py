import torch
from dataclasses import dataclass
from pde.graph_grid.graph_utils import gen_perim
from enum import Flag, auto


class P_Types(Flag):
    NORMAL = auto() # Standard PDE: u = f(us, x) enforced on this point.
    BOUNDARY = auto()   # Dirichlet: u = Dirichlet(x) enforced on this point.
    DERIV = auto()  # deriv(u, us) = Neumann(x) enforced on this point. Can be used as central Neumann derivative or edge, depending on other nodes.
    GHOST = auto() # No function on point. Ghost point. u function inherited from central DERIV point.

    BOTH = BOUNDARY | DERIV
    GRAD = NORMAL | GHOST # u requires fitting on point.


@dataclass
class Point:
    point_type: P_Types
    X: torch.Tensor
    value: float = None
    derivatives: dict[tuple, float] = None
    """ value:  If NORMAL, value = initial value. 
                If BOUNDARY, value = boundary value. 
        derivatives: If DERIV, derivatives = {((x, y), value), ...} where (x, y) is the derivative order.  """

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



if __name__ == "__main__":
    main()