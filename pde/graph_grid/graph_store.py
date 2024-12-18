import torch
from dataclasses import dataclass
from enum import Flag, auto


class P_Types(Flag):
    """ Types of points.
        Note: Checks should be done as PRIMITIAVE in POINT_VALUE.
    """
    # Primitive types
    PDE = auto() # Standard PDE: u = f(us, x) enforced on this point.
    NONE = auto() # No function on point. Ghost point. u function inherited from central DERIV point.

    FIX = auto()   # Dirichlet: u = Dirichlet(x) enforced on this point.
    DERIV = auto()  # deriv(u, us) = Neumann(x) enforced on this point. Can be used as central Neumann derivative or edge, depending on other nodes.

    GRAD = auto()   # u requires fitting on point.

    # User BC types
    DirichBC = FIX  # Dirichlet BC enforced on point.
    NeumCentralBC = DERIV | PDE | GRAD # Neumann BC + PDE enforced on point.
    NeumOffsetBC = DERIV | GRAD # Only Neumann BC on point.
    BothBC = FIX | DERIV  # Both Dirichlet and Neumann BC enforced on point.
    Ghost = NONE | GRAD # Ghost point

    # Normal point
    Normal = PDE | GRAD # Standard PDE enforced on point.

class P_TimeTypes(Flag):
    """ Point types for time dependent problems. """
    FIXED = auto()  # Fixed value point. No fitting required.
    MANUAL = auto()  # Manually set value. No fitting required.

    NORMAL = auto()  # Normal point. Fitting required.

@dataclass
class Deriv:
    """ Stores all derivative boundary conditions on a point.
        sum{ d^n u_k / dx_i dx_j } = value
    """
    comp: list[int]     # Which components of us to apply derivative to.
    orders: list[tuple[int, int]]       # Derivative orders. (i, j) = d^i/dx_i d^j/dx_j
    value: float            # Sum values

    def __post_init__(self):
        assert len(self.comp) == len(self.orders), "Number of components must equal number of derivative orders."

@dataclass
class Point:
    """ value:  If NORMAL, value = initial value.
                If BOUNDARY, value = boundary value.
        derivatives: Sum of derivatives = value. """
    point_type: P_Types
    X: torch.Tensor
    value: float|list[float] = None
    derivatives: list[Deriv] = None

    def __post_init__(self):
        if P_Types.DERIV in self.point_type:
            assert self.derivatives is not None, "Derivatives must be provided for DERIV points."

    def __repr__(self):
        if self.value is None:
            return f'\033[33mPoint:\n     X={self.X}, \n     Type={self.point_type})\n\033[0m'
        else:
            return f'\033[33mPoint:\n     X={self.X}, \n     Type={self.point_type}, \n     Value={self.value})\n\033[0m'

@dataclass
class T_Point:
    """ value:  If NORMAL, value = initial value.
                If BOUNDARY, value = boundary value.
        derivatives: Sum of derivatives = value. """
    point_type: list[P_TimeTypes]   # [N_component]
    X: torch.Tensor
    init_val: float|list[float]



@dataclass
class DerivGraph:
    edge_idx: torch.Tensor
    weights: torch.Tensor

    device: str = "cpu"
    def cuda(self):
        self.edge_idx = self.edge_idx.cuda(non_blocking=True)
        self.weights = self.weights.cuda(non_blocking=True)
        self.device = "cuda"


def main():
    torch.set_printoptions(precision=3, sci_mode=False)



if __name__ == "__main__":
    main()