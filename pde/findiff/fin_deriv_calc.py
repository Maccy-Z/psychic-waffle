import torch
from torch_geometric.nn import MessagePassing

from pde.graph_grid.graph_store import DerivGraph
from pde.BaseDerivCalc import BaseDerivCalc
from pde.utils_sparse import coo_row_select, coo_col_select, CSRToInt32


class FinDerivCalc(MessagePassing, BaseDerivCalc):
    """ Compute Grad^n(u) using finite differences. """

    def __init__(self, fd_graphs: dict[tuple, DerivGraph], pde_mask):
        """ edge_index (dict[tuple, Tensor]): Graph connectivity in COO format with shape [2, E].
            edge_coeffs (dict[tuple, Tensor]): Finite difference coefficients for each edge."""
        super().__init__(aggr='add')  # Aggregation method can be 'add', 'mean', etc.

        self.fd_graphs = fd_graphs
        self.pde_mask = pde_mask

    def derivative(self, Xs) -> dict[tuple, torch.Tensor]:
        return self(Xs)

    def forward(self, Xs) -> dict[tuple, torch.Tensor]:
        """
        Args:
            x (Tensor): Node feature matrix of shape [N, F].
        """
        # Include original node value
        derivatives = {(0, 0): Xs[self.pde_mask]}
        for order, graph in self.fd_graphs.items():
            edge_idx = graph.edge_idx
            coeff = graph.weights

            derivatives[order] = self.propagate(edge_idx, x=Xs.unsqueeze(-1), edge_coeff=coeff.unsqueeze(-1))[self.pde_mask].squeeze()
        return derivatives

    def message(self, x_j, edge_coeff):
        """
        Constructs messages from source nodes to target nodes.

        Args:
            x_j (Tensor): Source node features for each edge of shape [E, F].
            edge_coeff (Tensor): Edge coefficients for each edge of shape [E, 1] or [E, F].

        Returns:
            Tensor: Messages to be aggregated.
        """
        return  edge_coeff * x_j


class FinDerivCalcSPMV(BaseDerivCalc):
    """ Using sparse matrix-vector multiplication to compute Grad^n(u) using finite differences. """
    def __init__(self, fd_graphs: dict[tuple, DerivGraph], eq_mask: torch.Tensor, grad_mask: torch.Tensor, N_us_tot, device="cpu"):
        """ Initialise sparse matrices from finite difference graphs.
            Compute d = A * u [eq_mask]. Compile eq_mask into A.
            Jacobian is A[eq_mask][us_mask]

            eq_mask: Constraint functions (PDEs / BCs)
            grad_mask: u points that need to be updated
            N_us_tot: Total number of u points on graph.
        """
        self.eq_mask = eq_mask
        self.grad_mask = grad_mask
        self.device = device

        self.fd_spms = {}       # shape = [N_diff], [N_eqs, N_us_tot]
        self.jac_spms = []      # shape = [N_diff], [N_eqs, N_grad]

        # Order (0, 0) is original node value
        only_us = torch.eye(N_us_tot, device=self.device).to_sparse_coo()
        only_us = coo_row_select(only_us, self.eq_mask)
        only_us = coo_col_select(only_us, self.grad_mask)
        self.jac_spms.append(only_us.to_sparse_csr())   # shape = [N_diff], [N_eqs, N_us_tot]
        for order, graph in fd_graphs.items():
            edge_idx = graph.edge_idx
            edge_coeff = graph.weights
            sp_mat = torch.sparse_coo_tensor(edge_idx, edge_coeff.squeeze(), (N_us_tot, N_us_tot)).T.coalesce()
            sp_mat = coo_row_select(sp_mat, self.eq_mask)        # shape = [N_eqs, N_us_tot]
            self.fd_spms[order] = CSRToInt32(sp_mat.to_sparse_csr())

            sp_mat_jac = coo_col_select(sp_mat, self.grad_mask)   # shape = [N_eqs, N_grad]
            sp_mat_jac = CSRToInt32(sp_mat_jac.to_sparse_csr())
            self.jac_spms.append(sp_mat_jac)


    def derivative(self, Xs) -> dict[tuple, torch.Tensor]:
        """ Xs.shape = [N_points]
            spm.shape = [N_pde, N_points]
        """
        derivatives = {(0, 0): Xs[self.eq_mask]}

        for order, spm in self.fd_spms.items():

            derivatives[order] = torch.mv(spm, Xs)
        return derivatives

    def jacobian(self) -> list[torch.FloatTensor]:
        """ Linear transform, so jacobian is the same as the sparse matrix.
            return.shape: [N_diff], [N_pde, N_points]
         """
        return self.jac_spms


class NeumanBCCalc(FinDerivCalcSPMV):
    """ Compute FinDiff derivatives for (linear) Neumann BCs, and full jacobian for R = sum_n grad_n(u) - constant.
        Precompute the selection derivatives and jacobian, that directly returns residuals / residual jacobian without going through autograd / sparse matmuls
    """
    def __init__(self, fd_graphs: dict[tuple, DerivGraph], eq_mask: torch.Tensor, grad_mask: torch.Tensor, deriv_orders: dict[int, list], N_us_tot, device="cpu"):
        """
            deriv_orders: dict[point_idx, list[deriv_position]]. Derivative order for each derivative BC
        """
        # Construct all required derivatives and jacobian.
        super().__init__(fd_graphs, eq_mask, grad_mask, N_us_tot, device=device)

        # Build up full derivative matrix, combining all derivatives. [sum_n(deriv_n)] u - N = 0
        deriv_mat = []
        for eq_idx, derivs in enumerate(deriv_orders.values()):
            deriv_row = []
            for deriv in derivs:
                deriv_row.append(self.fd_spms[deriv][eq_idx])

            deriv_row = torch.stack(deriv_row, dim=0)
            deriv_row_sum = torch.sparse.sum(deriv_row, dim=0)
            deriv_mat.append(deriv_row_sum)

        deriv_mat = torch.stack(deriv_mat, dim=0).coalesce()
        deriv_mat = deriv_mat.to_sparse_csr()          # shape = [N_eqs, N_us_tot]
        self.deriv_mat = CSRToInt32(deriv_mat)

        self.jac_mat = coo_col_select(deriv_mat, self.grad_mask).to_sparse_csr()   # shape = [N_eqs, N_grad]
        self.jac_mat = CSRToInt32(self.jac_mat)

        del self.fd_spms
        del self.jac_spms

    def derivative(self, Xs) -> torch.Tensor:
        neumann_resid = torch.mv(self.deriv_mat, Xs)
        return neumann_resid

    def jacobian(self) -> torch.Tensor:
        return self.jac_mat