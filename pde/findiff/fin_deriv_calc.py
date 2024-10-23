import torch
from torch_geometric.nn import MessagePassing

from pde.graph_grid.graph_store import DerivGraph
from pde.BaseDerivCalc import BaseDerivCalc

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
    def __init__(self, fd_graphs: dict[tuple, DerivGraph], pde_mask: torch.Tensor, grad_mask: torch.Tensor, N_us_tot, device="cpu"):
        """ Initialise sparse matrices from finite difference graphs.
            Compute d = A * u [pde_mask]. Compile pde_mask into A.
            Jacobian is A[pde_mask][us_mask]
        """
        self.pde_mask = pde_mask
        self.grad_mask = grad_mask
        self.device = device

        self.fd_spms = {}
        self.jac_spms = []      # shape = [N_diff], [N_pde, N_points]

        # Order (0, 0) is original node value
        only_us = torch.eye(N_us_tot, device=self.device).to_sparse_coo()
        only_us = self.coo_row_select(only_us)
        only_us = self.coo_col_select(only_us)
        self.jac_spms.append(only_us.to_sparse_csr())   # shape = [N_diff], [N_pde, N_points]
        for order, graph in fd_graphs.items():
            edge_idx = graph.edge_idx
            edge_coeff = graph.weights
            sp_mat = torch.sparse_coo_tensor(edge_idx, edge_coeff.squeeze(), (N_us_tot, N_us_tot)).T.coalesce()
            sp_mat = self.coo_row_select(sp_mat)        # shape = [N_pde, N_points]
            sp_mat_fwd = sp_mat.to_sparse_csr()
            self.fd_spms[order] = sp_mat_fwd

            sp_mat_jac = self.coo_col_select(sp_mat)   # shape = [N_pde, N_grad]
            sp_mat_jac = sp_mat_jac.to_sparse_csr()
            self.jac_spms.append(sp_mat_jac)


    def derivative(self, Xs) -> dict[tuple, torch.Tensor]:
        """ Xs.shape = [N_points]
            spm.shape = [N_pde, N_points]
        """
        derivatives = {(0, 0): Xs[self.pde_mask]}

        for order, spm in self.fd_spms.items():
            derivatives[order] = torch.mv(spm, Xs)
        return derivatives

    def jacobian(self) -> list[torch.FloatTensor]:
        """ Linear transform, so jacobian is the same as the sparse matrix.
            return.shape: [N_diff], [N_pde, N_points]
         """
        return self.jac_spms

    def coo_row_select(self, sparse_coo: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Selects rows from a COO sparse tensor based on a row-wise mask.
        Args:
            sparse_coo (torch.sparse_coo_tensor): The input sparse COO tensor.
        Returns:
            torch.sparse_coo_tensor: A new sparse COO tensor with only the selected rows.
        """
        # Extract indices and values from the sparse tensor
        indices = sparse_coo.coalesce().indices()  # Shape: [ndim, nnz]
        values = sparse_coo.coalesce().values()  # Shape: [nnz]

        # Assume the first dimension corresponds to rows
        row_indices = indices[0]

        # Create a mask for non-zero elements in the selected rows
        mask = self.pde_mask[row_indices]

        # Apply the mask to filter indices and values
        selected_indices = indices[:, mask]
        selected_values = values[mask]

        # Get the selected row numbers in sorted order
        selected_rows = self.pde_mask.nonzero(as_tuple=False).squeeze()

        # Ensure selected_rows is 1D
        if selected_rows.dim() == 0:
            selected_rows = selected_rows.unsqueeze(0)

        # Create a mapping from old row indices to new row indices
        # This ensures that the new tensor has contiguous row indices starting from 0
        # Example: If rows 1 and 3 are selected, row 1 -> 0 and row 3 -> 1 in the new tensor
        row_mapping = torch.arange(len(selected_rows), device=selected_rows.device)
        # Create a dictionary-like mapping using scatter
        mapping = torch.full((sparse_coo.size(0),), -1, dtype=torch.long, device=selected_rows.device)
        mapping[selected_rows] = row_mapping
        # Map the selected row indices
        new_row_indices = mapping[selected_indices[0]]

        if (new_row_indices == -1).any():
            raise RuntimeError("Some row indices were not mapped correctly.")

        # Replace the row indices with the new row indices
        new_indices = selected_indices.clone()
        new_indices[0] = new_row_indices

        # Define the new size: number of selected rows and the remaining dimensions
        new_size = [self.pde_mask.sum().item()] + list(sparse_coo.size())[1:]

        # Create the new sparse COO tensor
        new_sparse_coo = torch.sparse_coo_tensor(new_indices, selected_values, size=new_size)

        return new_sparse_coo

    def coo_col_select(self, sparse_coo: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
        """
        Selects columns from a COO sparse tensor based on a column-wise mask.
        Args:
            sparse_coo (torch.sparse_coo_tensor): The input sparse COO tensor.
        Returns:
            torch.sparse_coo_tensor: A new sparse COO tensor with only the selected columns.
        """
        # Extract indices and values from the sparse tensor
        sparse_coo = sparse_coo.coalesce()  # Ensure indices are coalesced
        indices = sparse_coo.indices()      # Shape: [ndim, nnz]
        values = sparse_coo.values()        # Shape: [nnz]

        # Assume the second dimension corresponds to columns
        col_indices = indices[1]

        # Create a mask for non-zero elements in the selected columns
        mask = self.grad_mask[col_indices]

        # Apply the mask to filter indices and values
        selected_indices = indices[:, mask]
        selected_values = values[mask]

        # Get the selected column numbers in sorted order
        selected_cols = self.grad_mask.nonzero(as_tuple=False).squeeze()

        # Ensure selected_cols is 1D
        if selected_cols.dim() == 0:
            selected_cols = selected_cols.unsqueeze(0)

        # Create a mapping from old column indices to new column indices
        # This ensures that the new tensor has contiguous column indices starting from 0
        row_mapping = torch.arange(len(selected_cols), device=selected_cols.device)
        # Initialize a mapping tensor with -1 (invalid)
        mapping = torch.full((sparse_coo.size(1),), -1, dtype=torch.long, device=selected_cols.device)
        # Assign new indices to the selected columns
        mapping[selected_cols] = row_mapping
        # Map the selected column indices
        new_col_indices = mapping[selected_indices[1]]

        if (new_col_indices == -1).any():
            raise RuntimeError("Some column indices were not mapped correctly.")

        # Replace the column indices with the new column indices
        new_indices = selected_indices.clone()
        new_indices[1] = new_col_indices

        # Define the new size: number of rows remains the same, number of selected columns
        new_size = list(sparse_coo.size())
        new_size[1] = self.grad_mask.sum().item()

        # Create the new sparse COO tensor
        new_sparse_coo = torch.sparse_coo_tensor(new_indices, selected_values, size=new_size)

        return new_sparse_coo