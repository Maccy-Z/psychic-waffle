import torch
import cupy as cp
from torch.autograd.functional import jacobian

from pde.BaseU import UBase
from pde.cartesian_grid.U_grid import USplitGrid, UNormalGrid
from pde.cartesian_grid.PDE_Grad import PDEForward
from pde.utils import get_split_indices, clamp
from pde.BasePDEGrad import PDEFwdBase

def get_jac_calc(us_grid: UBase, pde_forward: PDEForward, cfg):
    if cfg.jac_mode == "dense":
        jacob_calc = JacobCalc(us_grid, pde_forward)
    elif cfg.jac_mode == "split":
        jacob_calc = SplitJacobCalc(us_grid, pde_forward, num_blocks=cfg.num_blocks)
    elif cfg.jac_mode == "sparse":
        jacob_calc = SparseJacobCalc(us_grid, pde_forward, num_blocks=cfg.num_blocks)
    else:
        raise ValueError(f"Invalid jac_mode {cfg.jac_mode = }. Must be 'dense' or 'split'")

    return jacob_calc


class JacobCalc:
    """
        Calculate Jacobian of PDE.
    """
    def __init__(self, sol_grid: UBase, pde_func:PDEFwdBase):
        self.sol_grid = sol_grid
        self.pde_func = pde_func

        self.device = sol_grid.device

    def jacobian(self):
        """ Returns Jacobain of function. Return shape: [N_pde, N_us].
            PDEs are indexed by pde_mask and Us are indexed by us_grad_mask. 2D coordinates are stacked into 1D here.
            Columns for each function. Row for each u value.
         """
        # us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()
        #
        # subgrid = UNormalGrid(self.sol_grid, us_grad_mask, pde_mask)
        #
        # us_grad = subgrid.get_us_grad()
        # jacobian, residuals = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]
        us_grad = self.sol_grid.get_us_grad()
        # jacobian, residuals = torch.func.jacrev(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, self.sol_grid)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]
        jacobian, residuals = self.pde_func.jac_block(us_grad, self.sol_grid)

        return jacobian, residuals


class SplitJacobCalc(JacobCalc):
    """
        Compute Jacobian in split blocks for efficiency.
    """
    def __init__(self, sol_grid: UBase, pde_func:PDEForward, num_blocks: int,):
        self.sol_grid = sol_grid
        self.pde_func = pde_func
        self.device = sol_grid.device

        # Sparsity pattern of Jacobian and splitting. Always fixed
        self.jacob_shape = self.sol_grid.N_us_fit  # == len(torch.nonzero(us_grad_mask))
        self.num_blocks = num_blocks
        self.split_idxs = get_split_indices(self.jacob_shape, num_blocks)
        self.block_size = self.jacob_shape // self.num_blocks
        self.us_stride = (self.sol_grid.N[1] + 1).item() # Stride of us in grid. Second element of N is the number of points in x direction. Overestimate.

    def jacobian(self):
        # Build up Jacobian from sections of PDE and us for efficiency.
        jacob_shape = self.jacob_shape
        us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

        # Rectangular blocks of Jacobian between [xmin, xmax] (pde) and [ymin, ymax] (us)
        # Make new masks for pde and us for each block.
        jacobian = torch.zeros((jacob_shape, jacob_shape), device=self.device)
        residuals = torch.zeros(jacob_shape, device=self.device)

        # Indices of non-zero elements in pde_mask and us_grad_mask
        pde_true_idx, us_grad_idx = self.sol_grid.mask_nonzero_idx()

        for xmin, xmax in self.split_idxs:      # (pde_min, pde_max)
            # Diagonal blocks of Jacobian
            ymin = clamp(xmin - self.us_stride, 0, jacob_shape)         # (us_min, us_max)
            ymax = clamp(xmin + self.block_size + self.us_stride, 0, jacob_shape)

            pde_slice = slice(xmin, xmax)
            us_slice = slice(ymin, ymax)

            # Subset of pde equations
            want_pde_idx = pde_true_idx[pde_slice]
            pde_submask = torch.zeros_like(pde_mask)
            pde_submask[want_pde_idx[:, 0], want_pde_idx[:, 1]] = True

            # Subset of us
            want_us_idx = us_grad_idx[us_slice]
            us_grad_submask = torch.zeros_like(us_grad_mask)
            us_grad_submask[want_us_idx[:, 0], want_us_idx[:, 1]] = True

            # Further clip region PDE is calculated to around pde_mask and us_grad_mask to avoid unnecessary calculations
            want_us_idx = want_us_idx.cpu() # Faster on CPU
            a, b = torch.min(want_us_idx, dim=0)[0], torch.max(want_us_idx, dim=0)[0]
            a, b = a - 1, b + 1  # Add one point of padding. This *should* always be enough depending on bc and pde_mask
            a = torch.clamp(a, min=0)
            us_region_mask = (slice(a[0], b[0] + 1), slice(a[1], b[1] + 1))

            subgrid = USplitGrid(self.sol_grid, us_region_mask, us_grad_submask, pde_submask)

            # Get Jacobian
            us_grad = subgrid.get_us_grad()  # us[region][grad_mask]
            jacob, resid = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # jacob.shape: [block_size, block_size+stride]

            # Fill in calculated parts
            jacobian[pde_slice, us_slice] = jacob
            residuals[pde_slice] = resid.flatten()

        return jacobian, residuals


class SparseJacobCalc(JacobCalc):
    """
        Compute Jacobian in split blocks for efficiency.
    """
    def __init__(self, sol_grid: UBase, pde_func:PDEForward, num_blocks: int,):
        self.sol_grid = sol_grid
        self.pde_func = pde_func
        self.device = sol_grid.device

        # Sparsity pattern of Jacobian and splitting. Always fixed
        self.jacob_shape = self.sol_grid.N_us_grad  # == len(torch.nonzero(us_grad_mask))
        self.num_blocks = num_blocks
        self.split_idxs = get_split_indices(self.jacob_shape, num_blocks)
        self.block_size = self.jacob_shape // self.num_blocks
        self.us_stride = (self.sol_grid.N[1] + 1).item() # Stride of us in grid. Second element of N is the number of points in x direction. Overestimate.

        self.csr_jac_builder = CSR_Builder(self.jacob_shape, self.jacob_shape, device=self.device)

    def jacobian(self):
        # Build up Jacobian from sections of PDE and us for efficiency.
        jacob_shape = self.jacob_shape
        us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

        # Rectangular blocks of Jacobian between [xmin, xmax] (pde) and [ymin, ymax] (us)
        # Make new masks for pde and us for each block.
        residuals = torch.zeros(jacob_shape, device=self.device)

        # Indices of non-zero elements in pde_mask and us_grad_mask
        pde_true_idx, us_grad_idx = self.sol_grid.mask_nonzero_idx()

        for xmin, xmax in self.split_idxs:      # (pde_min, pde_max)
            # Diagonal blocks of Jacobian
            ymin = clamp(xmin - self.us_stride, 0, jacob_shape)         # (us_min, us_max)
            ymax = clamp(xmin + self.block_size + self.us_stride, 0, jacob_shape)

            pde_slice = slice(xmin, xmax)
            us_slice = slice(ymin, ymax)

            # Subset of pde equations
            want_pde_idx = pde_true_idx[pde_slice]
            pde_submask = torch.zeros_like(pde_mask)
            pde_submask[want_pde_idx[:, 0], want_pde_idx[:, 1]] = True

            # Subset of us
            want_us_idx = us_grad_idx[us_slice]
            us_grad_submask = torch.zeros_like(us_grad_mask)
            us_grad_submask[want_us_idx[:, 0], want_us_idx[:, 1]] = True

            # Further clip region PDE is calculated to around pde_mask and us_grad_mask to avoid unnecessary calculations
            want_us_idx = want_us_idx.cpu() # Faster on CPU
            a, b = torch.min(want_us_idx, dim=0)[0], torch.max(want_us_idx, dim=0)[0]
            a, b = a - 1, b + 1  # Add one point of padding. This *should* always be enough depending on bc and pde_mask
            a = torch.clamp(a, min=0)
            us_region_mask = (slice(a[0], b[0] + 1), slice(a[1], b[1] + 1))

            subgrid = USplitGrid(self.sol_grid, us_region_mask, us_grad_submask, pde_submask)

            # Get Jacobian
            us_grad = subgrid.get_us_grad()  # us[region][grad_mask]
            jacob, resid = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # jacob.shape: [block_size, block_size+stride]

            # Fill in calculated parts
            self.csr_jac_builder.add_block(jacob, xmin, ymin)
            residuals[pde_slice] = resid.flatten()

        jacobian = self.csr_jac_builder.build()

        self.csr_jac_builder.reset()
        return jacobian, residuals


class CSR_Builder:
    """ Incrementally build a sparse CSR tensor from dense blocks. """
    def __init__(self, total_rows, total_cols, device=None):
        """
        Initializes the builder for a CSR sparse tensor.
        Parameters:
        - total_rows: int, total number of rows in the matrix.
        - total_cols: int, total number of columns in the matrix.
        - device: torch device (optional).
        """
        self.dtype = torch.int64
        self.total_rows = total_rows
        self.total_cols = total_cols
        self.device = device

        self.zero_ten = torch.tensor([0], dtype=self.dtype, device=self.device)

        # Internal storage for CSR components
        self.nnz_per_row = torch.tensor([0] * self.total_rows, device=self.device, dtype=self.dtype)  # Number of non-zero elements per row
        self.col_indices = []                # Column indices of non-zero elements
        self.values = []                     # Non-zero values

    def add_block(self, block_dense_values, block_row_offset, block_col_offset):
        """
        Adds a dense block to the CSR components using efficient tensor operations.

        Parameters:
        - block_dense_values: 2D tensor (n x m), dense block of values.
        - block_row_offset: int, the starting row index of the block in the overall matrix.
        - block_col_offset: int, the starting column index of the block in the overall matrix.
        """
        n, m = block_dense_values.shape
        crow_idxs, col_idxs, values = self.to_csr(block_dense_values)

        # Count non-zero elements per row in the block
        counts = crow_idxs[1:] - crow_idxs[:-1]

        # Update nnz_per_row for the corresponding global rows
        # Using scatter_add for efficient batch updates
        self.nnz_per_row[block_row_offset:block_row_offset + n] += counts

        # Calculate global column indices
        global_cols = col_idxs + block_col_offset
        self.col_indices.append(global_cols)

        # Extract the non-zero values
        non_zero_values = values
        self.values.append(non_zero_values)

    def build(self):
        """
        Builds and returns the sparse CSR tensor from the accumulated components.

        Returns:
        - csr_tensor: torch.sparse_csr_tensor, the constructed sparse CSR tensor.
        """
        # Compute crow_indices by cumulatively summing nnz_per_row
        crow_indices = torch.cat([
            self.zero_ten,
            torch.cumsum(self.nnz_per_row, dim=0, dtype=self.dtype)
        ])

        # Convert col_indices and values to tensors
        col_indices_tensor = torch.cat(self.col_indices).to(self.dtype)
        values_tensor = torch.cat(self.values)

        # Create the sparse CSR tensor
        csr_tensor = torch.sparse_csr_tensor(
            crow_indices,
            col_indices_tensor,
            values_tensor,
            size=(self.total_rows, self.total_cols),
        )
        return csr_tensor

    def reset(self):
        self.nnz_per_row = torch.tensor([0] * self.total_rows, device=self.device, dtype=torch.int32)  # Number of non-zero elements per row
        self.col_indices = []                # Column indices of non-zero elements
        self.values = []                     # Non-zero values

    def to_csr(self, A_torch):
        """ Cupy is faster than torch """
        A_cp = cp.asarray(A_torch)
        A_csr_cp = cp.sparse.csr_matrix(A_cp)

        crow_indices = torch.from_dlpack(A_csr_cp.indptr)
        col_indices = torch.from_dlpack(A_csr_cp.indices)
        values = torch.from_dlpack(A_csr_cp.data)
        return crow_indices, col_indices, values