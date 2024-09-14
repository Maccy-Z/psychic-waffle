import torch
from pde.U_grid import UGrid, USplitGrid, UNormalGrid
from pde.pdes.PDE_utils import PDEHandler
from pde.utils import get_split_indices

class JacobCalc:
    """ Used to calculate Jacobian of PDE.
    """
    def __init__(self, sol_grid: UGrid, pde_func:PDEHandler):
        self.sol_grid = sol_grid
        self.pde_func = pde_func

    def jacobian(self):
        """ Returns Jacobain of function. Return shape: [N_pde, N_us].
            PDEs are indexed by pde_mask and Us are indexed by us_grad_mask. 2D coordinates are stacked into 1D here.
            Columns for each function. Row for each u value.

         """
        us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

        subgrid = UNormalGrid(self.sol_grid, us_grad_mask, pde_mask)

        us_grad = subgrid.get_us_grad()
        jacobian, residuals = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # N equations, N+2 Us, jacob.shape: [N^d, N^d]


        return jacobian, residuals



class SplitJacobCalc:
    """ Compute Jacobian in split blocks for efficiency. """
    def __init__(self, sol_grid: UGrid, pde_func:PDEHandler, num_blocks: int,):
        self.device = sol_grid.device

        self.sol_grid = sol_grid
        self.pde_func = pde_func

        # Sparsity pattern of Jacobian and splitting. Always fixed
        self.jacob_shape = self.sol_grid.N_us_train.item()  # == len(torch.nonzero(us_grad_mask))
        self.num_blocks = num_blocks
        self.split_idxs = get_split_indices(self.jacob_shape, num_blocks)
        self.block_size = self.jacob_shape // self.num_blocks
        self.us_stride = self.sol_grid.N[1] + 1 # Stride of us in grid. Second element of N is the number of points in x direction. Overestimate.

    def jacobian(self):
        # Build up Jacobian from sections of PDE and us for efficiency.
        jacob_shape = self.jacob_shape
        us, us_grad_mask, pde_mask = self.sol_grid.get_us_mask()

        # Rectangular blocks of Jacobian between [xmin, xmax] (pde) and [ymin, ymax] (us)
        jacobian = torch.zeros((jacob_shape, jacob_shape), device=self.device)
        residuals = torch.zeros(jacob_shape, device=self.device)

        for xmin, xmax in self.split_idxs:
            # Diagonal blocks of Jacobian
            ymin = torch.clip(xmin - self.us_stride, 0, jacob_shape)
            ymax = torch.clip(xmin + self.block_size + self.us_stride, 0, jacob_shape)

            pde_slice = slice(xmin, xmax)
            us_slice = slice(ymin, ymax)

            # Subset of pde equations
            pde_true_idx = torch.nonzero(pde_mask)
            pde_true_idx = pde_true_idx[pde_slice]

            pde_submask = torch.zeros_like(pde_mask)
            pde_submask[pde_true_idx[:, 0], pde_true_idx[:, 1]] = True

            # Subset of us
            us_grad_idx = torch.nonzero(us_grad_mask)
            want_us_idx = us_grad_idx[us_slice]

            us_grad_submask = torch.zeros_like(us_grad_mask)
            us_grad_submask[want_us_idx[:, 0], want_us_idx[:, 1]] = True

            # Further clip region PDE is calculated to around pde_mask and us_grad_mask to avoid unnecessary calculations
            nonzero_idx = torch.nonzero(us_grad_submask)
            a, b = torch.min(nonzero_idx, dim=0)[0], torch.max(nonzero_idx, dim=0)[0]
            a, b = a - 1, b + 1  # Add one point of padding. This *should* always be enough depending on bc and pde_mask
            a, b = torch.clamp(a, min=0), torch.clamp(b, min=0)
            us_region_mask = (slice(a[0], b[0] + 1), slice(a[1], b[1] + 1))

            subgrid = USplitGrid(self.sol_grid, us_region_mask, us_grad_submask, pde_submask)

            # Get Jacobian
            us_grad = subgrid.get_us_grad()  # us[region][grad_mask]
            jacob, resid = torch.func.jacfwd(self.pde_func.residuals, has_aux=True, argnums=0)(us_grad, subgrid)  # jacob.shape: [block_size, block_size+stride]

            # Fill in calculated parts
            jacobian[pde_slice, us_slice] = jacob
            residuals[pde_slice] = resid.flatten()

        return jacobian, residuals


class SparseJacobianBuilder:
    def __init__(self, shape):
        """
        Initialize the SparseJacobianBuilder.

        :param shape: Tuple indicating the shape of the Jacobian tensor (e.g., (rows, cols)).
        """
        self.shape = shape
        self.indices = []
        self.values = []

    def add_block(self, pde_slice, us_slice, jacob_block):
        """
        Add a block to the Jacobian tensor.

        :param pde_slice: A slice object indicating the rows for the block.
        :param us_slice: A slice object indicating the columns for the block.
        :param jacob_block: A tensor containing the block values to be added.
        """
        # Calculate the indices for the block
        block_indices = torch.cartesian_prod(
            torch.arange(pde_slice.start, pde_slice.stop, device=jacob_block.device),
            torch.arange(us_slice.start, us_slice.stop, device=jacob_block.device)
        )

        # Append indices and values
        self.indices.append(block_indices.t())
        self.values.append(jacob_block.flatten())

    def build_sparse_tensor(self):
        """
        Finalize and retrieve the full sparse Jacobian tensor.

        :return: A sparse COO tensor representing the Jacobian.
        """

        # Concatenate all indices and values
        indices = torch.cat(self.indices, dim=1)
        values = torch.cat(self.values)

        # Create and return the sparse tensor
        return torch.sparse_coo_tensor(indices, values, self.shape).to_sparse_csr()
