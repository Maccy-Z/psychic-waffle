import atexit
import logging

import torch
import pyamgx
import cupy as cp

# Configure logging for debugging purposes
logger = logging.getLogger("pyamgx_handler")


class PyAMGXSolver:
    """ Solve Ax = b using AMGX """
    # Keep handle for destuctor
    A = None
    b = None
    x = None

    def __init__(self, cfg: dict):
        """ Initialise the AMGXSolver and create the necessary objects """
        self.cfg = pyamgx.Config().create_from_dict(cfg)
        self.resources = pyamgx.Resources().create_simple(self.cfg)
        self.mode = "dFFI"
        self.solver = pyamgx.Solver().create(self.resources, self.cfg, mode=self.mode)

        # Variable hodlers
        A = pyamgx.Matrix().create(self.resources, mode=self.mode)
        b = pyamgx.Vector().create(self.resources, mode=self.mode)
        x = pyamgx.Vector().create(self.resources, mode=self.mode)

        self.A, self.b, self.x = A, b, x

        self.setup = False

    def init_solver_cp(self, cp_matrix):
        """ Initialise problem matrix A from cupy sparse csr"""

        if self.setup:
            self.A.replace_coefficients(cp_matrix.data)
        else:
            # print(f'{cp_matrix.nnz = }')
            self.A.upload_CSR(cp_matrix)
            self.solver.setup(self.A)
            # self.setup = True

    def init_solver(self, tensor):
        """ Initialise problem matrix A from sparse csr cuda tensor """
        self.A.upload_csr_torch(tensor)
        self.solver.setup(self.A)

    def solve(self, b, x):
        """ Solve the system Ax = b """
        # b = cp.from_dlpack(b)
        # x_cp = cp.from_dlpack(x)
        # self.b.upload(b)
        # self.x.upload(x_cp)

        self.b.upload_torch(b)
        self.x.upload_torch(x)

        self.solver.solve(self.b, self.x)

        self.x.download_torch(x)
        # x = self.x.download_torch_zerocopy()

        return x

    def __del__(self):
        logger.info("Destroying solver AMGX objects.")
        self.solver.destroy()

        try:
            self.A.destroy()
            self.x.destroy()
            self.b.destroy()
        except AttributeError as e:
            print(e)

        # Created during init
        # TODO: Fix this
        # self.resources.destroy()
        self.cfg.destroy()


class PyAMGXManager:
    """ Hodler to ensure that AMGX is initialized and finalized """
    _instance = None
    _initialized = False

    mode: str = "dFFI"
    solvers = []

    def __new__(cls):
        if cls._instance is None:
            logger.debug("Creating a new pyAMGXManager instance.")
            cls._instance = super(PyAMGXManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self.__class__._initialized:
            self.initialize_library()

            atexit.register(self.destructor)
            self.__class__._initialized = True
        else:
            logger.debug("ExternalLibraryManager already initialized.")

    def initialize_library(self):
        # Replace with actual library initialization code
        logger.debug("AMGX initialized.")
        pyamgx.initialize()

    def destructor(self):
        try:
            logger.info("AMGX destructor called.")
            for s in self.solvers:
                s.__del__()

            pyamgx.finalize()
        except Exception as e:
            print("OHNO")
            print(e)
            pass

    def create_solver(self, cfg: dict) -> PyAMGXSolver:
        solver = PyAMGXSolver(cfg)
        self.solvers.append(solver)
        return solver


# Usage Example
def main():
    from copy import deepcopy

    cfg_dict = {
            "config_version": 2,
            "determinism_flag": 0,
            "exception_handling": 1,
            "solver": {
                "monitor_residual": 1,
                "print_solve_stats": 1,
                "solver": "GMRES",
                "convergence": "RELATIVE_INI_CORE",
                "preconditioner": {
                    "solver": "NOSOLVER"
                }
            }
        }

    amgx_manager = PyAMGXManager()
    solver = amgx_manager.create_solver(cfg_dict)
    solver2 = amgx_manager.create_solver(deepcopy(cfg_dict))

    A = torch.rand(5, 5, dtype=torch.float32, device='cuda').to_sparse_csr()
    b = torch.rand(5, device="cuda", dtype=torch.float32)
    x = torch.zeros(5, device="cuda", dtype=torch.float32)

    solver.init_solver(A)
    x = solver.solve(b, x)

    print("AMGX solution: ", x)
    print("pytorch solution : ", torch.linalg.solve(A.to_dense(), b))

if __name__ == "__main__":
    main()
