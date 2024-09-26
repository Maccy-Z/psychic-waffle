import torch
import pyamgx

pyamgx.initialize()

# Initialize config and resources:
cfg = pyamgx.Config().create_from_dict({
   "config_version": 2,
        "determinism_flag": 0,
        "exception_handling" : 1,
        "solver": {
            "monitor_residual": 1,
            "print_solve_stats": 1,
            "solver": "PBICGSTAB",
            "convergence": "RELATIVE_INI_CORE",
            "preconditioner": {
                "solver": "NOSOLVER"
        }
    }
})

rsc = pyamgx.Resources().create_simple(cfg)

# Create matrices and vectors:
A = pyamgx.Matrix().create(rsc, mode="dFFI")
b = pyamgx.Vector().create(rsc, mode="dFFI")
x = pyamgx.Vector().create(rsc, mode="dFFI")

# Create solver:
solver = pyamgx.Solver().create(rsc, cfg, mode="dFFI")
# Upload system:
M = torch.rand(5, 5, dtype=torch.float32, device='cuda').to_sparse_csr()
rhs = torch.rand(5, device="cuda", dtype=torch.float32) #np.random.rand(5).astype(np.float32)
sol = torch.zeros(5, device="cuda", dtype=torch.float32)

A.upload_csr_torch(M)
b.upload_torch(rhs)
x.upload_torch(sol)

# Setup and solve system:
solver.setup(A)
solver.solve(b, x)
# Download solution
x.download_torch(sol)
print("pyamgx solution: ", sol)
print("pytorch solution : ", torch.linalg.solve(M.to_dense(), rhs))
print()

# Clean up:
A.destroy()
x.destroy()
b.destroy()
solver.destroy()
rsc.destroy()
cfg.destroy()
pyamgx.finalize()