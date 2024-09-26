import numpy as np
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
            # "print_solve_stats": 1,
            "solver": "PBICGSTAB",
            "convergence": "RELATIVE_INI_CORE",
            "preconditioner": {
                "solver": "NOSOLVER"
        }
    }
})

rsc = pyamgx.Resources().create_simple(cfg)

# Create vectors:
b = pyamgx.Vector().create(rsc, mode="dFFI")
rhs = np.random.rand(22500).astype(np.float32)
b.upload(rhs)



b_retrieved = b.download_torch()
print(f'{b_retrieved = }')

b_zc = b.download_torch_zerocopy()
print(f'{b_zc = }')
# Download solution
print()
print("Setting zero")
b_new = torch.arange(22500, device='cuda', dtype=torch.float32)
print(f'{b_new = }')
b.upload_torch(b_new)
print(f'{b_retrieved = }')
print(f'{b_zc = }')

# Clean up:
b.destroy()
rsc.destroy()
cfg.destroy()
pyamgx.finalize()


"""
const void* get_cuda_pointer(AMGX_vector_handle vec)
{
    typedef Vector<typename TemplateMode<CASE>::Type> VectorLetterT;
    typedef CWrapHandle<AMGX_vector_handle, VectorLetterT> VectorW;

    VectorW wrapV(vec);
    VectorLetterT &v = *wrapV.wrapped();

    return v.raw();  // Return the raw CUDA pointer
}
"""