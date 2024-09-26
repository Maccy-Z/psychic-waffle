import torch
import gc


def process_and_delete_tensor(tensor):
    # Convert to sparse CSR format
    sparse_tensor = tensor.to_sparse_csr()

    # Process the sparse tensor if needed...

    # Delete the original dense tensor
    del tensor

    # Optionally free up GPU memory if the tensor is on the GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force garbage collection (optional, may help with large objects)
    gc.collect()

    return sparse_tensor


# Create a large tensor and pass it to the function
dense_tensor = torch.randn(10000, 10000, device='cuda')  # Large tensor on GPU
sparse_tensor = process_and_delete_tensor(dense_tensor)

print(dense_tensor)