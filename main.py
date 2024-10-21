import torch
from functools import lru_cache, wraps







# Example usage
@lru_cache_tensor(maxsize=256)
def compute_expensive_operation(tensor: torch.Tensor, multiplier: float) -> torch.Tensor:
    """
    An example function that performs an expensive computation on a tensor.
    """

    print(f"Computing {multiplier = } ")
    # Simulate an expensive operation
    return tensor * multiplier + torch.sin(tensor)


# Example calls
if __name__ == "__main__":
    t1 = torch.randn(100, 100)
    t2 = torch.randn(100, 100)

    result1 = compute_expensive_operation(t1, 2.5)  # Cached
    result2 = compute_expensive_operation(t1, 2.5)  # Retrieved from cache
    result3 = compute_expensive_operation(t2, 3.0)  # Cached
