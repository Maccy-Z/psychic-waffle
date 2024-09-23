import torch
import time


def benchmark_cpu_matmul(device, matrix_size=1000, num_iterations=100):
    """
    Benchmarks CPU performance for matrix multiplication using PyTorch.

    Args:
        matrix_size (int): The size of the square matrices to multiply.
        num_iterations (int): Number of multiplication iterations to perform.
    """
    # Ensure computations are on the CPU

    # Create two random matrices of the specified size
    A = torch.rand(matrix_size, matrix_size, device=device)
    B = torch.rand(matrix_size, matrix_size, device=device)

    # Warm-up run (optional but recommended)
    print("Warming up...")
    for _ in range(10):
        C = torch.matmul(A, B)

    # Start timing
    print(f"Starting benchmark: {num_iterations} iterations of {matrix_size}x{matrix_size} matrix multiplication.")
    start_time = time.time()

    for _ in range(num_iterations):
        C = torch.matmul(A, B)

    end_time = time.time()

    # Calculate total and average time
    total_time = end_time - start_time
    avg_time = total_time / num_iterations

    print(f"Total time for {num_iterations} iterations: {total_time:.6f} seconds")
    print(f"Average time per multiplication: {avg_time:.6f} seconds")


if __name__ == "__main__":
    # You can adjust the matrix size and number of iterations as needed
    device = torch.device('cpu')

    MATRIX_SIZE = 2000  # Size of the square matrices (e.g., 1000 for 1000x1000)
    NUM_ITERATIONS = 200  # Number of multiplication operations to benchmark

    benchmark_cpu_matmul(device, matrix_size=MATRIX_SIZE, num_iterations=NUM_ITERATIONS)
