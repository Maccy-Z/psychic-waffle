import time
from multiprocessing import Pool, cpu_count
import torch

def test_fn(a):
    a = torch.tensor([1, 2, 3])#.numpy()
    return a#, a


def main():
    args = [i for i in range(5000)]
    st = time.time()
    with Pool(processes=16) as p:
        out = p.starmap(test_fn, args)

    x = zip(*out)
    print(f"Time: {time.time() - st:.3f}")

if __name__ == "__main__":
    main()