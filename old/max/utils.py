import scipy
import torch
import math


class ParamHolder(torch.nn.Module):
    """Holds parameters, with constraint that parameters are positive using a softplus transformation."""
    raw_kern_len: torch.nn.Parameter
    raw_kern_scale: torch.nn.Parameter
    raw_noise: torch.nn.Parameter

    def __init__(self, kern_len=math.log(2), kern_scale=math.log(2), noise_var=math.log(2)):
        super().__init__()

        self.raw_kern_len = torch.nn.Parameter(self.inv_softplus(torch.tensor(kern_len)))
        self.raw_kern_scale = torch.nn.Parameter(self.inv_softplus(torch.tensor(kern_scale)))

        if noise_var is None:
            self.raw_noise = torch.nn.Parameter(self.inv_softplus(torch.tensor(math.log(2))))
        else:
            self.raw_noise = torch.nn.Parameter(self.inv_softplus(torch.tensor(noise_var)), requires_grad=False)

    # Return parameters in their original scale
    def get_params(self):
        kern_len = torch.nn.functional.softplus(self.raw_kern_len) + 1e-4
        kern_scale = torch.nn.functional.softplus(self.raw_kern_scale) + 1e-4
        noise = torch.nn.functional.softplus(self.raw_noise) + 1e-6
        return kern_len, kern_scale, noise

    @staticmethod
    def inv_softplus(x: torch.Tensor):
        if x > 20:
            return x
        else:
            return torch.log(torch.exp(x) - 1)


class KDTreeDict:
    """
        Modified dict with:
            Nearest neighbour search using KDTree
            Sorting observations by x or t

            keys, values are always sorted by order of insertion
    """

    def __init__(self):
        self.keys, self.values = [], []
        self.t = []
        self.tree = None
        self.dirty = False  # Flag to check if the tree needs rebuilding

    def add(self, key, value, t=0):
        self.keys.append([key])
        self.values.append(value)
        self.t.append(t)
        self.dirty = True  # Set flag to True as tree needs to be updated

    def build_tree(self):
        if self.dirty and self.keys:
            self.tree = scipy.spatial.KDTree(self.keys)
            self.dirty = False  # Reset flag as tree is up-to-date

    def find_nearest_key(self, key):
        self.build_tree()  # Ensure tree is built and up-to-date
        if self.tree is None:
            raise ValueError("The KDTreeDict is empty.")

        # print(f'{key = }')
        distance, index = self.tree.query([key])
        return self.keys[index][0]

    def __getitem__(self, key):
        nearest_key = self.find_nearest_key(key)
        return self.values[self.keys.index([nearest_key])]

    def __repr__(self):
        return '{' + ', '.join(f'{k[0]:.3g}: {repr(v)}' for k, v in zip(self.keys, self.values)) + '}'

    def items(self, sort_key="key"):
        if sort_key == "key":
            sorted_pairs = sorted(zip(self.keys, self.values), key=lambda pair: pair[0])
            return [(k[0], v) for k, v in sorted_pairs]

        elif sort_key == "t":
            combined = zip(self.keys, self.values, self.t)
            sorted_combined = sorted(combined, key=lambda item: item[2])
            return [(k[0], v) for k, v, t in sorted_combined]
        else:
            keys = [k[0] for k in self.keys]
            return zip(keys, self.values)

    def perturb(self, eps):
        self.values[0] = self.values[0] + eps
        #self.values[1] = self.values[1] + eps


# Print a string in color
def c_print(text, color: str):
    color_codes = {
        "black": "\033[30m",
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "reset": "\033[0m"  # Reset text color to default
    }

    code = color_codes[color]

    print(f"{code}{text}{color_codes['reset']}")


if __name__ == "__main__":
    ParamHolder()
