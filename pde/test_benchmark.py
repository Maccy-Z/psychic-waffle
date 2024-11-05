import torch

from pde.graph_grid.graph_store import Point, P_Types
from pde.graph_grid.U_graph import UGraph
from pde.graph_grid.graph_utils import test_grid, gen_perim, plot_interp_graph
from pde.config import Config
from pde.NeuralPDE_Graph import NeuralPDEGraph
from pdes.PDEs import Poisson, LearnedFunc
from pde.utils import setup_logging
from pde.loss import DummyLoss


import itertools
import copy

def flatten_search_space(search_space, parent_key='', sep='.'):
    """
    Recursively flattens a nested search space into a dictionary with dot-separated keys.

    Args:
        search_space (dict): The nested search space dictionary.
        parent_key (str): The base key string for recursion.
        sep (str): Separator for nested keys.

    Returns:
        dict: A flattened dictionary with keys as paths and values as lists of possible values.
    """
    items = {}
    for key, value in search_space.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_search_space(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items

def set_nested_value(config, key_path, value, sep='.'):
    """
    Sets a value in a nested dictionary given a dot-separated key path.

    Args:
        config (dict): The configuration dictionary to update.
        key_path (str): The dot-separated key path (e.g., 'model.learning_rate').
        value: The value to set.
        sep (str): Separator used in the key path.

    Returns:
        None: The config is updated in place.
    """
    keys = key_path.split(sep)
    d = config
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def generate_configurations(base_config, search_space, sep='.'):
    """
    Generates a list of configuration dictionaries by varying parameters in the search space.

    Args:
        base_config (dict): The base configuration dictionary.
        search_space (dict): A nested dictionary specifying parameters to vary.
        sep (str): Separator used for nested keys.

    Returns:
        list of dict: A list containing all possible configuration dictionaries.
    """
    # Flatten the search space to handle nested parameters
    flat_search_space = flatten_search_space(search_space, sep=sep)

    # Extract parameter names and their respective lists of values
    param_keys = list(flat_search_space.keys())
    param_values = list(flat_search_space.values())

    # Compute the Cartesian product of all parameter values
    all_combinations = list(itertools.product(*param_values))

    config_list = []
    for combination in all_combinations:
        # Deep copy the base configuration to avoid mutating it
        new_config = copy.deepcopy(base_config)

        # Update the new configuration with the current combination of parameters
        for key, value in zip(param_keys, combination):
            set_nested_value(new_config, key, value, sep=sep)

        config_list.append(new_config)

    return config_list, all_combinations

def find_pareto_optimal_indices(evaluation_results):
    """
    Identifies the indices of Pareto-optimal configurations based on minimizing time and loss.

    Args:
        evaluation_results (list of tuples): Each tuple contains (time_taken, loss).

    Returns:
        list: Indices of Pareto-optimal configurations.
    """
    import numpy as np

    # Convert to a NumPy array for efficient processing
    data = np.array(evaluation_results)

    # Extract time and loss columns
    times = data[:, 0]
    losses = data[:, 1]

    # Sort the configurations by time in ascending order
    sorted_indices = np.argsort(times)
    sorted_times = times[sorted_indices]
    sorted_losses = losses[sorted_indices]

    pareto_indices = []
    min_loss_so_far = np.inf  # Initialize with infinity since we want to minimize loss

    # Iterate through the sorted configurations
    for idx, (time, loss) in zip(sorted_indices, zip(sorted_times, sorted_losses)):
        if loss <= min_loss_so_far:
            pareto_indices.append(int(idx))
            min_loss_so_far = loss
        # Else, the configuration is dominated and not Pareto-optimal

    # Sort the Pareto indices for readability (optional)
    # pareto_indices.sort()

    return pareto_indices


def load_graph():
    u_graph = torch.load("save_u_graph.pth")
    return u_graph

base_solve_cfg = {
    "config_version": 2,
    "determinism_flag": 0,
    "exception_handling": 1,

    "solver": {
        "monitor_residual": 1,
        "solver": "FGMRES",  # "PBICGSTAB", #
        "convergence": "RELATIVE_INI_CORE",
        "tolerance": 1e-2,
        "max_iters": 100,
        "gmres_n_restart": 100,
        "preconditioner": "NOSOLVER",

        "preconditioner": {
            "smoother": {"solver": "JACOBI_L1",
                         "relaxation_factor": 1.9,
                         },
            "solver": "AMG",
            "coarse_solver": "DENSE_LU_SOLVER",
            "algorithm": "AGGREGATION",
            "selector": "SIZE_8",
            "max_iters": 2,
            "presweeps": 10,
            "postsweeps": 10,
            "cycle": "V",
            "max_levels": 3,
        },
    }
}
test_params = {"solver": {"max_iters": [80, 90, 100],
                       "preconditioner": {"smoother": {"relaxation_factor": [1.5, 1.75, 1.9]},
                                          "selector": ["SIZE_4", "SIZE_8"],
                                          "presweeps": [9, 10, 12],
                                          #"postsweeps": [5, 7, 10],
                                          "max_levels": [2, 3, 4],
                                          }
                       }
               }



def main():
    test_cfgs, all_cfgs = generate_configurations(base_solve_cfg, test_params)

    cfg = Config()
    pde_fn = Poisson(cfg, device=cfg.DEVICE)
    u_graph = load_graph()

    log_vals = []
    for test_cfg, test_vals in zip(test_cfgs, all_cfgs):
        torch.cuda.empty_cache()
        test_cfg["solver"]["preconditioner"]["postsweeps"] = test_cfg["solver"]["preconditioner"]["presweeps"]
        cfg.fwd_cfg.lin_solve_cfg = test_cfg

        u_graph.reset()
        pde_adj = NeuralPDEGraph(pde_fn, u_graph, cfg, DummyLoss())

        pde_adj.forward_solve()

        log = pde_adj.newton_solver.logging
        t = log['time']
        residual = log['residual'].item()
        print(f'{test_vals = }, {t = :.3g}, {residual = :.3g}')
        log_vals.append((t, residual))

        pde_adj.newton_solver.lin_solver.amgx_solver.__del__()

    pareto_indices = find_pareto_optimal_indices(log_vals)
    print()
    print()

    for idx in pareto_indices:
        print(f'{all_cfgs[idx]}: t = {log_vals[idx][0]:.3g}, residual = {log_vals[idx][1]:.3g}')

if __name__ == "__main__":
    # Sample evaluation results: (time_taken, loss)
    setup_logging()
    torch.manual_seed(0)
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=200)
    main()

