import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data

class FinDiffGrad(MessagePassing):
    """ Compute Grad^n(u) using finite differences.

     """
    def __init__(self):
        super().__init__(aggr='add')  # Aggregation method can be 'add', 'mean', etc.

    def forward(self, x, edge_index, edge_coeff):
        """
        Args:
            x (Tensor): Node feature matrix of shape [N, F].
            edge_index (LongTensor): Graph connectivity in COO format with shape [2, E].
            edge_coeff (Tensor): Edge coefficients of shape [E, 1] or [E, F].

        Returns:
            Tensor: Gradient for each node of shape [N, F].
        """
        return self.propagate(edge_index, x=x, edge_coeff=edge_coeff)

    def message(self, x_j, edge_coeff):
        """
        Constructs messages from source nodes to target nodes.

        Args:
            x_j (Tensor): Source node features for each edge of shape [E, F].
            edge_coeff (Tensor): Edge coefficients for each edge of shape [E, 1] or [E, F].

        Returns:
            Tensor: Messages to be aggregated.
        """

        print(f'{x_j[:3] = }')
        print(f'{edge_coeff[:3] = }')
        return  edge_coeff * x_j


def main():
    # Example graph with 5 nodes and 8 original edges
    N = 5  # Number of nodes
    E_original = 8  # Number of original edges

    # Node features (e.g., scalar values)
    x = torch.randn(N, 1)  # Shape: [N, F] where F=1

    # Original edge indices in COO format
    edge_index_original = torch.tensor([
        [0, 1, 2, 3, 4, 0, 1, 2],  # Source nodes
        [1, 2, 3, 4, 0, 2, 3, 4]  # Target nodes
    ], dtype=torch.long)

    # Original edge coefficients (one coefficient per edge)
    edge_coeff_original = torch.randn(E_original, 1)  # Shape: [E_original, 1]

    # Create self-loop edges
    self_loops = torch.arange(0, N, dtype=torch.long).unsqueeze(0).repeat(2, 1)  # Shape: [2, N]
    edge_index_self = self_loops  # Self-loop edges

    # Self-loop coefficients (precomputed weights, typically negative if previously subtracting f_i)
    # Example: Set self-loop coefficients to -1.0 for simplicity
    self_loop_coeff = torch.full((N, 1), -1.0)  # Shape: [N, 1]

    # Combine original edges with self-loop edges
    edge_index = torch.cat([edge_index_original, edge_index_self], dim=1)  # Shape: [2, E_total]
    edge_coeff = torch.cat([edge_coeff_original, self_loop_coeff], dim=0)  # Shape: [E_total, 1]

    # Create a PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_coeff=edge_coeff)

    # Instantiate the gradient layer
    gradient_layer = FinDiffGrad()

    # Compute the gradient
    grad = gradient_layer(data.x, data.edge_index, data.edge_coeff)

    print("Computed Gradient:")
    print(grad)


if __name__ == "__main__":
    main()