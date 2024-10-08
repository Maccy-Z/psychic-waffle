import networkx as nx
import matplotlib.pyplot as plt
import torch
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse import csr_matrix

def show_graph(edge_idx, Xs, us):
    """
    Plots a graph where the color of each node represents its value from the specified attribute.
    """
    edge_idx = edge_idx.cpu().clone().numpy().T
    Xs = Xs.cpu().clone().numpy()
    us = us.cpu().clone().numpy()

    # Remove self loops
    edge_idx = edge_idx[edge_idx[:, 0] != edge_idx[:, 1]]

    # Create a graph
    G = nx.Graph()
    G.add_edges_from(edge_idx)

    # Convert positions array to a dictionary for networkx
    pos_dict = {i: Xs[i] for i in range(Xs.shape[0])}

    # Draw the graph
    plt.figure(figsize=(8, 6))

    # Draw nodes with color reflecting their values
    nx.draw_networkx_nodes(
        G,
        pos=pos_dict,
        node_size=300,
        nodelist=sorted(G.nodes()),
        node_color=[us[node] for node in sorted(G.nodes())],
        cmap=plt.cm.viridis,
    )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos=pos_dict,
        arrowstyle='-|>',  # Arrow style
        arrowsize=10,  # Arrow size
        edge_color='grey',
        width=1,
        connectionstyle='arc3,rad=0.1',  # Adds curvature to the edges
        arrows=True  # Display arrows on the edges
    )

    # Create labels for nodes with their values
    labels = {i: f'{i}' for i in range(len(us))}
    nx.draw_networkx_labels(G, pos=pos_dict, labels=labels)

    # Display the graph
    #plt.title("Graph Visualization with NumPy Inputs")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def gen_perim(width, height, spacing):
    """
    Generates a tensor of 2D points representing the perimeter of a rectangle.
    Parameters:
    - width: Width of the rectangle.
    - height: Height of the rectangle.
    - spacing: Spacing between consecutive points.
    Returns:
    - A tensor of shape (N, 2), where N is the number of points, and each row is a 2D point (x, y).
    """
    # Create points for the bottom and top sides of the rectangle
    bottom = torch.stack([torch.arange(0, width, spacing), torch.zeros_like(torch.arange(0, width, spacing))], dim=-1)
    top = torch.stack([torch.arange(0, width, spacing), torch.full_like(torch.arange(0, width, spacing), height)], dim=-1)

    print(bottom)
    # Create points for the left and right sides of the rectangle
    left = torch.stack([torch.zeros_like(torch.arange(0, height, spacing)), torch.arange(0, height, spacing)], dim=-1)
    right = torch.stack([torch.full_like(torch.arange(0, height+1e-7, spacing), width), torch.arange(0, height+1e-7, spacing)], dim=-1)

    # Remove duplicated corner points
    left = left[1:] if len(left) > 1 else left  # Remove the bottom-left corner
    #right = right[1:] if len(right) > 1 else right  # Remove the top-right corner

    # Concatenate all points
    points = torch.cat([bottom, right, top.flip(0), left.flip(0)], dim=0)

    return points


def diag_permute(A_sparse: torch.Tensor):
    """ Reorder the rows and columns of a matrix using the reverse Cuthill-McKee algorithm.
        Args:
        - A (torch.Tensor): Input matrix.
        Returns:
        - A_permuted (torch.Tensor): Permuted matrix.
    """
    A_sparse = A_sparse.detach().cpu()
    A_sparse = A_sparse.coalesce()

    A = csr_matrix((A_sparse.values().numpy(), A_sparse.indices().numpy()), shape=A_sparse.size())

    perm = reverse_cuthill_mckee(A)

    return perm

