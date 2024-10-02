import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, remove_self_loops

def show_graph(data: Data, value_name='us', pos_name="Xs"):
    """
    Plots a graph where the color of each node represents its value from the specified attribute.
    """
    data_copy = data.clone()
    data_copy.edge_index = remove_self_loops(data_copy.edge_index)[0]

    # Convert the PyTorch Geometric Data object to a NetworkX graph
    G = to_networkx(data_copy, node_attrs=[value_name], edge_attrs=[])

    # Get node positions from 'pos' attribute (if present)
    pos = {i: (data_copy[pos_name][i, 0].item(), data_copy[pos_name][i, 1].item()) for i in range(data_copy[pos_name].size(0))}

    # Extract the node values to color the nodes
    if value_name in data_copy:
        node_colors = data_copy[value_name].view(-1).tolist()
    else:
        raise ValueError(f"Attribute '{value_name}' not found in the graph data.")

    # Draw the graph using NetworkX and matplotlib
    plt.figure(figsize=(8, 6))
    nx.draw(
        G, pos, node_color=node_colors, with_labels=True, cmap=plt.cm.viridis,
        node_size=300, edge_color='gray', linewidths=1, font_size=12
    )
    # Create a ScalarMappable object to add the color bar
    plt.title("Graph Visualization with Node Colors Representing 'x' Values")
    plt.show()