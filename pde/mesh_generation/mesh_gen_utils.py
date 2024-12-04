from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


@dataclass
class MeshProps:
    min_area: float
    max_area: float
    lengthscale: float


def min_dist_to_boundary(point, seg_points, segment_indices):
    """
    Calculate the minimum distance from a point to a list of segments defined by indices.
    :param point: The point (x, y) as a 1D NumPy array.
    :param seg_points: A 2D NumPy array of shape (n, 2) representing all points.
    :param segment_indices: A 2D NumPy array of shape (m, 2), each row containing two indices
                            into the `points` array, representing the start and end of a segment.
    :return: The minimum distance from the point to the segments.
    """
    # Extract segment start and end points from the points array
    segment_starts = seg_points[segment_indices[:, 0]]
    segment_ends = seg_points[segment_indices[:, 1]]

    # Vector from start to end of each segment
    segment_vectors = segment_ends - segment_starts
    # Vector from start of each segment to the point
    point_vectors = point - segment_starts

    # Project point_vectors onto segment_vectors
    projection_lengths = np.einsum('ij,ij->i', point_vectors, segment_vectors) / np.einsum('ij,ij->i', segment_vectors, segment_vectors)
    projection_lengths = np.clip(projection_lengths, 0, 1)

    # Closest points on each segment to the point
    closest_points = segment_starts + (projection_lengths[:, np.newaxis] * segment_vectors)

    # Distances from the point to each closest point on the segments
    distances = np.linalg.norm(point - closest_points, axis=1)

    return np.min(distances)


def extract_interor_edges(triangles):
    """
    Extract all edges from the elements of the mesh.
    """
    # Extract all edges from the elements
    edges = np.vstack([
        triangles[:, [0, 1]],
        triangles[:, [1, 2]],
        triangles[:, [2, 0]]
    ])

    # Sort the indices within each edge to ensure consistency
    all_edges = np.sort(edges, axis=1)
    unique_all_edges, all_counts = np.unique(np.sort(all_edges, axis=1), axis=0, return_counts=True)
    interior_edges = unique_all_edges[all_counts == 2]
    return interior_edges


def extract_mesh_data(mesh):
    """ Extract the mesh data from the mesh object. """
    points, triangles, bound_edges = mesh.points, mesh.elements, mesh.facets
    p_markers, f_markers = mesh.point_markers, mesh.facet_markers

    points, triangles, bound_edges = np.array(points), np.array(triangles), np.array(bound_edges)
    p_markers, f_markers = np.array(p_markers), np.array(f_markers)

    int_edges = extract_interor_edges(triangles)
    return (points, triangles), (p_markers, f_markers), (int_edges, bound_edges)


def plot_mesh(mesh):
    # Points: List of (x, y) coordinates
    # Elements: List of 3-tuples of triangle vertex indices
    # Facets: List of 2-tuples of bounding edge vertex indices
    point_props, markers, edges = extract_mesh_data(mesh)
    points, triangles = point_props
    p_markers, f_markers = markers
    int_edges, bound_edges = edges

    # Plot the points
    cmap = cm.viridis
    norm = mcolors.Normalize(vmin=min(p_markers), vmax=max(p_markers))
    scatter = plt.scatter(
        points[:, 0],  # X coordinates
        points[:, 1],  # Y coordinates
        c=p_markers,  # Color mapping based on p_markers
        cmap=cmap,  # Colormap
        norm=norm,  # Normalization
        marker='o',  # Marker style
        s=15  # Marker size
    )
    # Plot interior edges
    # for edge in int_edges:
    #     point1 = points[edge[0]]
    #     point2 = points[edge[1]]
    #     plt.plot([point1[0], point2[0]], [point1[1], point2[1]], c="gray", linewidth=1)  # 'k-' means black line
    #
    # # Plot exterior edges
    # cmap = cm.magma
    # norm = mcolors.Normalize(vmin=min(f_markers), vmax=max(f_markers))
    # for edge, f_mark in zip(bound_edges, f_markers):
    #     point1 = points[edge[0]]
    #     point2 = points[edge[1]]
    #     c = cmap(norm(f_mark))
    #     plt.plot([point1[0], point2[0]], [point1[1], point2[1]], c=c, linewidth=2.0)

    # Plot bounding facets (edges)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('2D CFD Mesh')
    # plt.grid(True)
    plt.tight_layout()
    plt.show()
