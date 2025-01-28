import numpy as np
import sys
import pickle
from cprint import c_print
import json
import meshpy.triangle as tri
from pde.mesh_generation.mesh_gen_utils import (MeshProps, min_dist_to_boundary,
                            plot_mesh, extract_mesh_data)
from pde.mesh_generation.geometries import MeshFacet, Circle, Box, Line, Ellipse
from pde.graph_grid.graph_store import P_Types as PT
from pde.utils import dict_key_by_value

# Custom function to control mesh refinement
def refine_fn(vertices, area, props: MeshProps, points, segments):
    # Wrapper function hides exceptions raised here.
    try:
        """ True if area is too big. False if area is small enough"""
        if area < props.min_area:
            return False
        if area > props.max_area:
            return True
        centroid = np.mean(vertices, axis=0)
        dist = min_dist_to_boundary(centroid, points, segments)

        # Increase refinement near the boundaries and if the area is too large
        threshold = (props.max_area - props.min_area) * (1 - np.exp(-dist / props.lengthscale)) + props.min_area

    except Exception as e:
        c_print(f"Exception raised: {e}", color="bright_red")
        print(e)
        raise e
    return area > threshold


def create_mesh(coords: list[MeshFacet], mesh_props: MeshProps):
    # Collate together all facet objects
    points, segments = np.empty((0, 2)), np.empty((0, 2), dtype=int)
    # Segments for dist calculation
    dist_p, dist_seg = np.empty((0, 2)), np.empty((0, 2), dtype=int)

    holes = []
    seg_marks, p_marks = [], []
    marker_names = {0: "Normal"}

    for i, facets in enumerate(coords):
        cur_p = len(points)

        points = np.concatenate((points, facets.points))
        segments = np.concatenate((segments, facets.segments + cur_p))

        if facets.dist_req:
            cur_dist_p = len(dist_p)
            dist_p = np.concatenate((dist_p, facets.points))
            dist_seg = np.concatenate((dist_seg, facets.segments + cur_dist_p))

        # Default marker is 0, so start at 1
        mark_id = i + 1
        seg_marks += [mark_id] * len(facets.segments)
        p_marks += [mark_id] * len(facets.points)
        if facets.hole:
            holes.append(facets.hole)

        marker_names[mark_id] = facets.name


    # Create the mesh info object
    mesh_info = tri.MeshInfo()
    mesh_info.set_points(points, point_markers=p_marks)
    mesh_info.set_facets(segments, facet_markers=seg_marks)
    mesh_info.set_holes(holes)

    # Create the mesh
    mesh = tri.build(mesh_info, refinement_func=lambda x, y: refine_fn(x, y, mesh_props, dist_p, dist_seg))
    return mesh, marker_names


def gen_points_full():
    min_area = 7e-3
    max_area = 10e-3
    xmin, xmax = 0, 3
    ymin, ymax = 0.0, 1.5
    circle_center = (0.5, 0.4)
    circle_radius = 0.1

    lengthscale = np.sqrt(2*min_area)
    # print(lengthscale)

    mesh_props = MeshProps(min_area, max_area, lengthscale=0.4)

    coords = [#Box(Xmin, Xmax, hole=False, name="farfield", remove_edge=2),
              Line([xmin, ymin], [xmax, ymin], True, name=PT.DirichBC),
              Line([xmin, ymax], [xmax, ymax], True, name=PT.DirichBC),
              Line([xmin, ymin], [xmin, ymax], True, name=PT.NeumOffsetBC),
              Line([xmax, ymax], [xmax, ymin], True, name=PT.DirichBC),
              Circle(circle_center, circle_radius, lengthscale, True, name=PT.DirichBC),
              Circle((1.0, 0.5), circle_radius, lengthscale, True, name=PT.DirichBC),
              Circle((1.0, 0.8), circle_radius, lengthscale, True, name=PT.DirichBC),
              Ellipse((2.0, 1), 0.2, 0.75, np.pi/3, lengthscale, True, dist_req=True, name=PT.DirichBC),
        # Line([1, 0.1], [1, 0.5], name="Inlet1")
              ]
    mesh, marker_tags = create_mesh(coords, mesh_props)
    point_props, markers, _ = extract_mesh_data(mesh)
    points, _ = point_props
    p_markers, _ = markers

    p_tags = [marker_tags[int(i)] for i in p_markers]

    #plot_mesh(mesh)
    return points, p_tags


def generate_box_points_spacing(xmax, ymax, spacing=1.0):
    """
    Generates an array of 2D points outlining a box from [0, 0] to [xmax, ymax],
    with approximately the specified spacing between consecutive points.

    Parameters:
    - xmax (float): The maximum x-coordinate of the box.
    - ymax (float): The maximum y-coordinate of the box.
    - spacing (float, optional): Desired spacing between points. Default is 1.0.

    Returns:
    - numpy.ndarray: An array of shape (N, 2) containing the 2D points,
                     where N depends on the spacing and box dimensions.
    """
    if spacing <= 0:
        raise ValueError("Spacing must be a positive number.")

    def compute_num_points(length, _spacing):
        return max(int(np.ceil(length / _spacing)), 1)  # At least 1 point

    num_points_bottom = compute_num_points(xmax, spacing)
    num_points_right = compute_num_points(ymax, spacing)
    num_points_top = compute_num_points(xmax, spacing)
    num_points_left = compute_num_points(ymax, spacing)

    # Bottom edge: from (0, 0) to (xmax, 0)
    bottom = np.linspace([0, 0], [xmax, 0], num=num_points_bottom, endpoint=True)
    # Right edge: from (xmax, 0) to (xmax, ymax)
    right = np.linspace([xmax, spacing], [xmax, ymax], num=num_points_right, endpoint=False)
    # Top edge: from (xmax, ymax) to (0, ymax)
    top = np.linspace([xmax, ymax], [0, ymax], num=num_points_top, endpoint=True)
    # Left edge: from (0, ymax) to (0, 0)
    left = np.linspace([0, spacing], [0, ymax], num=num_points_left, endpoint=False)

    box_points = np.vstack((bottom, right, top, left))
    idxs = ["Wall" for _ in range(len(bottom))] + ["Right" for _ in range(len(right))] + ["Wall" for _ in range(len(top))] + ["Left" for _ in range(len(left))]
    return box_points, idxs

def gen_mesh_time(xmin, xmax, ymin, ymax, areas=None):
    if areas is None:
        min_area = 5.e-3
        max_area = 10e-3
    else:
        min_area, max_area = areas

    circle_center = (0.5, 0.4)
    circle_radius = 0.1

    lnscale = np.sqrt(1.5*min_area)
    # print(lnscale)

    mesh_props = MeshProps(min_area, max_area, lengthscale=0.5)

    coords = [
                Line([xmin, ymin], [xmax, ymin], True, name="Wall"),     # Bottom
                Line([xmin, ymax], [xmax, ymax], True, name="Wall"),     # Top
                Line([xmin, ymin], [xmin, ymax], True, name="Left"),    # Left
                Line([xmax, ymax], [xmax, ymin], True, name="Right"),   # Right
              ]
    mesh, marker_tags = create_mesh(coords, mesh_props)
    point_props, markers, _ = extract_mesh_data(mesh)
    points, _ = point_props
    p_markers, _ = markers

    # Extra layer
    points = points + 2 * lnscale
    box_points, box_tags = generate_box_points_spacing(xmax + 2*lnscale, ymax + 2*lnscale, lnscale)
    bc_points, bc_tags = generate_box_points_spacing(xmax + 4*lnscale, ymax + 4*lnscale, lnscale)
    box_points = box_points + lnscale
    p_tags = [marker_tags[0] for _ in p_markers]
    box_tags = [marker_tags[0] for _ in box_tags]
    points = np.vstack((points, box_points, bc_points))
    p_tags = p_tags + box_tags + bc_tags

    # # Set corner tags very carefully
    # corners = [[xmin, xmin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]
    # wall_tag = dict_key_by_value(marker_tags, "Wall")
    # for i, (point, tag) in enumerate(zip(points, p_markers, strict=True)):
    #     if np.any(np.all(np.isclose(corners, point), axis=1)):
    #         p_markers[i] = wall_tag        # Corresponds to wall
    # for i, (point, tag) in enumerate(zip(points, p_markers, strict=True)):
    #     if point[0] < xmin + X and tag not in [wall_tag, wall_tag + 1, left_tag]:
    #         p_markers[i] = dict_key_by_value(marker_tags, "Left_extra")

    #p_tags = [marker_tags[int(i)] for i in p_markers]

    return points, p_tags


def main():
    #exit(4)
    try:
        points, p_tags = gen_mesh_time()
        #pickle.dump((points, p_tags), sys.stdout.buffer)
    except Exception as e:
        raise e


if __name__ == "__main__":
    #from pde.graph_grid.graph_store import P_Types as PT
    main()
