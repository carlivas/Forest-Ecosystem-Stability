import numpy as np
from scipy.spatial import KDTree

def boundary_check(boundary, positions, radii=None):
    # Returns a boolean array indicating whether the positions are beyond the boundary
    # if radii is given returns also circles with centers inside the boundary but with radii that reach outside
    positions = np.atleast_2d(positions)
    radii = np.zeros(len(positions)) if radii is None else np.asarray(radii)
    is_beyond_boundary_radii = np.column_stack([
        positions[:, 0] - radii < boundary[0, 0],  # Left boundary
        positions[:, 0] + radii > boundary[0, 1],  # Right boundary
        positions[:, 1] - radii < boundary[1, 0],  # Bottom boundary
        positions[:, 1] + radii > boundary[1, 1]   # Top boundary
    ])
    is_beyond_boundary_radii = np.atleast_2d(is_beyond_boundary_radii)
    return is_beyond_boundary_radii


def positions_shift_periodic_all(boundary, positions):
    w = boundary[0, 1] - boundary[0, 0]
    h = boundary[1, 1] - boundary[1, 0]
    
    positions_shifted = np.vstack([
        positions,
        positions + np.array([0, h]), # shift up
        positions + np.array([w, h]), # shift up and right
        positions + np.array([w, 0]), # shift right
        positions + np.array([w, -h]), # shift right and down
        positions + np.array([0, -h]), # shift down
        positions + np.array([-w, -h]), # shift down and left
        positions + np.array([-w, 0]), # shift left
        positions + np.array([-w, h]), # shift left and up
    ])
    
    index_pairs = np.vstack([np.tile(np.arange(len(positions)), 9), np.arange(positions_shifted.shape[0])]).T    
    was_shifted = np.zeros(len(positions_shifted), dtype=bool)
    was_shifted[len(positions_shifted)//9:] = True
    return positions_shifted, index_pairs, was_shifted
    
def positions_shift_periodic(boundary, positions, radii=None, duplicates=False):
    # Shifts the positions to opposite side(s) of the boundary if they are beyond it, if radii are given, these are included in checking for boundary crossing
    # Returns the shifted positions and an array of pairs of indices of the original positions and it's corresponding shifted position
        
    
    positions_shifted = np.empty((0, 2), dtype=float)
    index_pairs = np.empty((0, 2), dtype=int)
    was_shifted = np.empty(0, dtype=bool)
    
    if positions.size == 0:
        raise ValueError('No positions were given')
    
    positions = np.atleast_2d(positions)
    is_beyond_boundary_radii = boundary_check(boundary, positions, radii)
    is_beyond_boundary = boundary_check(boundary, positions)

    shifts = -2 * np.array([[boundary[0, 0], 0],
                            [boundary[0, 1], 0],
                            [0, boundary[1, 0]],
                            [0, boundary[1, 1]]])
    for i, p in enumerate(positions):
    
        # If the original position without radius is beyond the boundary
        if is_beyond_boundary[i].any():
            # dont include an entry for the original position
            shift = np.empty((0, 2), dtype=float)
            shifted = np.empty(0, dtype=bool)
        else:
            # include an entry for the original position
            shift = np.array([[0, 0]], dtype=float)
            shifted = np.array([False], dtype=bool)
        
        # If the position is beyond the boundary
        if np.any(is_beyond_boundary_radii[i]):
            
            # Get the shift for each boundary crossed
            shift_arr = shifts[is_beyond_boundary_radii[i]]
            
            # add the sum of the shifts to the original position
            shift = np.vstack([shift, np.sum(shift_arr, axis=0)])
            shifted = np.hstack([shifted, True])
                        
            # If duplicates are turned on also add the shifts corresponding to EACH boundary crossed
            if duplicates and len(shift_arr) > 1:
                shift = np.vstack([shift, shift_arr])
                shifted = np.hstack([shifted, np.ones(len(shift_arr), dtype=bool)])
        
        # Create the index pairs
        indices = np.array([[i, len(positions_shifted) + j] for j in range(shift.shape[0])])
        
        positions_shifted = np.vstack(
            [positions_shifted, p + shift]) if positions_shifted.size else p + shift
        index_pairs = np.vstack(
            [index_pairs, indices]) if index_pairs.size else indices
        was_shifted = np.hstack(
            [was_shifted, shifted]) if was_shifted.size else shifted

    index_pairs = np.atleast_2d(index_pairs)
    return positions_shifted, index_pairs, was_shifted

def get_all_collisions(positions, radii):
    # If there are no positions, there are no collisions
    if len(positions) == 0:
        return np.array([])
    kdtree = KDTree(positions)
    # Query pairs of points within the maximum collision distance
    max_collision_distance = 2 * np.max(radii)
    pairs = kdtree.query_pairs(max_collision_distance)
    # Filter pairs of indices of circles that are within collision distance
    collision_dist_matrix = radii[:, None] + radii[None, :]
    collisions_indices = np.array([(i, j) for i, j in pairs if np.sum(
        (kdtree.data[i] - kdtree.data[j])**2) < collision_dist_matrix[i, j]**2])
    return collisions_indices

def get_collisions_for_point(positions, radii, point, radius=0):
    # If there are no positions, there are no collisions
    if len(positions) == 0:
        return np.array([])
    kdtree = KDTree(positions)
    # Query the KDTree for points within the collision distance
    collision_distance = radius + radii
    indices = kdtree.query_ball_point(point, r=np.max(collision_distance))
    # Filter indices to include only those within the actual collision distance
    collisions_indices = np.array([i for i in indices if np.sum(
        (kdtree.data[i] - point)**2) < collision_distance[i]**2])
    return collisions_indices