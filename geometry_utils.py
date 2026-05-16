import cv2
import numpy as np


def approximate_contour_corners(contour, epsilon_ratio):
    contour_perimeter = cv2.arcLength(contour, True)
    approximated_contour = cv2.approxPolyDP(contour, epsilon_ratio * contour_perimeter, True)
    return [tuple(point[0]) for point in approximated_contour]


def group_polygon_edges_by_parallel_direction(points, angle_threshold_degrees=15.0, min_line_length=1.0):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 2:
        return []

    line_vectors = np.roll(points, -1, axis=0) - points
    line_lengths = np.linalg.norm(line_vectors, axis=1)
    valid_indices = [index for index, length in enumerate(line_lengths) if length >= min_line_length]
    if not valid_indices:
        return []

    normalized_vectors = np.zeros_like(line_vectors)
    normalized_vectors[valid_indices] = line_vectors[valid_indices] / line_lengths[valid_indices, None]
    cosine_threshold = np.cos(np.deg2rad(angle_threshold_degrees))

    groups = [[valid_indices[0]]]
    group_reference_indices = [valid_indices[0]]
    for current_index in valid_indices[1:]:
        matched_group_index = None
        best_cosine_similarity = -1.0

        for group_index, reference_index in enumerate(group_reference_indices):
            cosine_similarity = abs(float(np.dot(normalized_vectors[reference_index], normalized_vectors[current_index])))
            if cosine_similarity >= cosine_threshold and cosine_similarity > best_cosine_similarity:
                matched_group_index = group_index
                best_cosine_similarity = cosine_similarity

        if matched_group_index is None:
            groups.append([current_index])
            group_reference_indices.append(current_index)
        else:
            groups[matched_group_index].append(current_index)

    return groups


def find_direction_aligned_indices(reference_direction, candidate_directions, angle_threshold_degrees=30):
    reference_direction = np.asarray(reference_direction, dtype=np.float64)
    candidate_directions = np.asarray(candidate_directions, dtype=np.float64)
    if candidate_directions.size == 0:
        return np.empty(0, dtype=int)

    reference_length = np.linalg.norm(reference_direction)
    if reference_length < 1e-6:
        return np.empty(0, dtype=int)
    reference_direction /= reference_length

    if candidate_directions.ndim == 1:
        candidate_directions = candidate_directions.reshape(1, -1)

    candidate_lengths = np.linalg.norm(candidate_directions, axis=1)
    valid_mask = candidate_lengths > 1e-6
    if not np.any(valid_mask):
        return np.empty(0, dtype=int)

    valid_indices = np.flatnonzero(valid_mask)
    normalized_candidates = candidate_directions[valid_mask] / candidate_lengths[valid_mask, None]
    cosine_threshold = np.cos(np.deg2rad(angle_threshold_degrees))
    cosine_similarities = np.abs(normalized_candidates @ reference_direction)
    return valid_indices[cosine_similarities >= cosine_threshold]


def find_lines_parallel_to_segment(from_point, to_point, lines, threshold_angle_degrees=25):
    if lines is None:
        return np.empty(0, dtype=int)

    reference_direction = np.asarray(to_point, dtype=np.float64) - np.asarray(from_point, dtype=np.float64)
    reference_length = np.linalg.norm(reference_direction)
    if reference_length < 1e-6:
        return np.empty(0, dtype=int)
    reference_direction /= reference_length

    line_points = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    line_directions = line_points[:, 2:4] - line_points[:, 0:2]
    return find_direction_aligned_indices(
        reference_direction,
        line_directions,
        angle_threshold_degrees=threshold_angle_degrees,
    )


def find_lines_near_segment(from_point, to_point, lines, threshold_distance, overlap_percentage_threshold=0.0):
    if lines is None or to_point is None:
        return np.empty(0, dtype=int)

    reference_start = np.asarray(from_point, dtype=np.float64)
    reference_end = np.asarray(to_point, dtype=np.float64)
    reference_direction = reference_end - reference_start
    reference_length = np.linalg.norm(reference_direction)
    if reference_length < 1e-6:
        return np.empty(0, dtype=int)

    line_points = np.asarray(lines, dtype=np.float64).reshape(-1, 4)
    line_starts = line_points[:, 0:2]
    line_ends = line_points[:, 2:4]
    line_midpoints = (line_starts + line_ends) * 0.5
    normal = np.array([-reference_direction[1], reference_direction[0]], dtype=np.float64) / reference_length
    perpendicular_distances = np.abs((line_midpoints - reference_start) @ normal)

    reference_axis = reference_direction / reference_length
    start_projections = (line_starts - reference_start) @ reference_axis
    end_projections = (line_ends - reference_start) @ reference_axis
    line_projection_mins = np.minimum(start_projections, end_projections)
    line_projection_maxs = np.maximum(start_projections, end_projections)
    overlap_lengths = np.maximum(0.0, np.minimum(line_projection_maxs, reference_length) - np.maximum(line_projection_mins, 0.0))
    line_projection_lengths = np.maximum(line_projection_maxs - line_projection_mins, 1e-6)
    overlap_percentages = overlap_lengths / line_projection_lengths

    close_mask = perpendicular_distances <= threshold_distance
    overlap_mask = overlap_percentages >= overlap_percentage_threshold
    return np.flatnonzero(close_mask & overlap_mask)


def points_are_collinear(inner_point, middle_point, outer_point, threshold_distance_ratio=0.2):
    inner_point = np.asarray(inner_point, dtype=np.float64)
    middle_point = np.asarray(middle_point, dtype=np.float64)
    outer_point = np.asarray(outer_point, dtype=np.float64)
    line_direction = outer_point - inner_point
    line_length = np.linalg.norm(line_direction)
    if line_length < 1e-6:
        return False

    middle_offset = middle_point - inner_point
    perpendicular_distance = abs(line_direction[0] * middle_offset[1] - line_direction[1] * middle_offset[0]) / line_length
    return perpendicular_distance <= line_length * threshold_distance_ratio


def intersect_rays(line1, line2):
    origin_a, ray_point_a = line1
    origin_b, ray_point_b = line2
    origin_a = np.asarray(origin_a, dtype=np.float64)
    origin_b = np.asarray(origin_b, dtype=np.float64)
    ray_point_a = np.asarray(ray_point_a, dtype=np.float64)
    ray_point_b = np.asarray(ray_point_b, dtype=np.float64)
    direction_a = ray_point_a - origin_a
    direction_b = ray_point_b - origin_b
    determinant = direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]

    if abs(determinant) < 1e-9:
        return None

    delta = origin_b - origin_a
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / determinant
    u = (delta[0] * direction_a[1] - delta[1] * direction_a[0]) / determinant
    if t < 0 or u < 0:
        return None

    intersection = origin_a + t * direction_a
    return float(intersection[0]), float(intersection[1])


def as_int_point(point):
    return tuple(np.round(point).astype(int))
