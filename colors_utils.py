import cv2
import numpy as np


def contour_center(contour):
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        return (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )

    x, y, w, h = cv2.boundingRect(contour)
    return np.array((x + w // 2, y + h // 2), dtype=int)


def draw_cross(frame, center, color, size=6, thickness=2):
    x, y = center
    x = int(x)
    y = int(y)
    cv2.line(frame, (x - size, y), (x + size, y), color, thickness)
    cv2.line(frame, (x, y - size), (x, y + size), color, thickness)


def point_is_below_line(line_start, line_end, point):
    cross = ((line_end[0] - line_start[0]) * (point[1] - line_start[1])
             - (line_end[1] - line_start[1]) * (point[0] - line_start[0]))
    return (cross > 0)


def line_parameters(lineA, lineB, positions):
    lineA = np.asarray(lineA, dtype=np.float32)
    lineB = np.asarray(lineB, dtype=np.float32)
    line_dir = lineB - lineA
    line_len_sq = np.dot(line_dir, line_dir)

    if line_len_sq == 0:
        raise ValueError("lineA and lineB must be different positions")

    return [float(np.dot(np.asarray(position, dtype=np.float32) - lineA, line_dir) / line_len_sq)
            for position in positions]


def circle_positions(radius, num_positions):
    angles = np.linspace(0, 2 * np.pi, num_positions, endpoint=False) + np.pi / 2
    offsets = np.stack((np.cos(angles), np.sin(angles)), axis=1) * radius
    return offsets

def _position_array(marker_or_position):
    if hasattr(marker_or_position, "position"):
        marker_or_position = marker_or_position.position
    return np.asarray(marker_or_position, dtype=np.float32)


def small_dotproducts(line_markers, other_markers):
    result = []

    for i in range(len(line_markers)):
        next_line_marker = line_markers[i+1] if i < len(line_markers) - 1 else line_markers[i-1]
        start = _position_array(line_markers[i])
        end = _position_array(next_line_marker)
        direction = end - start

        smallest_marker = None
        smallest_dot = None

        for marker in other_markers:
            directionB = _position_array(marker) - start
            # dot = float(np.dot(direction, directionB))
            dot = np.dot(direction / np.linalg.norm(direction), directionB / np.linalg.norm(directionB))
            abs_dot = abs(dot)
            if abs_dot < 0.3:
                if smallest_dot is None or abs_dot < smallest_dot:
                    smallest_dot = abs_dot
                    smallest_marker = marker

        result.append((line_markers[i], smallest_marker, smallest_dot))

    return result


def rotation_angle_between(rvec_a, rvec_b):
    R_a, _ = cv2.Rodrigues(rvec_a)
    R_b, _ = cv2.Rodrigues(rvec_b)
    R_rel = R_a @ R_b.T
    rvec_rel, _ = cv2.Rodrigues(R_rel)
    return float(np.linalg.norm(rvec_rel))


def pose_distance(rvec, tvec, prev_rvec, prev_tvec, translation_weight=1.0):
    rot_error = rotation_angle_between(rvec, prev_rvec)
    trans_error = float(np.linalg.norm(tvec.reshape(3) - prev_tvec.reshape(3)))
    return rot_error + translation_weight * trans_error


def choose_best_pose(rvecs, tvecs, prev_rvec, prev_tvec):
    if prev_rvec is None or prev_tvec is None:
        return rvecs[0], tvecs[0]

    best_index = 0
    best_score = float("inf")

    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        score = pose_distance(rvec, tvec, prev_rvec, prev_tvec, translation_weight=0.05)
        if score < best_score:
            best_score = score
            best_index = i

    return rvecs[best_index], tvecs[best_index]