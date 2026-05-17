from dataclasses import dataclass
from itertools import permutations

import cv2
import numpy as np


DICE_DOT_TEMPLATES = [
    np.array([(0.5, 0.5)], dtype=np.float32),
    np.array([(0.3, 0.3), (0.7, 0.7)], dtype=np.float32),
    np.array([(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)], dtype=np.float32),
    np.array([(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)], dtype=np.float32),
    np.array([(0.3, 0.3), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.7, 0.7)], dtype=np.float32),
    np.array([(0.3, 0.25), (0.7, 0.25), (0.3, 0.5), (0.7, 0.5), (0.3, 0.75), (0.7, 0.75)], dtype=np.float32),
]

_NORMALIZED_CORNERS = np.array(
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    dtype=np.float32,
)


@dataclass
class TemplateMatch:
    face_value: int
    error: float
    homography: np.ndarray


def match_dice_dot_template(dot_positions, image_shape):
    dot_positions = _normalize_dot_positions(dot_positions, image_shape)
    if len(dot_positions) == 0:
        return None

    best_match = None
    for template_index, template_points in enumerate(DICE_DOT_TEMPLATES):
        if len(template_points) != len(dot_positions):
            continue

        match = _match_template(template_index + 1, template_points, dot_positions)
        if match is not None and (best_match is None or match.error < best_match.error):
            best_match = match

    return best_match


def draw_template_match(image, match):
    if match is None:
        return

    template_points = DICE_DOT_TEMPLATES[match.face_value - 1]
    projected_points = cv2.perspectiveTransform(template_points.reshape(-1, 1, 2), match.homography).reshape(-1, 2)
    image_height, image_width = image.shape[:2]

    for point in projected_points:
        point_x = int(round(point[0] * (image_width - 1)))
        point_y = int(round(point[1] * (image_height - 1)))
        cv2.circle(image, (point_x, point_y), 5, (255, 0, 0), 2, cv2.LINE_AA)


def draw_template_match_text(image, match, origin=(20, 94)):
    if match is None:
        return

    match_text = f"matched template: {match.face_value}"
    cv2.putText(image, match_text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(image, match_text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def _match_template(face_value, template_points, dot_positions):
    best_match = None
    template_with_corners = np.vstack((_NORMALIZED_CORNERS, template_points))

    for ordered_dot_positions in permutations(dot_positions):
        destination_points = np.vstack((_NORMALIZED_CORNERS, np.asarray(ordered_dot_positions, dtype=np.float32)))
        homography, _ = cv2.findHomography(template_with_corners, destination_points, method=0)
        if homography is None:
            continue

        projected_points = cv2.perspectiveTransform(template_points.reshape(-1, 1, 2), homography).reshape(-1, 2)
        error = float(np.mean(np.linalg.norm(projected_points - destination_points[4:], axis=1)))
        if best_match is None or error < best_match.error:
            best_match = TemplateMatch(face_value=face_value, error=error, homography=homography)

    return best_match


def _normalize_dot_positions(dot_positions, image_shape):
    dot_positions = np.asarray(dot_positions, dtype=np.float32)
    if dot_positions.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    dot_positions = dot_positions.reshape(-1, 2)
    image_height, image_width = image_shape[:2]
    scale = np.array([max(image_width - 1, 1), max(image_height - 1, 1)], dtype=np.float32)
    return dot_positions / scale
