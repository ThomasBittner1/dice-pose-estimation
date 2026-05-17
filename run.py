import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import drawing
import geometry_utils
from hough_utils import detect_hough_lines_in_contour_roi
from tracking_state import StabilityTracker


QUIT_KEYS = (ord("q"), ord("Q"))
DEBUG_KEYS = (ord("d"), ord("D"))
SPACE_KEY = ord(" ")
LEFT_ARROW_KEY = 2424832
RIGHT_ARROW_KEY = 2555904
COLOR_RANGES_PATH = Path(__file__).with_name("color_ranges.json")
DICE_BODY_CONTOUR_COLOR_NAME = "dice_body_contour"
DICE_FACE_COLOR_NAME = "dice_face_color"
DEFAULT_DICE_HSV_MIN = (26, 59, 30)
DEFAULT_DICE_HSV_MAX = (98, 255, 250)
DEFAULT_TOP_FACE_GREEN_HSV_MIN = (62, 37, 92)
DEFAULT_TOP_FACE_GREEN_HSV_MAX = (89, 255, 249)


def load_hsv_range(color_name, value_name, fallback):
    if not COLOR_RANGES_PATH.exists():
        return fallback

    try:
        with COLOR_RANGES_PATH.open("r", encoding="utf-8") as config_file:
            color_config = json.load(config_file)
    except (OSError, json.JSONDecodeError, TypeError):
        return fallback

    values = color_config.get(color_name, {}).get(value_name)
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return fallback

    try:
        return tuple(int(value) for value in values)
    except (TypeError, ValueError):
        return fallback


@dataclass
class AppConfig:
    video_source: str | int = "green_dice_1.mp4"
    record_video_path: str | None = None
    start_frame: int = 0 #550
    start_paused: bool = True
    flip_frame_horizontal: bool = False
    stable_similarity_threshold: float = 0.85
    dice_hsv_min: tuple[int, int, int] = field(
        default_factory=lambda: load_hsv_range(DICE_BODY_CONTOUR_COLOR_NAME, "min_vals", DEFAULT_DICE_HSV_MIN)
    )
    dice_hsv_max: tuple[int, int, int] = field(
        default_factory=lambda: load_hsv_range(DICE_BODY_CONTOUR_COLOR_NAME, "max_vals", DEFAULT_DICE_HSV_MAX)
    )
    top_face_green_hsv_min: tuple[int, int, int] = field(
        default_factory=lambda: load_hsv_range(DICE_FACE_COLOR_NAME, "min_vals", DEFAULT_TOP_FACE_GREEN_HSV_MIN)
    )
    top_face_green_hsv_max: tuple[int, int, int] = field(
        default_factory=lambda: load_hsv_range(DICE_FACE_COLOR_NAME, "max_vals", DEFAULT_TOP_FACE_GREEN_HSV_MAX)
    )
    count_sphere_required_count_frames: int = 5
    contour_epsilon_ratio: float = 0.02
    parallel_min_line_length: float = 0.1
    parallel_angle_threshold_degrees: float = 30.0
    hough_canny_threshold1: int = 0
    hough_canny_threshold2: int = 40
    hough_rho: float = 1.0
    hough_theta_degrees: float = 1.0
    hough_threshold: int = 20
    hough_min_line_length_min_pixels: int = 10
    hough_min_line_length_width_divisor: int = 5
    hough_max_line_gap_min_pixels: int = 4
    hough_max_line_gap_width_divisor: int = 20
    hough_vertical_keep_min_y_ratio: float = 0.15
    hough_vertical_keep_max_y_ratio: float = 0.85
    repair_hough_margin_percent: float = 0.0
    repair_hough_direction_angle_threshold_degrees: float = 30.0
    top_face_hough_margin_percent: float = 0.2
    top_face_hough_parallel_angle_threshold_degrees: float = 15.0
    top_face_hough_distance_threshold_width_ratio: float = 0.1
    top_face_hough_overlap_percentage_threshold: float = 0.4
    pip_blur_kernel_size: tuple[int, int] = (9, 9)
    pip_blur_sigma: float = 2.0
    pip_hough_dp: float = 1.2
    pip_hough_min_dist_min_pixels: int = 10
    pip_hough_min_dist_size_divisor: int = 5
    pip_hough_param1: int = 120
    pip_hough_param2: int = 20
    pip_min_radius_min_pixels: int = 3
    pip_min_radius_size_divisor: int = 16
    pip_max_radius_size_divisor: int = 6


@dataclass
class PipelineResult:
    preview: np.ndarray
    similarity_score: float = 0.0
    count_sphere_count: int | None = None
    count_sphere_position: tuple[int, int] | None = None
    top_face_warp: np.ndarray | None = None
    blurred_mask_preview: np.ndarray | None = None
    contour_points: np.ndarray | None = None


def close_debug_windows():
    for window_name in ("cropped_extracted_by_mask", "Dice Edges", "Dice Top Face", "Dice Blurred Mask"):
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def open_video_writer(record_video_path: str, frame_shape: tuple[int, ...], fps: float) -> cv2.VideoWriter | None:
    frame_height, frame_width = frame_shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(record_video_path, fourcc, fps, (frame_width, frame_height))
    if not video_writer.isOpened():
        print(f"Warning: Could not open video writer for {record_video_path}. Recording disabled.")
        return None

    return video_writer


def run_pipeline(
    frame: np.ndarray,
    config: AppConfig,
    previous_mask: np.ndarray | None,
    frame_number: int,
    debug_mode: bool,
) -> tuple[PipelineResult, np.ndarray | None]:
    preview = frame.copy()
    draw_frame_number(preview, frame_number, debug_mode)

    mask, outer_contour, outer_contour_rect = segment_dice(preview, config)
    similarity_score = geometry_utils.get_similarity_score_mask(mask, previous_mask)
    next_previous_mask = mask.copy()
    if outer_contour is None or outer_contour_rect is None:
        return PipelineResult(preview=preview, similarity_score=similarity_score), next_previous_mask

    if debug_mode:
        cv2.drawContours(preview, [outer_contour], -1, (255, 255, 255), 1)

    contour_result = extract_contour_geometry(
        frame,
        mask,
        outer_contour,
        outer_contour_rect,
        preview,
        frame_number,
        config,
        debug_mode,
    )
    if contour_result is None:
        return PipelineResult(preview=preview, similarity_score=similarity_score), next_previous_mask

    contour_points = contour_result
    top_face_result = estimate_top_face(frame, mask, outer_contour_rect, contour_points, preview, config, debug_mode)
    result = PipelineResult(
        preview=preview,
        similarity_score=similarity_score,
        contour_points=contour_points,
    )

    if top_face_result is not None:
        top_face_points, top_face_warp, count_sphere_position = top_face_result
        pip_count, blurred_mask_preview = detect_pips(top_face_warp, config, debug_mode)
        result.count_sphere_count = pip_count
        result.count_sphere_position = count_sphere_position
        result.top_face_warp = top_face_warp
        result.blurred_mask_preview = blurred_mask_preview

    return result, next_previous_mask


def segment_dice(frame: np.ndarray, config: AppConfig) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int, int, int] | None]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array(config.dice_hsv_min, dtype=np.uint8),
        np.array(config.dice_hsv_max, dtype=np.uint8),
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask, None, None

    outer_contour = max(contours, key=cv2.contourArea)
    return mask, outer_contour, cv2.boundingRect(outer_contour)


def extract_contour_geometry(
    frame: np.ndarray,
    mask: np.ndarray,
    outer_contour: np.ndarray,
    outer_contour_rect: tuple[int, int, int, int],
    preview: np.ndarray,
    frame_number: int,
    config: AppConfig,
    debug_mode: bool,
) -> np.ndarray | None:
    points = geometry_utils.approximate_contour_corners(outer_contour, config.contour_epsilon_ratio)
    points = np.squeeze(points)
    if points.ndim != 2 or len(points) < 3:
        return None

    draw_polygon_debug(preview, points, (0, 255, 0), line_thickness=3, text_thickness=2, enabled=debug_mode)

    points = repair_contour_points(frame, mask, outer_contour_rect, points, frame_number, config, debug_mode)

    draw_polygon_debug(preview, points, (0, 0, 255), line_thickness=2, text_thickness=1, enabled=debug_mode)
    return points


def repair_contour_points(
    frame: np.ndarray,
    mask: np.ndarray,
    outer_contour_rect: tuple[int, int, int, int],
    points: np.ndarray,
    frame_number: int,
    config: AppConfig,
    debug_mode: bool,
) -> np.ndarray:
    parallels = geometry_utils.group_polygon_edges_by_parallel_direction(
        points,
        min_line_length=config.parallel_min_line_length,
        angle_threshold_degrees=config.parallel_angle_threshold_degrees,
    )
    edge_lengths = get_polygon_edge_lengths(points)

    if len(points) == 4:
        return repair_four_point_contour(frame, mask, outer_contour_rect, points, parallels, edge_lengths, config, debug_mode)
    if len(points) == 5:
        return repair_five_point_contour(points, parallels, edge_lengths, frame_number, debug_mode)

    return points


def repair_four_point_contour(
    frame: np.ndarray,
    mask: np.ndarray,
    outer_contour_rect: tuple[int, int, int, int],
    points: np.ndarray,
    parallels: list[list[int]],
    edge_lengths: list[float],
    config: AppConfig,
    debug_mode: bool,
) -> np.ndarray:
    if not (len(parallels) == 2 and len(parallels[0]) == 2 and len(parallels[1]) == 2):
        return points

    _, _, _, outer_contour_h = outer_contour_rect
    hough_lines, hough_lines_lengths, hough_directions = detect_hough_lines_in_contour_roi(
        frame,
        mask,
        outer_contour_rect,
        canny_threshold1=config.hough_canny_threshold1,
        canny_threshold2=config.hough_canny_threshold2,
        hough_rho=config.hough_rho,
        hough_theta_degrees=config.hough_theta_degrees,
        hough_threshold=config.hough_threshold,
        min_line_length_min_pixels=config.hough_min_line_length_min_pixels,
        min_line_length_width_divisor=config.hough_min_line_length_width_divisor,
        max_line_gap_min_pixels=config.hough_max_line_gap_min_pixels,
        max_line_gap_width_divisor=config.hough_max_line_gap_width_divisor,
        vertical_keep_min_y_ratio=config.hough_vertical_keep_min_y_ratio,
        vertical_keep_max_y_ratio=config.hough_vertical_keep_max_y_ratio,
        margin_perc=config.repair_hough_margin_percent,
        do_imshow=debug_mode,
    )

    ratio = 0.5
    if hough_lines is not None:
        highest_point_indices = np.argsort(points[:, 1], axis=0)[:2]
        highest_line_direction = (points[highest_point_indices[1]] - points[highest_point_indices[0]]).astype("float64")
        highest_line_direction /= np.linalg.norm(highest_line_direction)

        similar_direction_indices = geometry_utils.find_direction_aligned_indices(
            highest_line_direction,
            hough_directions,
            angle_threshold_degrees=config.repair_hough_direction_angle_threshold_degrees,
        )
        if len(similar_direction_indices):
            longest_similar_hough_index = np.argmax(hough_lines_lengths[similar_direction_indices])
            longest_similar_line = hough_lines[similar_direction_indices][longest_similar_hough_index][0]
            ratio = ((longest_similar_line[1] + longest_similar_line[3]) * 0.5) / outer_contour_h

    longer_index = np.argmax([edge_lengths[x[0]] + edge_lengths[x[1]] for x in parallels])
    return insert_points_on_edge_pair(points, parallels[longer_index], ratio)


def insert_points_on_edge_pair(points: np.ndarray, edge_pair: list[int], ratio: float) -> np.ndarray:
    inserts = {}
    for edge_pair_index, from_point_index in enumerate(edge_pair):
        edge_ratio = 1 - ratio if edge_pair_index == 1 else ratio
        new_point = (
            points[from_point_index] * (1 - edge_ratio)
            + points[(from_point_index + 1) % len(points)] * edge_ratio
        ).astype(int)
        inserts[(from_point_index + 1) % len(points)] = new_point

    points = points.tolist()
    for at_index in sorted(inserts.keys(), reverse=True):
        points.insert(at_index, inserts[at_index])

    return np.array(points, dtype=int)


def repair_five_point_contour(
    points: np.ndarray,
    parallels: list[list[int]],
    edge_lengths: list[float],
    frame_number: int,
    debug_mode: bool,
) -> np.ndarray:
    if len([x for x in parallels if len(x) == 1]) == 2:
        if debug_mode:
            print("invalid", frame_number + 1)
        return points

    parallels = normalize_five_point_parallel_groups(parallels)
    if len(parallels) != 3:
        return points

    single_index = np.argmin([len(x) for x in parallels])
    double_indices = list(set([0, 1, 2]) - {single_index})
    longer_pair_index2 = np.argmax(
        [max(edge_lengths[parallels[x][0]], edge_lengths[parallels[x][1]]) for x in double_indices]
    )
    longer_pair_index = double_indices[longer_pair_index2]
    longer_pair = parallels[longer_pair_index]

    single_edge = parallels[single_index][0]
    cut = 0 if abs(single_edge - longer_pair[0]) != 1 else 1
    keep = 1 - cut
    ratio = edge_lengths[single_edge] / (edge_lengths[single_edge] + edge_lengths[longer_pair[keep]])
    from_point_index = longer_pair[cut]
    new_point = (
        points[from_point_index] * ratio
        + points[(from_point_index + 1) % len(points)] * (1 - ratio)
    ).astype(int)

    points = points.tolist()
    points.insert((from_point_index + 1) % len(points), new_point)
    return np.array(points, dtype=int)


def normalize_five_point_parallel_groups(parallels: list[list[int]]) -> list[list[int]]:
    parallels = [group.copy() for group in parallels]
    if len(parallels) != 2:
        return parallels

    bigger_index = np.argmax([len(x) for x in parallels])
    triple = parallels[bigger_index]
    separate = None
    for edge_index in range(len(triple)):
        if (
            abs(triple[edge_index] - triple[(edge_index - 1) % 3]) != 1
            and abs(triple[edge_index] - triple[(edge_index + 1) % 3]) != 1
        ):
            separate = triple[edge_index]
            break

    if separate is None:
        return parallels

    others = set(triple) - {separate}
    parallels.pop(bigger_index)
    parallels.append([others.pop()])
    parallels.append([separate, others.pop()])
    return parallels


def estimate_top_face(
    frame: np.ndarray,
    mask: np.ndarray,
    outer_contour_rect: tuple[int, int, int, int],
    points: np.ndarray,
    preview: np.ndarray,
    config: AppConfig,
    debug_mode: bool,
) -> tuple[list[tuple[int, int]], np.ndarray, tuple[int, int]] | None:
    if len(points) != 6:
        return None

    ordered_points, cross_point = order_points_for_top_face(frame, mask, outer_contour_rect, points, config, debug_mode)
    if cross_point is None:
        return None

    draw_polygon_debug(preview, ordered_points, (255, 0, 0), line_thickness=2, text_thickness=2, enabled=debug_mode)
    if debug_mode:
        cv2.circle(preview, geometry_utils.as_int_point(cross_point), 5, (0, 0, 255), -1)

    top_face_points = [
        geometry_utils.as_int_point(ordered_points[0]),
        geometry_utils.as_int_point(ordered_points[1]),
        geometry_utils.as_int_point(cross_point),
        geometry_utils.as_int_point(ordered_points[5]),
    ]
    draw_top_face_debug(preview, top_face_points, debug_mode)

    top_face_warp = warp_top_face(frame, top_face_points)
    return top_face_points, top_face_warp, get_top_face_label_position(top_face_points)


def order_points_for_top_face(
    frame: np.ndarray,
    mask: np.ndarray,
    outer_contour_rect: tuple[int, int, int, int],
    points: np.ndarray,
    config: AppConfig,
    debug_mode: bool,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    highest_two = np.argsort(points[:, 1])[:2]
    cross_points = [None] * len(highest_two)
    hough_lines, hough_lines_lengths, _ = detect_hough_lines_in_contour_roi(
        frame,
        mask,
        outer_contour_rect,
        canny_threshold1=config.hough_canny_threshold1,
        canny_threshold2=config.hough_canny_threshold2,
        hough_rho=config.hough_rho,
        hough_theta_degrees=config.hough_theta_degrees,
        hough_threshold=config.hough_threshold,
        min_line_length_min_pixels=config.hough_min_line_length_min_pixels,
        min_line_length_width_divisor=config.hough_min_line_length_width_divisor,
        max_line_gap_min_pixels=config.hough_max_line_gap_min_pixels,
        max_line_gap_width_divisor=config.hough_max_line_gap_width_divisor,
        vertical_keep_min_y_ratio=config.hough_vertical_keep_min_y_ratio,
        vertical_keep_max_y_ratio=config.hough_vertical_keep_max_y_ratio,
        margin_perc=config.top_face_hough_margin_percent,
        do_imshow=debug_mode,
    )

    outer_contour_x, outer_contour_y, outer_contour_w, _ = outer_contour_rect
    crop_offset = np.array([outer_contour_x, outer_contour_y], dtype=np.float64)
    cropped_points = points.astype(np.float64) - crop_offset
    distance_sums = np.zeros(2, dtype=np.float64)
    final_highest_point = None
    final_cross_point = None
    best_index = 0

    for index, highest_point in enumerate(highest_two):
        point_indices = (np.arange(6) + highest_point) % len(points)
        cross_point = estimate_cross_point(points, point_indices)
        if cross_point is None:
            continue

        if top_face_triangle_is_collinear(cropped_points, point_indices):
            final_highest_point = highest_point
            final_cross_point = cross_point
            if debug_mode:
                print("triangle is line")

        cross_points[index] = cross_point

        if hough_lines is None:
            final_highest_point = highest_point
            final_cross_point = cross_point
            break

        cropped_cross_point = np.asarray(cross_point, dtype=np.float64) - crop_offset
        distance_sums[index] += sum_matching_hough_lengths(
            cropped_points,
            point_indices,
            cropped_cross_point,
            hough_lines,
            hough_lines_lengths,
            outer_contour_w,
            config,
        )

    if final_highest_point is None:
        best_index = int(np.argmax(distance_sums))
        final_highest_point = highest_two[best_index]
    if final_cross_point is None:
        final_cross_point = cross_points[best_index]

    return np.roll(points, -final_highest_point, axis=0), final_cross_point


def estimate_cross_point(points: np.ndarray, point_indices: np.ndarray) -> tuple[float, float] | None:
    return geometry_utils.intersect_rays(
        (
            points[point_indices[5]],
            points[point_indices[5]] + points[point_indices[1]] - points[point_indices[0]],
        ),
        (
            points[point_indices[1]],
            points[point_indices[1]] + points[point_indices[5]] - points[point_indices[0]],
        ),
    )


def top_face_triangle_is_collinear(cropped_points: np.ndarray, point_indices: np.ndarray) -> bool:
    return (
        geometry_utils.points_are_collinear(
            cropped_points[point_indices[0]],
            cropped_points[point_indices[1]],
            cropped_points[point_indices[2]],
            threshold_distance_ratio=0.2,
        )
        and geometry_utils.points_are_collinear(
            cropped_points[point_indices[2]],
            cropped_points[point_indices[3]],
            cropped_points[point_indices[4]],
            threshold_distance_ratio=0.2,
        )
    )


def sum_matching_hough_lengths(
    cropped_points: np.ndarray,
    point_indices: np.ndarray,
    cropped_cross_point: np.ndarray,
    hough_lines: np.ndarray,
    hough_lines_lengths: np.ndarray,
    outer_contour_w: int,
    config: AppConfig,
) -> float:
    total = 0.0
    for segment_start_index in (1, 5, 3):
        start_point = cropped_points[point_indices[segment_start_index]]
        parallel_indices = geometry_utils.find_lines_parallel_to_segment(
            start_point,
            cropped_cross_point,
            hough_lines,
            threshold_angle_degrees=config.top_face_hough_parallel_angle_threshold_degrees,
        )
        close_indices2 = geometry_utils.find_lines_near_segment(
            start_point,
            cropped_cross_point,
            hough_lines[parallel_indices],
            threshold_distance=outer_contour_w * config.top_face_hough_distance_threshold_width_ratio,
            overlap_percentage_threshold=config.top_face_hough_overlap_percentage_threshold,
        )
        close_indices = parallel_indices[close_indices2]
        total += float(np.sum(hough_lines_lengths[close_indices]))

    return total


def warp_top_face(frame: np.ndarray, top_face_points: list[tuple[int, int]]) -> np.ndarray:
    source_points = np.array(top_face_points, dtype=np.float32)
    top_width = max(
        1,
        int(round(max(np.linalg.norm(source_points[1] - source_points[0]), np.linalg.norm(source_points[2] - source_points[3])))),
    )
    top_height = max(
        1,
        int(round(max(np.linalg.norm(source_points[3] - source_points[0]), np.linalg.norm(source_points[2] - source_points[1])))),
    )
    top_size = max(top_width, top_height)
    destination_points = np.array(
        [[0, 0], [top_size - 1, 0], [top_size - 1, top_size - 1], [0, top_size - 1]],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(source_points, destination_points)
    return cv2.warpPerspective(frame, homography, (top_size, top_size))


def detect_pips(top_face_warp: np.ndarray, config: AppConfig, debug_mode: bool) -> tuple[int, np.ndarray]:
    hsv = cv2.cvtColor(top_face_warp, cv2.COLOR_BGR2HSV)
    green_range = cv2.inRange(
        hsv,
        np.array(config.top_face_green_hsv_min, dtype=np.uint8),
        np.array(config.top_face_green_hsv_max, dtype=np.uint8),
    )
    mask = cv2.bitwise_not(green_range)
    blurred_mask = cv2.GaussianBlur(mask, config.pip_blur_kernel_size, config.pip_blur_sigma)
    blurred_mask_preview = cv2.cvtColor(blurred_mask, cv2.COLOR_GRAY2BGR)
    top_size = top_face_warp.shape[0]
    min_radius = max(config.pip_min_radius_min_pixels, top_size // config.pip_min_radius_size_divisor)
    max_radius = max(min_radius + 1, top_size // config.pip_max_radius_size_divisor)
    circles = cv2.HoughCircles(
        blurred_mask,
        cv2.HOUGH_GRADIENT,
        dp=config.pip_hough_dp,
        minDist=max(config.pip_hough_min_dist_min_pixels, top_size // config.pip_hough_min_dist_size_divisor),
        param1=config.pip_hough_param1,
        param2=config.pip_hough_param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return 0, blurred_mask_preview

    rounded_circles = np.round(circles[0]).astype(int)
    if debug_mode:
        for circle_x, circle_y, circle_radius in rounded_circles:
            cv2.circle(blurred_mask_preview, (circle_x, circle_y), circle_radius, (0, 255, 255), 2)
            cv2.circle(blurred_mask_preview, (circle_x, circle_y), 2, (255, 0, 255), -1)
    num_dots = min(len(rounded_circles), 6) if len(rounded_circles) else None
    return num_dots, blurred_mask_preview


def draw_pipeline_result(
    result: PipelineResult,
    stability_tracker: StabilityTracker,
    count_sphere_renderer: drawing.CountSphereRenderer,
    debug_mode: bool,
):
    tracking_state = stability_tracker.update(
        0 if result.count_sphere_count is None else result.similarity_score
    )
    draw_tracking_debug(result.preview, tracking_state, result.similarity_score, debug_mode)

    count_sphere_renderer.update_and_draw(
        result.preview,
        result.count_sphere_count,
        result.count_sphere_position,
        stability_tracker.is_stable,
    )


def show_windows(result: PipelineResult, debug_mode: bool):
    cv2.imshow("Dice Final", result.preview)
    if debug_mode and result.top_face_warp is not None:
        cv2.imshow("Dice Top Face", result.top_face_warp)
        if result.blurred_mask_preview is not None:
            cv2.imshow("Dice Blurred Mask", result.blurred_mask_preview)


def draw_frame_number(preview: np.ndarray, frame_number: int, debug_mode: bool):
    if not debug_mode:
        return

    frame_text = f"frame: {frame_number}"
    text_size, _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    text_x = preview.shape[1] - text_size[0] - 20
    cv2.putText(preview, frame_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_tracking_debug(preview: np.ndarray, tracking_state, similarity_score: float, debug_mode: bool):
    if not debug_mode:
        return

    state_text = tracking_state.name.lower()
    cv2.putText(preview, state_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(preview, state_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    similarity_text = f"similarity: {similarity_score:.3f}"
    cv2.putText(preview, similarity_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(preview, similarity_text, (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def draw_polygon_debug(
    preview: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
    line_thickness: int,
    text_thickness: int,
    enabled: bool,
):
    if not enabled:
        return

    for point_index, point in enumerate(points):
        point_tuple = geometry_utils.as_int_point(point)
        next_point = geometry_utils.as_int_point(points[(point_index + 1) % len(points)])
        cv2.circle(preview, point_tuple, 5, color, -1)
        cv2.line(preview, point_tuple, next_point, color, line_thickness)
        cv2.putText(preview, str(point_index), point_tuple, cv2.FONT_HERSHEY_SIMPLEX, 1.45, color, text_thickness, cv2.LINE_AA)


def draw_top_face_debug(preview: np.ndarray, top_face_points: list[tuple[int, int]], debug_mode: bool):
    if not debug_mode:
        return

    for point_index, point in enumerate(top_face_points):
        next_point = top_face_points[(point_index + 1) % len(top_face_points)]
        cv2.line(preview, point, next_point, (255, 0, 0), 4)


def get_top_face_label_position(top_face_points: list[tuple[int, int]]) -> tuple[int, int]:
    min_x = min(point[0] for point in top_face_points)
    max_x = max(point[0] for point in top_face_points)
    label_x = (min_x + max_x) // 2
    label_y = min(point[1] for point in top_face_points)
    return label_x, label_y


def get_polygon_edge_lengths(points: np.ndarray) -> list[float]:
    return [
        np.linalg.norm(points[(index + 1) % len(points)] - points[index])
        for index in range(len(points))
    ]


def main(config: AppConfig | None = None):
    if config is None:
        config = AppConfig()

    debug_mode = False
    capture = cv2.VideoCapture(config.video_source)
    if not capture.isOpened():
        print(f"Error: Could not open video source {config.video_source}.")
        sys.exit(1)

    if config.start_frame > 0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    capture_fps = capture.get(cv2.CAP_PROP_FPS)
    if capture_fps <= 0:
        capture_fps = 30.0
    playback_delay_ms = max(1, int(round(1000 / capture_fps)))
    paused = config.start_paused
    paused_frame = None
    redraw_paused_frame = False
    frame_number = config.start_frame - 1
    previous_mask = None
    stability_tracker = StabilityTracker(
        threshold=config.stable_similarity_threshold,
        required_stable_frames=config.count_sphere_required_count_frames,
        required_moving_frames=config.count_sphere_required_count_frames,
    )
    count_sphere_renderer = drawing.CountSphereRenderer(
        required_count_frames=config.count_sphere_required_count_frames,
    )
    current_frame = None
    video_writer = None
    video_writer_failed = False

    while True:
        if paused and paused_frame is not None and not redraw_paused_frame:
            key = cv2.waitKeyEx(playback_delay_ms)
            if key in QUIT_KEYS:
                break
            if key == SPACE_KEY:
                paused = False
                paused_frame = None
            elif key in DEBUG_KEYS:
                debug_mode = not debug_mode
                if not debug_mode:
                    close_debug_windows()
                redraw_paused_frame = True
            elif key == LEFT_ARROW_KEY:
                target_frame = max(config.start_frame, frame_number - 1)
                capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_number = target_frame - 1
                paused_frame = None
            elif key == RIGHT_ARROW_KEY:
                paused_frame = None
            continue

        if paused:
            if paused_frame is None:
                ret, frame = capture.read()
                if not ret:
                    print("Error: Failed to read frame from video source.")
                    break
                if config.flip_frame_horizontal:
                    cv2.flip(frame, 1, frame)
                frame_number += 1
                paused_frame = frame.copy()

            frame = paused_frame.copy()
            redraw_paused_frame = False
        else:
            ret, frame = capture.read()
            if not ret:
                print("Error: Failed to read frame from video source.")
                break
            if config.flip_frame_horizontal:
                cv2.flip(frame, 1, frame)
            frame_number += 1

        current_frame = frame.copy()

        result, previous_mask = run_pipeline(frame, config, previous_mask, frame_number, debug_mode)
        draw_pipeline_result(result, stability_tracker, count_sphere_renderer, debug_mode)

        if config.record_video_path is not None and not video_writer_failed:
            if video_writer is None:
                video_writer = open_video_writer(config.record_video_path, result.preview.shape, capture_fps)
                video_writer_failed = video_writer is None
            if video_writer is not None:
                video_writer.write(result.preview)
        show_windows(result, debug_mode)

        key = cv2.waitKeyEx(playback_delay_ms)
        if key in QUIT_KEYS:
            break
        if key == SPACE_KEY:
            paused = not paused
            paused_frame = current_frame.copy() if paused and current_frame is not None else None
        elif key in DEBUG_KEYS:
            debug_mode = not debug_mode
            if not debug_mode:
                close_debug_windows()
            if paused:
                redraw_paused_frame = True
        elif paused and key == LEFT_ARROW_KEY:
            target_frame = max(config.start_frame, frame_number - 1)
            capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            frame_number = target_frame - 1
            paused_frame = None
        elif paused and key == RIGHT_ARROW_KEY:
            paused_frame = None

    capture.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
