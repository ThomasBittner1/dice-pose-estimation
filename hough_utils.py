import cv2
import numpy as np


def detect_hough_lines_in_contour_roi(
    frame,
    mask,
    outer_contour_rect,
    canny_threshold1,
    canny_threshold2,
    hough_rho,
    hough_theta_degrees,
    hough_threshold,
    min_line_length_min_pixels,
    min_line_length_width_divisor,
    max_line_gap_min_pixels,
    max_line_gap_width_divisor,
    vertical_keep_min_y_ratio,
    vertical_keep_max_y_ratio,
    margin_perc,
    do_imshow=False,
):
    outer_contour_x, outer_contour_y, outer_contour_w, outer_contour_h = outer_contour_rect
    cropped_mask = mask[outer_contour_y:outer_contour_y + outer_contour_h, outer_contour_x:outer_contour_x + outer_contour_w]
    cropped_frame = frame[outer_contour_y:outer_contour_y + outer_contour_h, outer_contour_x:outer_contour_x + outer_contour_w]

    cropped_extracted_by_mask = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)
    cropped_gray = cv2.cvtColor(cropped_extracted_by_mask, cv2.COLOR_BGR2GRAY)
    extracted_edges = cv2.Canny(cropped_gray, canny_threshold1, canny_threshold2)
    extracted_edges = cv2.bitwise_and(extracted_edges, extracted_edges, mask=cropped_mask)

    edges_preview = cv2.cvtColor(extracted_edges, cv2.COLOR_GRAY2BGR)
    hough_lines = cv2.HoughLinesP(
        extracted_edges,
        hough_rho,
        np.deg2rad(hough_theta_degrees),
        threshold=hough_threshold,
        minLineLength=max(min_line_length_min_pixels, outer_contour_w // min_line_length_width_divisor),
        maxLineGap=max(max_line_gap_min_pixels, outer_contour_w // max_line_gap_width_divisor),
    )
    if hough_lines is not None:
        hough_lines_points = hough_lines.reshape(-1, 4)

        line_mid_y = (hough_lines_points[:, 1] + hough_lines_points[:, 3]) / 2.0
        min_y = outer_contour_h * vertical_keep_min_y_ratio
        max_y = outer_contour_h * vertical_keep_max_y_ratio
        vertical_keep_mask = (line_mid_y >= min_y) & (line_mid_y <= max_y)
        margin_x = outer_contour_w * margin_perc
        margin_y = outer_contour_h * margin_perc
        first_point_near_border = (
            (hough_lines_points[:, 0] <= margin_x)
            | (hough_lines_points[:, 0] >= outer_contour_w - margin_x)
            | (hough_lines_points[:, 1] <= margin_y)
            | (hough_lines_points[:, 1] >= outer_contour_h - margin_y)
        )
        second_point_near_border = (
            (hough_lines_points[:, 2] <= margin_x)
            | (hough_lines_points[:, 2] >= outer_contour_w - margin_x)
            | (hough_lines_points[:, 3] <= margin_y)
            | (hough_lines_points[:, 3] >= outer_contour_h - margin_y)
        )
        border_keep_mask = ~(first_point_near_border & second_point_near_border)
        hough_lines = hough_lines_points[vertical_keep_mask & border_keep_mask].reshape(-1, 1, 4)

        for line_index, line in enumerate(hough_lines[:, 0]):
            x1, y1, x2, y2 = line
            label_point = ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.line(edges_preview, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(edges_preview, str(line_index), label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(edges_preview, str(line_index), label_point, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(cropped_extracted_by_mask, (x1, y1), (x2, y2), (255, 255, 0), 2)

    if do_imshow:
        cv2.imshow("cropped_extracted_by_mask", cropped_extracted_by_mask)
        cv2.imshow("Dice Edges", edges_preview)

    if hough_lines is None:
        return None, None, None

    hough_lines_points = hough_lines.reshape(-1, 4).astype(np.float64)
    hough_directions = hough_lines_points[:, 2:4] - hough_lines_points[:, 0:2]
    hough_lines_lengths = np.linalg.norm(hough_directions, axis=1)
    valid_line_mask = hough_lines_lengths > 1e-6
    hough_directions = hough_directions[valid_line_mask]
    hough_lines_lengths = hough_lines_lengths[valid_line_mask]
    if len(hough_directions) > 0:
        hough_directions /= hough_lines_lengths[:, None]

    return hough_lines, hough_lines_lengths, hough_directions
