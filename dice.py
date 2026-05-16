import sys

import cv2
import numpy as np

CAMERA_INDEX = 0
PLAYBACK_DELAY_MS = 20
LEFT_ARROW_KEY = 2424832
RIGHT_ARROW_KEY = 2555904

START_FRAME = 6 # 218 #217 # 544 # 156 # 288 # 14
START_PAUSED = True
RUN_WHILE_ON_PAUSE = False


def find_strong_corner_points(outer_contour, epsilon_ratio):
    contour_perimeter = cv2.arcLength(outer_contour, True)
    approximated_contour = cv2.approxPolyDP(outer_contour, epsilon_ratio * contour_perimeter, True)
    return [tuple(point[0]) for point in approximated_contour]


def find_parallel_line_islands(points, angle_threshold_degrees=15.0, min_line_length=1.0):
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

    islands = [[valid_indices[0]]]
    island_reference_indices = [valid_indices[0]]
    for current_index in valid_indices[1:]:
        matched_island_index = None
        best_cosine_similarity = -1.0

        for island_index, reference_index in enumerate(island_reference_indices):
            cosine_similarity = abs(float(np.dot(normalized_vectors[reference_index], normalized_vectors[current_index])))
            if cosine_similarity >= cosine_threshold and cosine_similarity > best_cosine_similarity:
                matched_island_index = island_index
                best_cosine_similarity = cosine_similarity

        if matched_island_index is None:
            islands.append([current_index])
            island_reference_indices.append(current_index)
        else:
            islands[matched_island_index].append(current_index)

    return islands


def find_similar_directions(reference_direction, candidate_directions, angle_threshold_degrees=30):
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


def find_most_parallel_lines(from_point, to_point, hough_lines, threshold_angle_degrees=25):
    if hough_lines is None:
        return np.empty(0, dtype=int)

    reference_direction = np.asarray(to_point, dtype=np.float64) - np.asarray(from_point, dtype=np.float64)
    reference_length = np.linalg.norm(reference_direction)
    if reference_length < 1e-6:
        return np.empty(0, dtype=int)
    reference_direction /= reference_length

    hough_lines_points = np.asarray(hough_lines, dtype=np.float64).reshape(-1, 4)
    hough_line_directions = hough_lines_points[:, 2:4] - hough_lines_points[:, 0:2]
    return find_similar_directions(reference_direction, hough_line_directions, angle_threshold_degrees=threshold_angle_degrees)


def find_close_lines(from_point, to_point, hough_lines, threshold_distance, overlap_percentage_threshold=0.0):
    if hough_lines is None or to_point is None:
        return np.empty(0, dtype=int)

    reference_start = np.asarray(from_point, dtype=np.float64)
    reference_end = np.asarray(to_point, dtype=np.float64)
    reference_direction = reference_end - reference_start
    reference_length = np.linalg.norm(reference_direction)
    if reference_length < 1e-6:
        return np.empty(0, dtype=int)

    hough_lines_points = np.asarray(hough_lines, dtype=np.float64).reshape(-1, 4)
    line_starts = hough_lines_points[:, 0:2]
    line_ends = hough_lines_points[:, 2:4]
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


def triangle_is_line(inner_point, middle_point, outer_point, threshold_distance_perc=0.2):
    inner_point = np.asarray(inner_point, dtype=np.float64)
    middle_point = np.asarray(middle_point, dtype=np.float64)
    outer_point = np.asarray(outer_point, dtype=np.float64)
    line_direction = outer_point - inner_point
    line_length = np.linalg.norm(line_direction)
    if line_length < 1e-6:
        return False

    middle_offset = middle_point - inner_point
    perpendicular_distance = abs(line_direction[0] * middle_offset[1] - line_direction[1] * middle_offset[0]) / line_length
    return perpendicular_distance <= line_length * threshold_distance_perc


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


def find_hough_lines(frame, mask, outer_contour_rect, margin_perc=0.0, do_imshow=False):
    # find straight lines and draw them
    outer_contour_x, outer_contour_y, outer_contour_w, outer_contour_h = outer_contour_rect
    cropped_mask = mask[outer_contour_y:outer_contour_y + outer_contour_h, outer_contour_x:outer_contour_x + outer_contour_w]
    cropped_frame = frame[outer_contour_y:outer_contour_y + outer_contour_h, outer_contour_x:outer_contour_x + outer_contour_w]

    cropped_extracted_by_mask = cv2.bitwise_and(cropped_frame, cropped_frame, mask=cropped_mask)
    cropped_gray = cv2.cvtColor(cropped_extracted_by_mask, cv2.COLOR_BGR2GRAY)
    extracted_edges = cv2.Canny(cropped_gray, 0, 40)

    extracted_edges = cv2.bitwise_and(extracted_edges, extracted_edges, mask=cropped_mask)

    edges_preview = cv2.cvtColor(extracted_edges, cv2.COLOR_GRAY2BGR)
    hough_lines = cv2.HoughLinesP(extracted_edges, 1, np.pi / 180, threshold=20, minLineLength=max(10, outer_contour_w // 5), maxLineGap=max(4, outer_contour_w // 20))
    if hough_lines is not None:
        hough_lines_points = hough_lines.reshape(-1, 4)

        # remove lines below 15 % and lines above 15 % of the image
        line_mid_y = (hough_lines_points[:, 1] + hough_lines_points[:, 3]) / 2.0
        min_y = outer_contour_h * 0.15
        max_y = outer_contour_h * 0.85
        vertical_keep_mask = (line_mid_y >= min_y) & (line_mid_y <= max_y)
        margin_x = outer_contour_w * margin_perc
        margin_y = outer_contour_h * margin_perc
        first_point_near_border = ((hough_lines_points[:, 0] <= margin_x) | (hough_lines_points[:, 0] >= outer_contour_w - margin_x) |
                                   (hough_lines_points[:, 1] <= margin_y) | (hough_lines_points[:, 1] >= outer_contour_h - margin_y))
        second_point_near_border = ((hough_lines_points[:, 2] <= margin_x) | (hough_lines_points[:, 2] >= outer_contour_w - margin_x) |
                                    (hough_lines_points[:, 3] <= margin_y) | (hough_lines_points[:, 3] >= outer_contour_h - margin_y))
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
        cv2.imshow('cropped_extracted_by_mask', cropped_extracted_by_mask)
        cv2.imshow("Dice Edges", edges_preview)

    # calculate hough_directions
    if hough_lines is None:
        return None, None, None

    else:
        hough_lines_points = hough_lines.reshape(-1, 4).astype(np.float64)
        hough_directions = hough_lines_points[:, 2:4] - hough_lines_points[:, 0:2]
        hough_lines_lengths = np.linalg.norm(hough_directions, axis=1)
        valid_line_mask = hough_lines_lengths > 1e-6
        hough_directions = hough_directions[valid_line_mask]
        hough_lines_lengths = hough_lines_lengths[valid_line_mask]
        if len(hough_directions) > 0:
            hough_directions /= hough_lines_lengths[:, None]

        return hough_lines, hough_lines_lengths, hough_directions



def main():
    cap = cv2.VideoCapture('green_cube_2.mp4')
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        sys.exit(1)

    if START_FRAME > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    paused = START_PAUSED
    paused_frame = None
    frame_number = START_FRAME - 1

    while True:
        if paused and paused_frame is not None and not RUN_WHILE_ON_PAUSE:
            key = cv2.waitKeyEx(PLAYBACK_DELAY_MS)
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = False
                paused_frame = None
            elif key == LEFT_ARROW_KEY:
                target_frame = max(START_FRAME, frame_number - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_number = target_frame - 1
                paused_frame = None
            elif key == RIGHT_ARROW_KEY:
                paused_frame = None
            continue

        if paused:
            if paused_frame is None:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from webcam.")
                    break
                cv2.flip(frame, 1, frame)
                frame_number += 1
                paused_frame = frame.copy()
            frame = paused_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break
            cv2.flip(frame, 1, frame)
            frame_number += 1
        # print ('frame_number: ', frame_number)
        preview = frame.copy()
        top_face_warp = None
        blurred_mask_preview = None
        frame_text = f"frame: {frame_number}"
        text_size, _ = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = preview.shape[1] - text_size[0] - 20
        cv2.putText(preview, frame_text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


        # extract green color
        hsv = cv2.cvtColor(preview, cv2.COLOR_BGR2HSV)
        # mask = cv2.inRange(hsv, np.array([39, 85, 0], dtype=np.uint8), np.array([95, 255, 253], dtype=np.uint8))
        mask = cv2.inRange(hsv, np.array([26, 59, 30], dtype=np.uint8), np.array([98, 255, 250], dtype=np.uint8))

        # draw the outer contour of the mask into preview
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            outer_contour = max(contours, key=cv2.contourArea)
            outer_contour_rect = cv2.boundingRect(outer_contour)
            outer_contour_x, outer_contour_y, outer_contour_w, outer_contour_h = outer_contour_rect

            cv2.drawContours(preview, [outer_contour], -1, (255, 255, 255), 1)
            # for contour_point in outer_contour.reshape(-1, 2):
            #     cv2.circle(preview, tuple(contour_point), 4, (255, 255, 0), -1)

            points = find_strong_corner_points(outer_contour, 0.02) #corner_epsilon_ratio)
            points = np.squeeze(points)
            parallels = find_parallel_line_islands(points, min_line_length=0.1, angle_threshold_degrees=30)
            edge_lengths = [np.linalg.norm(points[(i+1) % len(points)]-points[i]) for i in range(len(points))]

            # drawing the points that where found from find_strong_corner_points
            #
            for p, point in enumerate(points):
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
                cv2.line(preview, point, points[p + 1 if p < len(points) - 1 else 0], (0, 255, 0), 3)
                cv2.putText(preview, str(p), point, cv2.FONT_HERSHEY_SIMPLEX, 1.45, (0, 255, 0), 2, cv2.LINE_AA)

            if len(points) == 4: # [[0, 2], [1, 3]]
                # hough_lines = hough_lines.squeeze()

                if len(parallels) == 2 and len(parallels[0]) == 2 and len(parallels[1]) == 2:
                    hough_lines, hough_lines_lengths, hough_directions = find_hough_lines(frame, mask, outer_contour_rect, do_imshow=True)

                    ratio = 0.5
                    if hough_lines is not None:
                        highest_point_indices = np.argsort(points[:, 1], axis=0)[:2]
                        highest_line_direction = (points[highest_point_indices[1]] - points[highest_point_indices[0]]).astype('float64')
                        highest_line_direction /= np.linalg.norm(highest_line_direction)

                        similar_directions_indices = find_similar_directions(highest_line_direction, hough_directions, angle_threshold_degrees=30)
                        if len(similar_directions_indices):
                            longest_similar_hough_index2 = np.argmax(hough_lines_lengths[similar_directions_indices])
                            longest_similar_line = hough_lines[similar_directions_indices][longest_similar_hough_index2][0]
                            ratio = ((longest_similar_line[1] + longest_similar_line[3]) * 0.5) / outer_contour_h

                    longer_index = np.argmax([edge_lengths[x[0]] + edge_lengths[x[1]] for x in parallels])
                    longer_pair = parallels[longer_index]


                    inserts = {}
                    for x, from_point_index in enumerate(longer_pair):
                        if x == 1:
                            ratio = 1 - ratio

                        new_point = (points[from_point_index] * (1-ratio) + points[(from_point_index + 1) % len(points)] * ratio).astype(int)
                        inserts[(from_point_index + 1) % len(points)] = new_point
                    for at_index in sorted(inserts.keys(), reverse=True):
                        points = points.tolist()
                        points.insert(at_index, inserts[at_index])
                        points = np.array(points, dtype=int)


            elif len(points) == 5:
                if len([x for x in parallels if len(x) == 1]) == 2:
                    print ('invalid', frame_number + 1)
                else:
                    if len(parallels) == 2: # parallels == [[0, 3], [1, 2, 4]]; [[0, 2, 3], [1, 4]]
                        bigger_index = np.argmax([len(x) for x in parallels]) # 1 - smaller_one
                        tripple = parallels[bigger_index]
                        for x in range(len(tripple)):
                            if abs(tripple[x] - tripple[(x-1) % 3]) != 1 and abs(tripple[x] - tripple[(x+1) % 3]) != 1:
                                separate = tripple[x]
                        others = set(tripple) - set([separate])
                        parallels.pop(bigger_index)
                        parallels.append([others.pop()])
                        parallels.append([separate, others.pop()])

                    if len(parallels) == 3: # parallels == [x,x],[x,x],[x] # [[0], [1, 3], [2, 4]] # wrong: [[1, 4], [2], [0, 3]]
                        single_index = np.argmin([len(x) for x in parallels])
                        double_indices = list(set([0,1,2]) - set([single_index]))
                        longer_pair_index2 = np.argmax([max(edge_lengths[parallels[x][0]], edge_lengths[parallels[x][1]]) for x in double_indices])
                        longer_pair_index = double_indices[longer_pair_index2]
                        longer_pair = parallels[longer_pair_index]

                        single_edge = parallels[single_index][0]
                        cut = 0 if abs(single_edge-longer_pair[0]) != 1 else 1
                        keep = 1-cut
                        ratio = edge_lengths[single_edge] / (edge_lengths[single_edge] + edge_lengths[longer_pair[keep]])
                        from_point_index = longer_pair[cut]
                        new_point = (points[from_point_index] * ratio + points[(from_point_index+1) % len(points)] * (1 - ratio)).astype(int)
                        points = points.tolist()
                        points.insert((from_point_index+1) % len(points), new_point)
                        points = np.array(points, dtype=int)


            # drawing the points that got fixed
            #
            if True:
                for p, point in enumerate(points):
                    cv2.circle(preview, point, 5, (0, 0, 255), -1)
                    cv2.line(preview, point, points[p + 1 if p < len(points) - 1 else 0], (0, 0, 255), 2)
                    cv2.putText(preview, str(p), point, cv2.FONT_HERSHEY_SIMPLEX, 1.45, (0, 0, 255), 1, cv2.LINE_AA)

            if len(points) == 6:
                highest_two = np.argsort(points[:, 1])[:2]
                cross_points = [None] * len(highest_two)

                hough_lines, hough_lines_lengths, _ = find_hough_lines(frame, mask, outer_contour_rect, margin_perc=0.2, do_imshow=True)

                two_distance_sums = np.zeros(2, dtype=np.float64)

                crop_offset = np.array([outer_contour_x, outer_contour_y], dtype=np.float64)
                cropped_points = points.astype(np.float64) - crop_offset

                final_highest_p = None
                final_cross_point = None
                for index, highest_p in enumerate(highest_two):
                    point_indices = (np.arange(6) + highest_p) % len(points)

                    cross_point = intersect_rays((points[point_indices[5]], points[point_indices[5]] + points[point_indices[1]] - points[point_indices[0]]),
                                                 (points[point_indices[1]], points[point_indices[1]] + points[point_indices[5]] - points[point_indices[0]]))
                    if cross_point is None:
                        continue

                    if triangle_is_line(cropped_points[point_indices[0]], cropped_points[point_indices[1]], cropped_points[point_indices[2]], threshold_distance_perc=0.2) and \
                        triangle_is_line(cropped_points[point_indices[2]], cropped_points[point_indices[3]], cropped_points[point_indices[4]], threshold_distance_perc=0.2):
                        final_highest_p = highest_p
                        final_cross_point = cross_point
                        print ('triangle is line')

                    cross_points[index] = cross_point

                    if hough_lines is None:
                        final_highest_p = highest_p
                        final_cross_point = cross_point
                        break


                    cropped_cross_point = None if cross_point is None else np.asarray(cross_point, dtype=np.float64) - crop_offset
                    most_parallel_line_indices = find_most_parallel_lines(cropped_points[point_indices[1]], cropped_cross_point, hough_lines,
                                                                          threshold_angle_degrees = 15)
                    close_line_indices2 = find_close_lines(cropped_points[point_indices[1]], cropped_cross_point, hough_lines[most_parallel_line_indices],
                                                          threshold_distance = outer_contour_w*0.1, overlap_percentage_threshold=0.4)
                    close_line_indices = most_parallel_line_indices[close_line_indices2]
                    two_distance_sums[index] += np.sum(hough_lines_lengths[close_line_indices])

                    most_parallel_line_indices = find_most_parallel_lines(cropped_points[point_indices[5]], cropped_cross_point, hough_lines,
                                                                          threshold_angle_degrees = 15)
                    close_line_indices2 = find_close_lines(cropped_points[point_indices[5]], cropped_cross_point, hough_lines[most_parallel_line_indices],
                                                          threshold_distance = outer_contour_w * 0.1, overlap_percentage_threshold=0.4)
                    close_line_indices = most_parallel_line_indices[close_line_indices2]
                    two_distance_sums[index] += np.sum(hough_lines_lengths[close_line_indices])

                    most_parallel_line_indices = find_most_parallel_lines(cropped_points[point_indices[3]], cropped_cross_point, hough_lines,
                                                                          threshold_angle_degrees = 15)
                    close_line_indices2 = find_close_lines(cropped_points[point_indices[3]], cropped_cross_point, hough_lines[most_parallel_line_indices],
                                                          threshold_distance = outer_contour_w * 0.1, overlap_percentage_threshold=0.4)
                    close_line_indices = most_parallel_line_indices[close_line_indices2]
                    two_distance_sums[index] += np.sum(hough_lines_lengths[close_line_indices])

                if final_highest_p == None:
                    best_index = np.argmax(two_distance_sums)
                    final_highest_p = highest_two[best_index]
                if final_cross_point is None:
                    final_cross_point = cross_points[best_index]
                points = np.roll(points, -final_highest_p, axis=0)  # reorder so the highest one is at the top

                # drawing the points that got reordered
                #
                if True:
                    for p, point in enumerate(points):
                        cv2.circle(preview, point, 5, (255, 0, 0), -1)
                        cv2.line(preview, point, points[p + 1 if p < len(points) - 1 else 0], (255, 0, 0), 2)
                        cv2.putText(preview, str(p), point, cv2.FONT_HERSHEY_SIMPLEX, 1.45, (255, 0, 0), 2, cv2.LINE_AA)

                if final_cross_point is not None:

                    cv2.circle(preview, as_int_point(final_cross_point), 5, (0, 0, 255), -1)

                    top_face_points = [as_int_point(points[0]), as_int_point(points[1]), as_int_point(final_cross_point), as_int_point(points[5])]
                    for p,point in enumerate(top_face_points):
                        cv2.line(preview, point, top_face_points[p + 1 if p < len(top_face_points) - 1 else 0], (255, 0, 0), 4)

                    # homography the top_face_points into a new image, and show it with cv2.imshow
                    #
                    source_points = np.array(top_face_points, dtype=np.float32)
                    top_width = max(1, int(round(max(np.linalg.norm(source_points[1] - source_points[0]), np.linalg.norm(source_points[2] - source_points[3])))))
                    top_height = max(1, int(round(max(np.linalg.norm(source_points[3] - source_points[0]), np.linalg.norm(source_points[2] - source_points[1])))))
                    top_size = max(top_width, top_height)
                    destination_points = np.array([[0, 0], [top_size - 1, 0], [top_size - 1, top_size - 1], [0, top_size - 1]], dtype=np.float32)
                    homography = cv2.getPerspectiveTransform(source_points, destination_points)
                    top_face_warp = cv2.warpPerspective(frame, homography, (top_size, top_size))

                    hsv = cv2.cvtColor(top_face_warp, cv2.COLOR_BGR2HSV)
                    green_range = cv2.inRange(hsv, np.array([61, 42, 100], dtype=np.uint8), np.array([81, 255, 248], dtype=np.uint8))
                    mask = cv2.bitwise_not(green_range)
                    # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    num_dots = 0
                    blurred_mask = cv2.GaussianBlur(mask, (9, 9), 2)
                    blurred_mask_preview = blurred_mask
                    min_radius = max(3, top_size // 16)
                    max_radius = max(min_radius + 1, top_size // 6)
                    circles = cv2.HoughCircles(
                        blurred_mask,
                        cv2.HOUGH_GRADIENT,
                        dp=1.2,
                        minDist=max(10, top_size // 5),
                        param1=120,
                        param2=20,
                        minRadius=min_radius,
                        maxRadius=max_radius,
                    )

                    # print the radien of all circles found
                    if circles is not None:
                        rounded_circles = np.round(circles[0]).astype(int)
                        num_dots = len(rounded_circles)
                        for circle_x, circle_y, circle_radius in rounded_circles:
                            cv2.circle(top_face_warp, (circle_x, circle_y), circle_radius, (0, 255, 255), 2)
                            cv2.circle(top_face_warp, (circle_x, circle_y), 2, (255, 0, 255), -1)
                    label_x = min(point[0] for point in top_face_points)
                    label_y = max(20, min(point[1] for point in top_face_points) - 12)
                    cv2.putText(preview, str(num_dots), (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
                            
        cv2.imshow("Dice Final", preview)


        if top_face_warp is not None:
            cv2.imshow("Dice Top Face", top_face_warp)
            if blurred_mask_preview is not None:
                cv2.imshow("Dice Blurred Mask", blurred_mask_preview)

        key = cv2.waitKeyEx(PLAYBACK_DELAY_MS)
        if key in (ord("q"), ord("Q")):
            break
        if key == ord(" "):
            paused = not paused
            if paused:
                paused_frame = frame.copy()
            else:
                paused_frame = None
        elif paused and key == LEFT_ARROW_KEY:
            target_frame = max(START_FRAME, frame_number - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            frame_number = target_frame - 1
            paused_frame = None
        elif paused and key == RIGHT_ARROW_KEY:
            paused_frame = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
