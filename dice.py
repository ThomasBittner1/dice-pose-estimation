import sys
from dataclasses import dataclass

import cv2
import numpy as np

import geometry_utils
from hough_utils import detect_hough_lines_in_contour_roi


@dataclass
class AppConfig:
    camera_index = 0
    playback_delay_ms = 20
    left_arrow_key = 2424832
    right_arrow_key = 2555904
    start_frame = 6 # 218 #217 # 544 # 156 # 288 # 14
    start_paused = True
    run_while_on_pause = False


def main(config: AppConfig | None = None):
    if config is None:
        config = AppConfig()

    cap = cv2.VideoCapture('green_cube_2.mp4')
    if not cap.isOpened():
        print(f"Error: Could not open camera {config.camera_index}.")
        sys.exit(1)

    if config.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    paused = config.start_paused
    paused_frame = None
    frame_number = config.start_frame - 1

    while True:
        if paused and paused_frame is not None and not config.run_while_on_pause:
            key = cv2.waitKeyEx(config.playback_delay_ms)
            if key in (ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = False
                paused_frame = None
            elif key == config.left_arrow_key:
                target_frame = max(config.start_frame, frame_number - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                frame_number = target_frame - 1
                paused_frame = None
            elif key == config.right_arrow_key:
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

            points = geometry_utils.approximate_contour_corners(outer_contour, 0.02) #corner_epsilon_ratio)
            points = np.squeeze(points)
            parallels = geometry_utils.group_polygon_edges_by_parallel_direction(points, min_line_length=0.1, angle_threshold_degrees=30)
            edge_lengths = [np.linalg.norm(points[(i+1) % len(points)]-points[i]) for i in range(len(points))]

            # drawing the points that where found from approximate_contour_corners
            #
            for p, point in enumerate(points):
                cv2.circle(preview, point, 5, (0, 0, 255), -1)
                cv2.line(preview, point, points[p + 1 if p < len(points) - 1 else 0], (0, 255, 0), 3)
                cv2.putText(preview, str(p), point, cv2.FONT_HERSHEY_SIMPLEX, 1.45, (0, 255, 0), 2, cv2.LINE_AA)

            if len(points) == 4: # [[0, 2], [1, 3]]
                # hough_lines = hough_lines.squeeze()

                if len(parallels) == 2 and len(parallels[0]) == 2 and len(parallels[1]) == 2:
                    hough_lines, hough_lines_lengths, hough_directions = detect_hough_lines_in_contour_roi(frame, mask, outer_contour_rect, do_imshow=True)

                    ratio = 0.5
                    if hough_lines is not None:
                        highest_point_indices = np.argsort(points[:, 1], axis=0)[:2]
                        highest_line_direction = (points[highest_point_indices[1]] - points[highest_point_indices[0]]).astype('float64')
                        highest_line_direction /= np.linalg.norm(highest_line_direction)

                        similar_directions_indices = geometry_utils.find_direction_aligned_indices(highest_line_direction, hough_directions, angle_threshold_degrees=30)
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

                hough_lines, hough_lines_lengths, _ = detect_hough_lines_in_contour_roi(frame, mask, outer_contour_rect, margin_perc=0.2, do_imshow=True)

                two_distance_sums = np.zeros(2, dtype=np.float64)

                crop_offset = np.array([outer_contour_x, outer_contour_y], dtype=np.float64)
                cropped_points = points.astype(np.float64) - crop_offset

                final_highest_p = None
                final_cross_point = None
                for index, highest_p in enumerate(highest_two):
                    point_indices = (np.arange(6) + highest_p) % len(points)

                    cross_point = geometry_utils.intersect_rays((points[point_indices[5]], points[point_indices[5]] + points[point_indices[1]] - points[point_indices[0]]),
                                                 (points[point_indices[1]], points[point_indices[1]] + points[point_indices[5]] - points[point_indices[0]]))
                    if cross_point is None:
                        continue

                    if geometry_utils.points_are_collinear(cropped_points[point_indices[0]], cropped_points[point_indices[1]], cropped_points[point_indices[2]], threshold_distance_ratio=0.2) and \
                        geometry_utils.points_are_collinear(cropped_points[point_indices[2]], cropped_points[point_indices[3]], cropped_points[point_indices[4]], threshold_distance_ratio=0.2):
                        final_highest_p = highest_p
                        final_cross_point = cross_point
                        print ('triangle is line')

                    cross_points[index] = cross_point

                    if hough_lines is None:
                        final_highest_p = highest_p
                        final_cross_point = cross_point
                        break


                    cropped_cross_point = None if cross_point is None else np.asarray(cross_point, dtype=np.float64) - crop_offset
                    most_parallel_line_indices = geometry_utils.find_lines_parallel_to_segment(cropped_points[point_indices[1]], cropped_cross_point, hough_lines,
                                                                          threshold_angle_degrees = 15)
                    close_line_indices2 = geometry_utils.find_lines_near_segment(cropped_points[point_indices[1]], cropped_cross_point, hough_lines[most_parallel_line_indices],
                                                          threshold_distance = outer_contour_w*0.1, overlap_percentage_threshold=0.4)
                    close_line_indices = most_parallel_line_indices[close_line_indices2]
                    two_distance_sums[index] += np.sum(hough_lines_lengths[close_line_indices])

                    most_parallel_line_indices = geometry_utils.find_lines_parallel_to_segment(cropped_points[point_indices[5]], cropped_cross_point, hough_lines,
                                                                          threshold_angle_degrees = 15)
                    close_line_indices2 = geometry_utils.find_lines_near_segment(cropped_points[point_indices[5]], cropped_cross_point, hough_lines[most_parallel_line_indices],
                                                          threshold_distance = outer_contour_w * 0.1, overlap_percentage_threshold=0.4)
                    close_line_indices = most_parallel_line_indices[close_line_indices2]
                    two_distance_sums[index] += np.sum(hough_lines_lengths[close_line_indices])

                    most_parallel_line_indices = geometry_utils.find_lines_parallel_to_segment(cropped_points[point_indices[3]], cropped_cross_point, hough_lines,
                                                                          threshold_angle_degrees = 15)
                    close_line_indices2 = geometry_utils.find_lines_near_segment(cropped_points[point_indices[3]], cropped_cross_point, hough_lines[most_parallel_line_indices],
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

                    cv2.circle(preview, geometry_utils.as_int_point(final_cross_point), 5, (0, 0, 255), -1)

                    top_face_points = [geometry_utils.as_int_point(points[0]), geometry_utils.as_int_point(points[1]), geometry_utils.as_int_point(final_cross_point), geometry_utils.as_int_point(points[5])]
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

        key = cv2.waitKeyEx(config.playback_delay_ms)
        if key in (ord("q"), ord("Q")):
            break
        if key == ord(" "):
            paused = not paused
            if paused:
                paused_frame = frame.copy()
            else:
                paused_frame = None
        elif paused and key == config.left_arrow_key:
            target_frame = max(config.start_frame, frame_number - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            frame_number = target_frame - 1
            paused_frame = None
        elif paused and key == config.right_arrow_key:
            paused_frame = None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
