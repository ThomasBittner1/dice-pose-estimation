import cv2


def draw_count_sphere(image, count, text_origin):
    text = str(count)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 3
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    text_width, text_height = text_size
    center = (text_origin[0] + text_width // 2, text_origin[1] - text_height // 2)
    radius = max(24, max(text_width, text_height + baseline) // 2 + 14)

    center = (
        min(max(radius + 4, center[0]), image.shape[1] - radius - 4),
        min(max(radius + 4, center[1]), image.shape[0] - radius - 4),
    )
    text_origin = (center[0] - text_width // 2, center[1] + text_height // 2)

    overlay = image.copy()
    for current_radius in range(radius, 0, -1):
        ratio = current_radius / radius
        color = (
            int(45 + 40 * (1 - ratio)),
            int(135 + 80 * (1 - ratio)),
            int(235 + 20 * (1 - ratio)),
        )
        cv2.circle(overlay, center, current_radius, color, -1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    cv2.circle(image, center, radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, text, text_origin, font_face, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, text_origin, font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
