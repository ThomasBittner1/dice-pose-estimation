import cv2


class CountSphereRenderer:
    def __init__(self, required_count_frames=1, fade_speed=0.35, position_blend=0.2):
        self.required_count_frames = required_count_frames
        self.fade_speed = fade_speed
        self.position_blend = position_blend
        self.opacity = 0.0
        self.center = None
        self.count = None
        self.pending_count = None
        self.pending_count_frames = 0

    def get_visible_count(self, count):
        if count is None:
            return self.count

        if count == self.pending_count:
            self.pending_count_frames += 1
        else:
            self.pending_count = count
            self.pending_count_frames = 1

        if self.pending_count_frames >= self.required_count_frames:
            self.count = self.pending_count

        return self.count

    def update_and_draw(self, image, count, text_origin, is_stable):
        visible_count = self.get_visible_count(count)

        if text_origin is not None and visible_count is not None:
            target_center = _get_count_sphere_center(image, visible_count, text_origin)
            if self.center is None:
                self.center = target_center
            else:
                self.center = (
                    self.center[0] * (1.0 - self.position_blend) + target_center[0] * self.position_blend,
                    self.center[1] * (1.0 - self.position_blend) + target_center[1] * self.position_blend,
                )

        target_opacity = 1.0 if is_stable and visible_count is not None and self.center is not None else 0.0
        self.opacity += (target_opacity - self.opacity) * self.fade_speed

        if self.opacity <= 0.01 or self.center is None or visible_count is None:
            return

        _draw_count_sphere_at_center(image, visible_count, self.center, self.opacity)


def _get_count_sphere_center(image, count, text_origin):
    text = str(count)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 3
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    text_width, text_height = text_size
    radius = max(24, max(text_width, text_height + baseline) // 2 + 14)
    center = (text_origin[0] + text_width // 2, text_origin[1] - text_height // 2)

    return (
        min(max(radius + 4, float(center[0])), image.shape[1] - radius - 4),
        min(max(radius + 4, float(center[1])), image.shape[0] - radius - 4),
    )


def _draw_count_sphere_at_center(image, count, center, opacity):
    text = str(count)
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.1
    thickness = 3
    text_size, baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    text_width, text_height = text_size
    radius = max(24, max(text_width, text_height + baseline) // 2 + 14)

    center = (int(round(center[0])), int(round(center[1])))
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

    cv2.addWeighted(overlay, 0.82 * opacity, image, 1.0 - 0.82 * opacity, 0, image)
    decoration_overlay = image.copy()
    cv2.circle(decoration_overlay, center, radius, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(decoration_overlay, text, text_origin, font_face, font_scale, (20, 20, 20), thickness + 2, cv2.LINE_AA)
    cv2.putText(decoration_overlay, text, text_origin, font_face, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    cv2.addWeighted(decoration_overlay, opacity, image, 1.0 - opacity, 0, image)
