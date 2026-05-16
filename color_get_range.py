import json
import sys
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

import colors_utils

COLOR_NAME = "green"
DEFAULT_MIN_VALS = (26, 59, 30)
DEFAULT_MAX_VALS = (98, 255, 250)
DISPLAY_COLOR = (0, 255, 0)

CONFIG_PATH = Path("hsv_config.json")
CALIBRATION_PATH = Path("camera_calibration.json")
POSE_MARKER_POINTS = {
    "red": (0.0, 0.0, 0.0),
    "green": (0.0, 4.5, 0.0),
    "yellow": (-7.071, 0.0, -2.929),
    "blue": (-7.071, 4.5, -2.928),
}


def _is_valid_hsv_triplet(values):
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        return False
    return (
        isinstance(values[0], int) and 0 <= values[0] <= 179
        and isinstance(values[1], int) and 0 <= values[1] <= 255
        and isinstance(values[2], int) and 0 <= values[2] <= 255
    )


def save_config(config):
    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def get_config():
    if CONFIG_PATH.exists():
        try:
            with CONFIG_PATH.open("r", encoding="utf-8") as file:
                config = json.load(file)
            saved_color = config.get(COLOR_NAME, {}) if isinstance(config, dict) else {}
            min_vals = saved_color.get("min_vals")
            max_vals = saved_color.get("max_vals")
            invert = bool(saved_color.get("invert", False))
            if _is_valid_hsv_triplet(min_vals) and _is_valid_hsv_triplet(max_vals):
                return {COLOR_NAME: {"min_vals": tuple(min_vals), "max_vals": tuple(max_vals), "invert": invert}}
        except (json.JSONDecodeError, OSError):
            pass

    return {
        COLOR_NAME: {
            "min_vals": DEFAULT_MIN_VALS,
            "max_vals": DEFAULT_MAX_VALS,
            "invert": False,
        }
    }


class SliderChangeCommand(QtGui.QUndoCommand):
    def __init__(self, slider_row, old_value, new_value):
        super().__init__(f"Change {slider_row.label.text()} from {old_value} to {new_value}")
        self.slider_row = slider_row
        self.old_value = old_value
        self.new_value = new_value

    def undo(self):
        self.slider_row.set_value_from_undo(self.old_value)

    def redo(self):
        self.slider_row.set_value_from_undo(self.new_value)


class HuePickCommand(QtGui.QUndoCommand):
    def __init__(self, controls_dialog, color_name, old_min, old_max, new_min, new_max):
        super().__init__(f"Pick hue for {color_name}")
        self.controls_dialog = controls_dialog
        self.color_name = color_name
        self.old_min = old_min
        self.old_max = old_max
        self.new_min = new_min
        self.new_max = new_max

    def undo(self):
        self.controls_dialog._set_hue_range(self.color_name, self.old_min, self.old_max)

    def redo(self):
        self.controls_dialog._set_hue_range(self.color_name, self.new_min, self.new_max)


class SliderRow(QtWidgets.QHBoxLayout):
    def __init__(self, label_text, minimum, maximum, value, on_change, undo_stack):
        super().__init__()

        self.on_change = on_change
        self.undo_stack = undo_stack
        self._drag_start_value = value
        self._ignore_undo = False

        self.label = QtWidgets.QLabel(label_text)
        self.label.setMinimumWidth(60)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)

        self.value_label = QtWidgets.QLabel(str(value))
        self.value_label.setMinimumWidth(40)

        self.addWidget(self.label)
        self.addWidget(self.slider)
        self.addWidget(self.value_label)

        self.slider.valueChanged.connect(self._handle_value_changed)
        self.slider.sliderPressed.connect(self._handle_slider_pressed)
        self.slider.sliderReleased.connect(self._handle_slider_released)

    def _handle_value_changed(self, value):
        self.value_label.setText(str(value))
        self.on_change()

    def _handle_slider_pressed(self):
        self._drag_start_value = self.slider.value()

    def _handle_slider_released(self):
        if self._ignore_undo:
            return

        new_value = self.slider.value()
        old_value = self._drag_start_value

        if new_value != old_value:
            self.undo_stack.push(SliderChangeCommand(self, old_value, new_value))

    def set_value_from_undo(self, value):
        self._ignore_undo = True
        self.slider.setValue(value)
        self._ignore_undo = False

    def value(self):
        return self.slider.value()


class VideoLabel(QtWidgets.QLabel):
    clicked = QtCore.Signal(int, int)
    hovered = QtCore.Signal(int, int)
    left = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setMinimumSize(640, 480)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background-color: black;")
        self.setFrameShape(QtWidgets.QFrame.Shape.Box)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self.clicked.emit(int(event.position().x()), int(event.position().y()))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        self.hovered.emit(int(event.position().x()), int(event.position().y()))
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.left.emit()
        super().leaveEvent(event)


class HSVControlsDialog(QtWidgets.QDialog):
    def __init__(self, config):
        super().__init__()
        self.setWindowTitle("HSV Controls")
        self.resize(1500, 700)

        self.undo_stack = QtGui.QUndoStack(self)
        self.rows = {}
        self.pick_buttons = {}
        self.active_picker_color = None
        self._updating_pick_buttons = False
        self.latest_hsv_frame = None
        self.latest_frame_size = None
        self.latest_source_frame_size = None
        self.latest_pixmap_size = None
        self._last_rgb_frame = None

        root_layout = QtWidgets.QHBoxLayout(self)

        controls_container = QtWidgets.QWidget()
        controls_layout = QtWidgets.QVBoxLayout(controls_container)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setChecked(True)
        controls_layout.addWidget(self.play_button)

        self.hover_hsv_label = QtWidgets.QLabel("Hover HSV: -")
        controls_layout.addWidget(self.hover_hsv_label)

        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(controls_container)
        scroll_area.setMinimumWidth(420)

        self.video_label = VideoLabel()
        self.video_label.clicked.connect(self._handle_video_clicked)
        self.video_label.hovered.connect(self._handle_video_hovered)
        self.video_label.left.connect(self._handle_video_left)
        self.hover_hsv_popup = QtWidgets.QLabel(self.video_label)
        self.hover_hsv_popup.setStyleSheet("background-color: rgba(0, 0, 0, 190); color: white; border: 1px solid #808080; padding: 4px;")
        self.hover_hsv_popup.hide()

        root_layout.addWidget(scroll_area, stretch=0)
        root_layout.addWidget(self.video_label, stretch=1)

        for color_name, color_data in config.items():
            group = QtWidgets.QGroupBox(color_name)
            group_layout = QtWidgets.QFormLayout(group)

            min_vals = color_data["min_vals"]
            max_vals = color_data["max_vals"]

            color_rows = {
                "h_min": SliderRow("H min", 0, 179, min_vals[0], self.save_values, self.undo_stack),
                "h_max": SliderRow("H max", 0, 179, max_vals[0], self.save_values, self.undo_stack),
                "s_min": SliderRow("S min", 0, 255, min_vals[1], self.save_values, self.undo_stack),
                "s_max": SliderRow("S max", 0, 255, max_vals[1], self.save_values, self.undo_stack),
                "v_min": SliderRow("V min", 0, 255, min_vals[2], self.save_values, self.undo_stack),
                "v_max": SliderRow("V max", 0, 255, max_vals[2], self.save_values, self.undo_stack),
            }

            self.rows[color_name] = color_rows

            for row in color_rows.values():
                group_layout.addRow(row)

            pick_button = QtWidgets.QPushButton("Pick Hue")
            pick_button.setCheckable(True)
            pick_button.toggled.connect(partial(self._handle_pick_button_toggled, color_name))
            self.pick_buttons[color_name] = pick_button
            group_layout.addRow("Picker", pick_button)

            invert_checkbox = QtWidgets.QCheckBox("Invert")
            invert_checkbox.setChecked(bool(color_data.get("invert", False)))
            invert_checkbox.toggled.connect(self.save_values)
            color_rows["invert"] = invert_checkbox
            group_layout.addRow("Mask", invert_checkbox)

            controls_layout.addWidget(group)

        controls_layout.addStretch()

        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.undo_shortcut.activated.connect(self.undo_stack.undo)

        self.redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.redo_shortcut.activated.connect(self.undo_stack.redo)

        self.quit_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Q"), self)
        self.quit_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.quit_shortcut.activated.connect(self.close)

    def _handle_pick_button_toggled(self, color_name, checked):
        if self._updating_pick_buttons:
            return

        self._updating_pick_buttons = True
        try:
            if checked:
                for other_color_name, other_button in self.pick_buttons.items():
                    if other_color_name != color_name:
                        other_button.setChecked(False)
                self.active_picker_color = color_name
            elif self.active_picker_color == color_name:
                self.active_picker_color = None
        finally:
            self._updating_pick_buttons = False

    def _handle_video_clicked(self, label_x, label_y):
        active_color = self.get_active_picker_color()
        if active_color is None or self.latest_hsv_frame is None:
            return

        mapped_point = self._map_label_point_to_source(label_x, label_y)
        if mapped_point is None:
            return

        image_x, image_y = mapped_point
        picked_hsv = self.latest_hsv_frame[image_y, image_x]
        picked_hue = int(picked_hsv[0])
        self.set_hue_range_from_pick(active_color, picked_hue)

    def _map_label_point_to_source(self, label_x, label_y):
        if self.latest_frame_size is None or self.latest_pixmap_size is None or self.latest_source_frame_size is None:
            return None

        frame_width, frame_height = self.latest_source_frame_size
        combined_width, combined_height = self.latest_frame_size
        pixmap_width, pixmap_height = self.latest_pixmap_size
        x_offset = max(0, (self.video_label.width() - pixmap_width) // 2)
        y_offset = max(0, (self.video_label.height() - pixmap_height) // 2)

        if not (x_offset <= label_x < x_offset + pixmap_width and y_offset <= label_y < y_offset + pixmap_height):
            return None

        preview_x = (label_x - x_offset) * combined_width / pixmap_width
        preview_y = (label_y - y_offset) * combined_height / pixmap_height
        if preview_x < 0 or preview_y < 0 or preview_x >= combined_width or preview_y >= combined_height:
            return None

        image_x = max(0, min(frame_width - 1, int(preview_x % frame_width)))
        image_y = max(0, min(frame_height - 1, int(preview_y % frame_height)))
        return image_x, image_y

    def _handle_video_hovered(self, label_x, label_y):
        if self.latest_hsv_frame is None:
            self._clear_hover_display()
            return

        mapped_point = self._map_label_point_to_source(label_x, label_y)
        if mapped_point is None:
            self._clear_hover_display()
            return

        image_x, image_y = mapped_point
        h, s, v = self.latest_hsv_frame[image_y, image_x]
        hover_text = f"Hover HSV: ({int(h)}, {int(s)}, {int(v)}) at ({image_x}, {image_y})"
        self.hover_hsv_label.setText(hover_text)
        self.hover_hsv_popup.setText(f"HSV: ({int(h)}, {int(s)}, {int(v)})\n({image_x}, {image_y})")
        self.hover_hsv_popup.adjustSize()
        popup_x = min(label_x + 18, max(0, self.video_label.width() - self.hover_hsv_popup.width()))
        popup_y = min(label_y + 18, max(0, self.video_label.height() - self.hover_hsv_popup.height()))
        self.hover_hsv_popup.move(popup_x, popup_y)
        self.hover_hsv_popup.show()

    def _handle_video_left(self):
        self._clear_hover_display()

    def _clear_hover_display(self):
        self.hover_hsv_label.setText("Hover HSV: -")
        self.hover_hsv_popup.hide()

    def update_video_frame(self, bgr_frame, hsv_frame, source_frame_size=None):
        self.latest_hsv_frame = hsv_frame
        frame_height, frame_width = bgr_frame.shape[:2]
        self.latest_frame_size = (frame_width, frame_height)
        self.latest_source_frame_size = source_frame_size or (frame_width, frame_height)

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._last_rgb_frame = rgb_frame

        bytes_per_line = frame_width * 3
        image = QtGui.QImage(self._last_rgb_frame.data, frame_width, frame_height,
                             bytes_per_line, QtGui.QImage.Format.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        self.latest_pixmap_size = (scaled_pixmap.width(), scaled_pixmap.height())
        self.video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._last_rgb_frame is None or self.latest_frame_size is None:
            return

        frame_width, frame_height = self.latest_frame_size
        bytes_per_line = frame_width * 3
        image = QtGui.QImage(
            self._last_rgb_frame.data,
            frame_width,
            frame_height,
            bytes_per_line,
            QtGui.QImage.Format.Format_RGB888,
        )

        pixmap = QtGui.QPixmap.fromImage(image)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )
        self.latest_pixmap_size = (scaled_pixmap.width(), scaled_pixmap.height())
        self.video_label.setPixmap(scaled_pixmap)

    def get_active_picker_color(self):
        return self.active_picker_color

    def _set_hue_range(self, color_name, h_min, h_max):
        self.rows[color_name]["h_min"].set_value_from_undo(h_min)
        self.rows[color_name]["h_max"].set_value_from_undo(h_max)
        self.save_values()

    def set_hue_range_from_pick(self, color_name, hue_value):
        old_min = self.rows[color_name]["h_min"].value()
        old_max = self.rows[color_name]["h_max"].value()

        new_min = max(0, hue_value - 10)
        new_max = min(179, hue_value + 10)

        if old_min == new_min and old_max == new_max:
            return

        self.undo_stack.push(
            HuePickCommand(
                self,
                color_name,
                old_min,
                old_max,
                new_min,
                new_max,
            )
        )

    def get_values(self):
        config = {}
        for color_name, color_rows in self.rows.items():
            config[color_name] = {
                "min_vals": (
                    color_rows["h_min"].value(),
                    color_rows["s_min"].value(),
                    color_rows["v_min"].value(),
                ),
                "max_vals": (
                    color_rows["h_max"].value(),
                    color_rows["s_max"].value(),
                    color_rows["v_max"].value(),
                ),
                "invert": color_rows["invert"].isChecked(),
            }
        return config

    def save_values(self):
        pass

def main():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    cap = cv2.VideoCapture('green_cube_3.mp4')
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    controls_dialog = HSVControlsDialog(get_config())
    controls_dialog.show()

    kernel = np.ones((5, 5), np.uint8)

    timer = QtCore.QTimer()
    last_captured_frame = None

    def display_webcam_frame():
        nonlocal last_captured_frame

        if not controls_dialog.isVisible():
            timer.stop()
            return

        if not controls_dialog.play_button.isChecked() and last_captured_frame is not None:
            frame = last_captured_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from webcam.")
                timer.stop()
                cap.release()
                controls_dialog.close()
                return

            cv2.flip(frame, 1, frame)
            last_captured_frame = frame.copy()

        if last_captured_frame is None:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_colors = controls_dialog.get_values()
        raw_mask_preview = None
        open_mask_preview = None
        mask_preview = None

        for color_name, color_data in current_colors.items():
            mask = cv2.inRange(
                hsv,
                np.array(color_data["min_vals"], dtype=np.uint8),
                np.array(color_data["max_vals"], dtype=np.uint8),
            )
            if color_data.get("invert", False):
                mask = cv2.bitwise_not(mask)
            raw_mask_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            open_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            close_mask = cv2.morphologyEx(open_mask, cv2.MORPH_CLOSE, kernel)
            open_mask_preview = cv2.cvtColor(open_mask, cv2.COLOR_GRAY2BGR)

            mask = close_mask
            mask_preview = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            print('hierarchy: ', hierarchy)
            display_color = DISPLAY_COLOR

            for contour in contours:
                cv2.drawContours(frame, [contour], -1, display_color, 2)

                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    text_x = int(moments["m10"] / moments["m00"])
                    text_y = int(moments["m01"] / moments["m00"])
                    sample_x = text_x
                    sample_y = text_y
                else:
                    x, y, w, h = cv2.boundingRect(contour)
                    text_x = x
                    text_y = y - 10 if y > 20 else y + h + 20
                    sample_x = x + w // 2
                    sample_y = y + h // 2

                sample_x = max(0, min(sample_x, hsv.shape[1] - 1))
                sample_y = max(0, min(sample_y, hsv.shape[0] - 1))

                h, s, v = hsv[sample_y, sample_x]
                hsv_text = f"{color_name}: ({int(h)}, {int(s)}, {int(v)})"


        if raw_mask_preview is None:
            raw_mask_preview = np.zeros_like(frame)
        if open_mask_preview is None:
            open_mask_preview = np.zeros_like(frame)
        if mask_preview is None:
            mask_preview = np.zeros_like(frame)
        top_row = np.hstack([frame, raw_mask_preview])
        bottom_row = np.hstack([open_mask_preview, mask_preview])
        combined_preview = np.vstack([top_row, bottom_row])
        controls_dialog.update_video_frame(combined_preview, hsv, source_frame_size=(frame.shape[1], frame.shape[0]))

    timer.timeout.connect(display_webcam_frame)
    timer.start(30)

    app.exec()

    final_values = controls_dialog.get_values()[COLOR_NAME]
    save_config({COLOR_NAME: final_values})
    print(f"mask = cv2.inRange(hsv, np.array({list(final_values['min_vals'])}, dtype=np.uint8), np.array({list(final_values['max_vals'])}, dtype=np.uint8))")
    if final_values.get("invert", False):
        print("mask = cv2.bitwise_not(mask)")

    timer.stop()
    cap.release()
    controls_dialog.close()


if __name__ == "__main__":
    main()
