import json
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from PySide6 import QtCore
from PySide6 import QtGui
from PySide6 import QtWidgets

COLOR_RANGES_PATH = Path(__file__).with_name("color_ranges.json")
DICE_BODY_CONTOUR_COLOR_NAME = "dice_body_contour"
DICE_FACE_COLOR_NAME = "dice_face_color"
DEFAULT_COLOR_CONFIG = {
    DICE_BODY_CONTOUR_COLOR_NAME: {
        "min_vals": (26, 59, 30),
        "max_vals": (98, 255, 250),
        "invert": False,
    },
    DICE_FACE_COLOR_NAME: {
        "min_vals": (62, 37, 92),
        "max_vals": (89, 255, 249),
        "invert": False,
    },
}
DISPLAY_COLORS = {
    DICE_BODY_CONTOUR_COLOR_NAME: (255, 255, 255),
    DICE_FACE_COLOR_NAME: (0, 255, 0),
}


@dataclass
class AppConfig:
    video_source: str | int = "green_dice.mp4"
    start_frame: int = 6
    start_paused: bool = True
    flip_frame_horizontal: bool = False


def get_default_config():
    color_config = {
        color_name: {
            "min_vals": tuple(color_data["min_vals"]),
            "max_vals": tuple(color_data["max_vals"]),
            "invert": bool(color_data.get("invert", False)),
        }
        for color_name, color_data in DEFAULT_COLOR_CONFIG.items()
    }

    if not COLOR_RANGES_PATH.exists():
        return color_config

    try:
        with COLOR_RANGES_PATH.open("r", encoding="utf-8") as config_file:
            saved_config = json.load(config_file)
    except (OSError, json.JSONDecodeError):
        return color_config

    for color_name, color_data in saved_config.items():
        if color_name not in color_config:
            continue
        if "min_vals" in color_data:
            color_config[color_name]["min_vals"] = tuple(color_data["min_vals"])
        if "max_vals" in color_data:
            color_config[color_name]["max_vals"] = tuple(color_data["max_vals"])
        if "invert" in color_data:
            color_config[color_name]["invert"] = bool(color_data["invert"])

    return color_config


def save_color_config(color_config):
    serializable_config = {
        color_name: {
            "min_vals": list(color_data["min_vals"]),
            "max_vals": list(color_data["max_vals"]),
            "invert": bool(color_data.get("invert", False)),
        }
        for color_name, color_data in color_config.items()
    }
    with COLOR_RANGES_PATH.open("w", encoding="utf-8") as config_file:
        json.dump(serializable_config, config_file, indent=2)
        config_file.write("\n")


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
    def __init__(self, label_text, minimum, maximum, value, undo_stack):
        super().__init__()

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
    def __init__(self, color_config, app_config):
        super().__init__()
        self.setWindowTitle("Calibrate Dice Colors")
        self.resize(1500, 950)

        self.undo_stack = QtGui.QUndoStack(self)
        self.rows = {}
        self.pick_buttons = {}
        self.video_labels = {}
        self.hover_hsv_labels = {}
        self.hover_hsv_popups = {}
        self.latest_frame_sizes = {}
        self.latest_source_frame_sizes = {}
        self.latest_pixmap_sizes = {}
        self._last_rgb_frames = {}
        self.active_picker_color = None
        self._updating_pick_buttons = False
        self.latest_hsv_frame = None

        root_layout = QtWidgets.QVBoxLayout(self)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.setCheckable(True)
        self.play_button.setChecked(not app_config.start_paused)
        root_layout.addWidget(self.play_button)

        sections = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        root_layout.addWidget(sections, stretch=1)

        for color_name, color_data in color_config.items():
            sections.addWidget(self._create_range_section(color_name, color_data))

        self.undo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Undo, self)
        self.undo_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.undo_shortcut.activated.connect(self.undo_stack.undo)

        self.redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence.StandardKey.Redo, self)
        self.redo_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.redo_shortcut.activated.connect(self.undo_stack.redo)

        self.quit_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Q"), self)
        self.quit_shortcut.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        self.quit_shortcut.activated.connect(self.close)

    def _create_range_section(self, color_name, color_data):
        section = QtWidgets.QGroupBox(color_name)
        section_layout = QtWidgets.QHBoxLayout(section)

        controls_container = QtWidgets.QWidget()
        controls_container.setMinimumWidth(360)
        controls_container.setMaximumWidth(460)
        controls_layout = QtWidgets.QVBoxLayout(controls_container)

        hover_hsv_label = QtWidgets.QLabel("Hover HSV: -")
        self.hover_hsv_labels[color_name] = hover_hsv_label
        controls_layout.addWidget(hover_hsv_label)

        form_group = QtWidgets.QGroupBox("HSV Range")
        form_layout = QtWidgets.QFormLayout(form_group)
        min_vals = color_data["min_vals"]
        max_vals = color_data["max_vals"]

        color_rows = {
            "h_min": SliderRow("H min", 0, 179, min_vals[0], self.undo_stack),
            "h_max": SliderRow("H max", 0, 179, max_vals[0], self.undo_stack),
            "s_min": SliderRow("S min", 0, 255, min_vals[1], self.undo_stack),
            "s_max": SliderRow("S max", 0, 255, max_vals[1], self.undo_stack),
            "v_min": SliderRow("V min", 0, 255, min_vals[2], self.undo_stack),
            "v_max": SliderRow("V max", 0, 255, max_vals[2], self.undo_stack),
        }

        self.rows[color_name] = color_rows
        for row in color_rows.values():
            form_layout.addRow(row)

        pick_button = QtWidgets.QPushButton("Pick Hue")
        pick_button.setCheckable(True)
        pick_button.toggled.connect(partial(self._handle_pick_button_toggled, color_name))
        self.pick_buttons[color_name] = pick_button
        form_layout.addRow("Picker", pick_button)

        invert_checkbox = QtWidgets.QCheckBox("Invert")
        invert_checkbox.setChecked(bool(color_data.get("invert", False)))
        color_rows["invert"] = invert_checkbox
        form_layout.addRow("Mask", invert_checkbox)

        controls_layout.addWidget(form_group)
        controls_layout.addStretch()

        video_label = VideoLabel()
        video_label.clicked.connect(partial(self._handle_video_clicked, color_name))
        video_label.hovered.connect(partial(self._handle_video_hovered, color_name))
        video_label.left.connect(partial(self._handle_video_left, color_name))
        self.video_labels[color_name] = video_label

        hover_popup = QtWidgets.QLabel(video_label)
        hover_popup.setStyleSheet("background-color: rgba(0, 0, 0, 190); color: white; border: 1px solid #808080; padding: 4px;")
        hover_popup.hide()
        self.hover_hsv_popups[color_name] = hover_popup

        section_layout.addWidget(controls_container, stretch=0)
        section_layout.addWidget(video_label, stretch=1)
        return section

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

    def _handle_video_clicked(self, color_name, label_x, label_y):
        active_color = self.get_active_picker_color() or color_name
        if self.latest_hsv_frame is None:
            return

        mapped_point = self._map_label_point_to_source(color_name, label_x, label_y)
        if mapped_point is None:
            return

        image_x, image_y = mapped_point
        picked_hsv = self.latest_hsv_frame[image_y, image_x]
        picked_hue = int(picked_hsv[0])
        self.set_hue_range_from_pick(active_color, picked_hue)

    def _map_label_point_to_source(self, color_name, label_x, label_y):
        if (
            color_name not in self.latest_frame_sizes
            or color_name not in self.latest_pixmap_sizes
            or color_name not in self.latest_source_frame_sizes
        ):
            return None

        video_label = self.video_labels[color_name]
        frame_width, frame_height = self.latest_source_frame_sizes[color_name]
        combined_width, combined_height = self.latest_frame_sizes[color_name]
        pixmap_width, pixmap_height = self.latest_pixmap_sizes[color_name]
        x_offset = max(0, (video_label.width() - pixmap_width) // 2)
        y_offset = max(0, (video_label.height() - pixmap_height) // 2)

        if not (x_offset <= label_x < x_offset + pixmap_width and y_offset <= label_y < y_offset + pixmap_height):
            return None

        preview_x = (label_x - x_offset) * combined_width / pixmap_width
        preview_y = (label_y - y_offset) * combined_height / pixmap_height
        if preview_x < 0 or preview_y < 0 or preview_x >= frame_width or preview_y >= frame_height:
            return None

        image_x = max(0, min(frame_width - 1, int(preview_x)))
        image_y = max(0, min(frame_height - 1, int(preview_y)))
        return image_x, image_y

    def _handle_video_hovered(self, color_name, label_x, label_y):
        if self.latest_hsv_frame is None:
            self._clear_hover_display(color_name)
            return

        mapped_point = self._map_label_point_to_source(color_name, label_x, label_y)
        if mapped_point is None:
            self._clear_hover_display(color_name)
            return

        image_x, image_y = mapped_point
        h, s, v = self.latest_hsv_frame[image_y, image_x]
        hover_text = f"Hover HSV: ({int(h)}, {int(s)}, {int(v)}) at ({image_x}, {image_y})"
        self.hover_hsv_labels[color_name].setText(hover_text)
        hover_popup = self.hover_hsv_popups[color_name]
        video_label = self.video_labels[color_name]
        hover_popup.setText(f"HSV: ({int(h)}, {int(s)}, {int(v)})\n({image_x}, {image_y})")
        hover_popup.adjustSize()
        popup_x = min(label_x + 18, max(0, video_label.width() - hover_popup.width()))
        popup_y = min(label_y + 18, max(0, video_label.height() - hover_popup.height()))
        hover_popup.move(popup_x, popup_y)
        hover_popup.show()

    def _handle_video_left(self, color_name):
        self._clear_hover_display(color_name)

    def _clear_hover_display(self, color_name):
        self.hover_hsv_labels[color_name].setText("Hover HSV: -")
        self.hover_hsv_popups[color_name].hide()

    def update_video_frame(self, color_name, bgr_frame, hsv_frame, source_frame_size=None):
        self.latest_hsv_frame = hsv_frame
        frame_height, frame_width = bgr_frame.shape[:2]
        self.latest_frame_sizes[color_name] = (frame_width, frame_height)
        self.latest_source_frame_sizes[color_name] = source_frame_size or (frame_width, frame_height)

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        self._last_rgb_frames[color_name] = rgb_frame

        bytes_per_line = frame_width * 3
        image = QtGui.QImage(self._last_rgb_frames[color_name].data, frame_width, frame_height,
                             bytes_per_line, QtGui.QImage.Format.Format_RGB888)

        pixmap = QtGui.QPixmap.fromImage(image)
        video_label = self.video_labels[color_name]
        scaled_pixmap = pixmap.scaled(
            video_label.size(),
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation,
        )

        self.latest_pixmap_sizes[color_name] = (scaled_pixmap.width(), scaled_pixmap.height())
        video_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for color_name, rgb_frame in self._last_rgb_frames.items():
            frame_width, frame_height = self.latest_frame_sizes[color_name]
            bytes_per_line = frame_width * 3
            image = QtGui.QImage(
                rgb_frame.data,
                frame_width,
                frame_height,
                bytes_per_line,
                QtGui.QImage.Format.Format_RGB888,
            )

            pixmap = QtGui.QPixmap.fromImage(image)
            video_label = self.video_labels[color_name]
            scaled_pixmap = pixmap.scaled(
                video_label.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            )
            self.latest_pixmap_sizes[color_name] = (scaled_pixmap.width(), scaled_pixmap.height())
            video_label.setPixmap(scaled_pixmap)

    def get_active_picker_color(self):
        return self.active_picker_color

    def _set_hue_range(self, color_name, h_min, h_max):
        self.rows[color_name]["h_min"].set_value_from_undo(h_min)
        self.rows[color_name]["h_max"].set_value_from_undo(h_max)

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

def main(config: AppConfig | None = None):
    if config is None:
        config = AppConfig()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    cap = cv2.VideoCapture(config.video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {config.video_source}.")
        return

    if config.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, config.start_frame)

    controls_dialog = HSVControlsDialog(get_default_config(), config)
    controls_dialog.show()

    kernel = np.ones((5, 5), np.uint8)

    timer = QtCore.QTimer()
    last_captured_frame = None

    def display_video_frame():
        nonlocal last_captured_frame

        if not controls_dialog.isVisible():
            timer.stop()
            return

        if not controls_dialog.play_button.isChecked() and last_captured_frame is not None:
            frame = last_captured_frame.copy()
        else:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to read frame from video source.")
                timer.stop()
                cap.release()
                controls_dialog.close()
                return

            if config.flip_frame_horizontal:
                cv2.flip(frame, 1, frame)
            last_captured_frame = frame.copy()

        if last_captured_frame is None:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_colors = controls_dialog.get_values()

        for color_name, color_data in current_colors.items():
            contour_preview = frame.copy()
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

            contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            display_color = DISPLAY_COLORS.get(color_name, (0, 255, 0))

            for contour in contours:
                cv2.drawContours(contour_preview, [contour], -1, display_color, 2)

            top_row = np.hstack([contour_preview, raw_mask_preview])
            bottom_row = np.hstack([open_mask_preview, mask_preview])
            range_preview = np.vstack([top_row, bottom_row])
            controls_dialog.update_video_frame(
                color_name,
                range_preview,
                hsv,
                source_frame_size=(frame.shape[1], frame.shape[0]),
            )

    timer.timeout.connect(display_video_frame)
    timer.start(30)

    app.exec()

    final_values = controls_dialog.get_values()
    save_color_config(final_values)
    print(f"Saved HSV ranges to {COLOR_RANGES_PATH}")
    for color_name, color_data in final_values.items():
        print(f"{color_name}_min: tuple[int, int, int] = {tuple(color_data['min_vals'])}")
        print(f"{color_name}_max: tuple[int, int, int] = {tuple(color_data['max_vals'])}")

    timer.stop()
    cap.release()
    controls_dialog.close()


if __name__ == "__main__":
    main()
