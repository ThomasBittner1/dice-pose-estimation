from enum import Enum


class TrackingState(Enum):
    STABLE = 0
    MOVING = 1


class StabilityTracker:
    def __init__(self, threshold=0.95, required_frames=5):
        self.threshold = threshold
        self.required_frames = required_frames
        self.state = TrackingState.MOVING
        self.stable_count = 0
        self.moving_count = 0

    def update(self, similarity_score):
        if similarity_score >= self.threshold:
            self.stable_count += 1
            self.moving_count = 0
        else:
            self.moving_count += 1
            self.stable_count = 0

        if self.stable_count > self.required_frames:
            self.state = TrackingState.STABLE
        elif self.moving_count > self.required_frames:
            self.state = TrackingState.MOVING

        return self.state

    @property
    def is_stable(self):
        return self.state == TrackingState.STABLE

    @property
    def is_moving(self):
        return self.state == TrackingState.MOVING
