"""A small webcam demo that adds procedural fire and water to hand landmarks."""

import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
WINDOW_NAME = "FIRE & WATER AR"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


@dataclass
class Particle:
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    age: float
    lifetime: float
    size: int


class HandEffect:
    """Stores the short-lived particles for one independently tracked hand."""

    def __init__(self) -> None:
        self.mode = ""
        self.particles: list[Particle] = []
        self.random = random.Random()

    def select_gesture(self, fingers: int) -> None:
        if fingers == 0:
            return
        mode = "fire" if fingers == 1 else "water"
        if mode != self.mode:
            self.particles.clear()
            self.mode = mode

    def update_and_draw(
        self,
        overlay: np.ndarray,
        anchors: list[tuple[int, int]],
        palm: tuple[int, int],
        mode: str,
        delta_time: float,
        elapsed: float,
    ) -> None:
        if self.mode != mode:
            self.particles.clear()
            self.mode = mode

        for _ in range(4):
            anchor_x, anchor_y = self.random.choice(anchors)
            if mode == "fire":
                velocity_x = self.random.uniform(-24.0, 24.0)
                velocity_y = self.random.uniform(-100.0, -42.0)
                lifetime = self.random.uniform(0.55, 1.05)
            else:
                velocity_x = self.random.uniform(-20.0, 20.0)
                velocity_y = self.random.uniform(35.0, 100.0)
                lifetime = self.random.uniform(0.65, 1.25)
            self.particles.append(
                Particle(
                    x=anchor_x + self.random.uniform(-10.0, 10.0),
                    y=anchor_y + self.random.uniform(-10.0, 10.0),
                    velocity_x=velocity_x,
                    velocity_y=velocity_y,
                    age=0.0,
                    lifetime=lifetime,
                    size=self.random.randint(2, 6),
                )
            )

        for particle in self.particles:
            particle.age += delta_time
            particle.x += particle.velocity_x * delta_time
            particle.y += particle.velocity_y * delta_time

            life_left = max(0.0, 1.0 - particle.age / particle.lifetime)
            size = max(1, int(particle.size * life_left))
            point = (int(particle.x), int(particle.y))
            if mode == "fire":
                glow = (0, int(90 + 100 * life_left), 255)
                core = (0, int(150 + 100 * life_left), 255)
                cv2.circle(overlay, point, size * 3, glow, -1, cv2.LINE_AA)
                cv2.circle(overlay, point, size, core, -1, cv2.LINE_AA)
            else:
                glow = (int(180 * life_left), int(115 + 100 * life_left), 30)
                core = (255, 235, 145)
                cv2.ellipse(
                    overlay,
                    point,
                    (size, size * 2),
                    0,
                    0,
                    360,
                    glow,
                    -1,
                    cv2.LINE_AA,
                )
                cv2.circle(overlay, point, max(1, size // 2), core, -1, cv2.LINE_AA)

        self.particles = [
            particle for particle in self.particles if particle.age < particle.lifetime
        ]

        palm_x, palm_y = palm
        if mode == "fire":
            flicker = math.sin(elapsed * 12.0 + palm_x) * 4.0
            for radius, color in ((29, (0, 95, 255)), (19, (0, 170, 255))):
                cv2.ellipse(
                    overlay,
                    (palm_x, int(palm_y + flicker)),
                    (radius, radius + 8),
                    0,
                    0,
                    360,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        else:
            wave = int(25 + 5 * math.sin(elapsed * 5.0 + palm_x))
            cv2.ellipse(
                overlay,
                palm,
                (wave + 15, wave),
                0,
                0,
                360,
                (255, 180, 35),
                2,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                overlay,
                palm,
                (wave + 25, wave + 8),
                0,
                20,
                160,
                (255, 245, 100),
                2,
                cv2.LINE_AA,
            )
            splash_height = int(10 + 5 * math.sin(elapsed * 8.0 + palm_x))
            for direction in (-1, 1):
                start = (palm_x + direction * 13, palm_y)
                end = (palm_x + direction * 27, palm_y - splash_height)
                cv2.line(overlay, start, end, (255, 235, 120), 2, cv2.LINE_AA)


def count_extended_fingers(landmarks: list) -> int:
    """Estimate the number of raised fingers using the hand's landmark positions."""
    count = 0
    for tip_index, joint_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
        if landmarks[tip_index].y < landmarks[joint_index].y:
            count += 1

    thumb_base = landmarks[2]
    thumb_middle = landmarks[3]
    thumb_tip = landmarks[4]
    base_to_tip = math.hypot(
        thumb_tip.x - thumb_base.x, thumb_tip.y - thumb_base.y
    )
    base_to_middle = math.hypot(
        thumb_middle.x - thumb_base.x, thumb_middle.y - thumb_base.y
    )
    if base_to_tip > base_to_middle * 1.35:
        count += 1
    return count


def draw_hand_outline(
    image: np.ndarray, landmarks: list, color: tuple[int, int, int]
) -> list[tuple[int, int]]:
    height, width = image.shape[:2]
    points = [
        (int(point.x * width), int(point.y * height)) for point in landmarks
    ]
    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )
    for start, end in connections:
        cv2.line(image, points[start], points[end], color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 3, (235, 245, 255), -1, cv2.LINE_AA)
    return points


def draw_hud(
    image: np.ndarray,
    hand_status: list[tuple[str, int, str]],
    fps: float,
) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 104), (12, 12, 20), -1)
    cv2.putText(
        image, "FIRE & WATER AR", (18, 34),
        cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 245, 225), 2, cv2.LINE_AA,
    )
    cv2.putText(
        image, "Q: quit", (image.shape[1] - 112, 32),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (185, 195, 210), 1, cv2.LINE_AA,
    )
    if hand_status:
        for index, (label, fingers, mode) in enumerate(hand_status):
            text = f"{label}: {fingers} finger(s) - {mode}"
            color = (80, 175, 255) if mode == "FIRE" else (255, 220, 100)
            if mode == "NO EFFECT":
                color = (175, 185, 200)
            cv2.putText(
                image, text, (18, 62 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA,
            )
    else:
        cv2.putText(
            image, "Show one or both hands to the camera",
            (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (185, 195, 210), 1, cv2.LINE_AA,
        )
    cv2.putText(
        image, f"FPS: {fps:.0f}", (image.shape[1] - 112, 82),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (205, 220, 230), 1, cv2.LINE_AA,
    )


def create_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_camera_scene(
    frame: np.ndarray,
    rgb_frame: np.ndarray,
    background_active: bool,
    segmenter: mp.solutions.selfie_segmentation.SelfieSegmentation,
) -> np.ndarray:
    if not background_active:
        return frame.copy()

    segmentation = segmenter.process(rgb_frame)
    mask = np.where(segmentation.segmentation_mask > 0.45, segmentation.segmentation_mask, 0)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (11, 11), 0)
    return (frame.astype(np.float32) * mask[:, :, np.newaxis]).astype(np.uint8)


def main() -> int:
    if not MODEL_PATH.is_file():
        print(
            f"Hand landmark model not found: {MODEL_PATH}\n"
            "Run setup.bat to install dependencies and download the model.",
            file=sys.stderr,
        )
        return 1

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        print(
            "Could not open webcam. Check that it is connected and not in use.",
            file=sys.stderr,
        )
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    effects: dict[str, HandEffect] = {}
    previous_frame_time = time.perf_counter()
    start_time = time.monotonic()
    previous_timestamp_ms = -1
    background_active = False

    try:
        with (
            create_landmarker() as landmarker,
            mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=0
            ) as segmenter,
        ):
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.", file=sys.stderr)
                    return 1
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                height, width = frame.shape[:2]
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=rgb_frame
                )
                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                now = time.perf_counter()
                delta_time = min(now - previous_frame_time, 0.1)
                previous_frame_time = now
                elapsed = time.monotonic() - start_time
                effect_overlay = np.zeros_like(frame)
                hand_status: list[tuple[str, int, str]] = []
                detected_hands: list[
                    tuple[str, int, list, list[tuple[int, int]], str]
                ] = []

                if result.hand_landmarks:
                    for index, landmarks in enumerate(result.hand_landmarks):
                        handedness = "Hand"
                        if index < len(result.handedness) and result.handedness[index]:
                            handedness = result.handedness[index][0].category_name
                        fingers = count_extended_fingers(landmarks)
                        effect = effects.get(handedness)
                        if fingers > 0:
                            effect = effects.setdefault(handedness, HandEffect())
                            effect.select_gesture(fingers)

                        mode = effect.mode if effect else ""
                        if mode:
                            background_active = True
                        display_mode = mode.upper() if mode else "NO EFFECT"
                        if fingers == 0 and mode:
                            display_mode += " (HELD)"

                        outline_color = (
                            (40, 145, 255) if mode == "fire"
                            else (255, 205, 65) if mode == "water"
                            else (150, 170, 185)
                        )
                        points = [
                            (int(point.x * width), int(point.y * height))
                            for point in landmarks
                        ]
                        detected_hands.append(
                            (handedness, fingers, landmarks, points, mode)
                        )
                        hand_status.append((handedness, fingers, display_mode))

                scene = create_camera_scene(
                    frame, rgb_frame, background_active, segmenter
                )

                for handedness, _, landmarks, points, mode in detected_hands:
                    outline_color = (
                        (40, 145, 255) if mode == "fire"
                        else (255, 205, 65) if mode == "water"
                        else (150, 170, 185)
                    )
                    draw_hand_outline(scene, landmarks, outline_color)
                    if mode:
                        anchors = [
                            points[point_index]
                            for point_index in (0, 5, 9, 13, 17, 8, 12, 16, 20)
                        ]
                        palm = (
                            int(sum(points[i][0] for i in (0, 5, 9, 13, 17)) / 5),
                            int(sum(points[i][1] for i in (0, 5, 9, 13, 17)) / 5),
                        )
                        effects[handedness].update_and_draw(
                            effect_overlay, anchors, palm, mode, delta_time, elapsed
                        )

                scene = cv2.addWeighted(scene, 1.0, effect_overlay, 0.78, 0)
                fps = 1.0 / max(delta_time, 0.001)
                draw_hud(scene, hand_status, fps)
                cv2.imshow(WINDOW_NAME, scene)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
