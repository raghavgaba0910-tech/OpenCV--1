"""FIRE & WATER AR - Ultra-Realistic Hand VFX
Real-time augmented reality fire and water elemental manipulation with MediaPipe and OpenCV.
"""

import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
WINDOW_NAME = "FIRE & WATER AR (Cinematic VFX)"
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


# =============================================================================
# PARTICLE DATA STRUCTURES & DEFINITIONS
# =============================================================================

@dataclass
class FireParticle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    lifetime: float
    size: float
    p_type: str  # "flame", "spark", "smoke"
    flicker_phase: float = 0.0


@dataclass
class WaterParticle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    lifetime: float
    size: float
    p_type: str  # "droplet", "splash", "mist", "bubble"
    stretch: float = 1.0


@dataclass
class FusionParticle:
    x: float
    y: float
    vx: float
    vy: float
    age: float
    lifetime: float
    size: float
    p_type: str  # "steam", "plasma", "ember", "splash"
    color: tuple[int, int, int] = (255, 255, 255)


# =============================================================================
# SMOOTHING & ACCURATE GESTURE DETECTION
# =============================================================================

class LandmarkFilter:
    """Exponential Moving Average (EMA) filter to eliminate webcam jitter."""

    def __init__(self, alpha: float = 0.68) -> None:
        self.alpha = alpha
        self.smoothed_points: Optional[np.ndarray] = None
        self.prev_time: float = time.monotonic()
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float32)

    def update(self, raw_points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        current_pts = np.array(raw_points, dtype=np.float32)
        now = time.monotonic()
        dt = max(now - self.prev_time, 0.001)
        self.prev_time = now

        if self.smoothed_points is None:
            self.smoothed_points = current_pts
            self.velocity = np.zeros(2, dtype=np.float32)
        else:
            prev_center = self.smoothed_points[9].copy()
            self.smoothed_points = (
                self.alpha * current_pts + (1.0 - self.alpha) * self.smoothed_points
            )
            curr_center = self.smoothed_points[9]
            instant_v = (curr_center - prev_center) / dt
            self.velocity = 0.7 * self.velocity + 0.3 * instant_v

        return [(int(p[0]), int(p[1])) for p in self.smoothed_points]


def count_extended_fingers(landmarks: list) -> int:
    """
    Robust 3D vector-based finger counting invariant to hand rotation, pitch, and roll.
    Works whether the hand is vertical, sideways, upside-down, or pointing toward camera.
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    p9 = np.array([landmarks[9].x, landmarks[9].y, landmarks[9].z])
    palm_scale = np.linalg.norm(p9 - wrist)
    if palm_scale < 1e-4:
        palm_scale = 1.0

    count = 0
    # 4 Fingers: Index (5..8), Middle (9..12), Ring (13..16), Pinky (17..20)
    finger_joints = [
        (5, 6, 8),    # Index: MCP, PIP, TIP
        (9, 10, 12),  # Middle: MCP, PIP, TIP
        (13, 14, 16), # Ring: MCP, PIP, TIP
        (17, 18, 20), # Pinky: MCP, PIP, TIP
    ]

    for mcp_idx, pip_idx, tip_idx in finger_joints:
        p_mcp = np.array([landmarks[mcp_idx].x, landmarks[mcp_idx].y, landmarks[mcp_idx].z])
        p_pip = np.array([landmarks[pip_idx].x, landmarks[pip_idx].y, landmarks[pip_idx].z])
        p_tip = np.array([landmarks[tip_idx].x, landmarks[tip_idx].y, landmarks[tip_idx].z])

        # Distance check relative to wrist and MCP
        d_tip_wrist = np.linalg.norm(p_tip - wrist)
        d_pip_wrist = np.linalg.norm(p_pip - wrist)
        d_tip_mcp = np.linalg.norm(p_tip - p_mcp)
        d_pip_mcp = np.linalg.norm(p_pip - p_mcp)

        # Vector alignment: MCP -> PIP vs PIP -> TIP
        v_base = p_pip - p_mcp
        v_ext = p_tip - p_pip
        norm_prod = np.linalg.norm(v_base) * np.linalg.norm(v_ext) + 1e-6
        cos_angle = np.dot(v_base, v_ext) / norm_prod

        if (d_tip_wrist > d_pip_wrist * 1.08 and d_tip_mcp > d_pip_mcp * 1.25) or cos_angle > 0.72:
            count += 1

    # Thumb: (1: CMC, 2: MCP, 3: IP, 4: TIP)
    p_thumb_tip = np.array([landmarks[4].x, landmarks[4].y, landmarks[4].z])
    p_thumb_mcp = np.array([landmarks[2].x, landmarks[2].y, landmarks[2].z])
    p_pinky_mcp = np.array([landmarks[17].x, landmarks[17].y, landmarks[17].z])
    p_index_mcp = np.array([landmarks[5].x, landmarks[5].y, landmarks[5].z])

    d_thumb_pinky = np.linalg.norm(p_thumb_tip - p_pinky_mcp)
    d_mcp_pinky = np.linalg.norm(p_thumb_mcp - p_pinky_mcp)
    d_thumb_index = np.linalg.norm(p_thumb_tip - p_index_mcp)

    if d_thumb_pinky > d_mcp_pinky * 1.22 and d_thumb_index > palm_scale * 0.42:
        count += 1

    return count


# =============================================================================
# ANATOMICAL HAND MASK GENERATION (NO MITTEN EFFECT, FULL COVERAGE)
# =============================================================================

def build_hand_masks(
    points: list[tuple[int, int]], shape: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Builds precision anatomical masks:
    - full_mask: Covers the entire palm and each individual finger with biologically accurate tapering
    - core_mask: High-energy inner core channel along the skeletal bones
    - palm_center: 2D coordinates of the biomechanical palm center
    - palm_scale: Pixel scale of the hand
    """
    height, width = shape
    full_mask = np.zeros((height, width), dtype=np.uint8)
    core_mask = np.zeros((height, width), dtype=np.uint8)

    p0 = np.array(points[0], dtype=np.float32)
    p9 = np.array(points[9], dtype=np.float32)
    palm_scale = float(np.linalg.norm(p9 - p0))
    if palm_scale < 15.0:
        palm_scale = 55.0

    # 1. Anatomical palm polygon
    palm_indices = [0, 1, 2, 5, 9, 13, 17, 0]
    palm_poly = np.array([points[i] for i in palm_indices], dtype=np.int32)
    cv2.fillPoly(full_mask, [palm_poly], 255)
    cv2.fillPoly(core_mask, [palm_poly], 255)

    # Cross-struts to ensure solid palm core
    struts = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17), (2, 5)]
    palm_thick = max(6, int(palm_scale * 0.38))
    for a, b in struts:
        cv2.line(full_mask, points[a], points[b], 255, palm_thick)
        cv2.line(core_mask, points[a], points[b], 255, max(2, palm_thick // 2))

    # 2. Individual finger segments with realistic anatomical tapering
    finger_defs = [
        ([(1, 2), (2, 3), (3, 4)], 0.23, 4),     # Thumb
        ([(5, 6), (6, 7), (7, 8)], 0.20, 8),     # Index
        ([(9, 10), (10, 11), (11, 12)], 0.21, 12),# Middle
        ([(13, 14), (14, 15), (15, 16)], 0.19, 16),# Ring
        ([(17, 18), (18, 19), (19, 20)], 0.17, 20),# Pinky
    ]

    for segments, thick_factor, tip_idx in finger_defs:
        base_t = max(5, int(palm_scale * thick_factor))
        for step, (a, b) in enumerate(segments):
            # Taper toward fingertip
            t = max(3, int(base_t * (1.0 - step * 0.16)))
            c_t = max(2, t // 2)
            cv2.line(full_mask, points[a], points[b], 255, t)
            cv2.circle(full_mask, points[b], t // 2, 255, -1)
            cv2.line(core_mask, points[a], points[b], 255, c_t)
            cv2.circle(core_mask, points[b], c_t // 2, 255, -1)

        tip_r = max(2, int(base_t * 0.38))
        cv2.circle(full_mask, points[tip_idx], tip_r, 255, -1)
        cv2.circle(core_mask, points[tip_idx], max(1, tip_r // 2), 255, -1)

    # Palm center position
    palm_x = int(sum(points[i][0] for i in (0, 5, 9, 13, 17)) / 5)
    palm_y = int(sum(points[i][1] for i in (0, 5, 9, 13, 17)) / 5)

    return full_mask, core_mask, (palm_x, palm_y), palm_scale


# =============================================================================
# REALISTIC ELEMENTAL EFFECT SIMULATOR
# =============================================================================

class HandEffect:
    """Simulates realistic Fire and Water elemental magic on a tracked hand."""

    def __init__(self) -> None:
        self.mode = ""
        self.fire_particles: list[FireParticle] = []
        self.water_particles: list[WaterParticle] = []
        self.random = random.Random()
        self.filter = LandmarkFilter()

    def select_gesture(self, fingers: int) -> None:
        if fingers == 0:
            return  # Fist holds current active effect
        new_mode = "fire" if fingers == 1 else "water"
        if new_mode != self.mode:
            self.fire_particles.clear()
            self.water_particles.clear()
            self.mode = new_mode

    # -------------------------------------------------------------------------
    # FIRE SIMULATION
    # -------------------------------------------------------------------------
    def _update_fire_particles(
        self,
        anchors: list[tuple[int, int]],
        palm: tuple[int, int],
        hand_velocity: np.ndarray,
        delta_time: float,
        palm_scale: float,
    ) -> None:
        # Spawn flame tongues, sparks, and smoke
        spawn_count = int(14 * (palm_scale / 50.0))
        spawn_count = max(8, min(spawn_count, 22))

        wind_x = -hand_velocity[0] * 0.18
        wind_y = -hand_velocity[1] * 0.18

        for _ in range(spawn_count):
            anchor_x, anchor_y = self.random.choice(anchors)
            roll = self.random.random()

            if roll < 0.65:
                # Rising flame tongue
                vx = self.random.uniform(-25.0, 25.0) + wind_x
                vy = self.random.uniform(-180.0, -90.0) + wind_y
                lifetime = self.random.uniform(0.35, 0.70)
                size = self.random.uniform(palm_scale * 0.12, palm_scale * 0.28)
                self.fire_particles.append(
                    FireParticle(
                        x=anchor_x + self.random.uniform(-6, 6),
                        y=anchor_y + self.random.uniform(-6, 6),
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=lifetime,
                        size=size,
                        p_type="flame",
                        flicker_phase=self.random.uniform(0, math.pi * 2),
                    )
                )
            elif roll < 0.90:
                # High-speed spark/ember
                vx = self.random.uniform(-70.0, 70.0) + wind_x * 1.5
                vy = self.random.uniform(-260.0, -110.0) + wind_y * 1.2
                lifetime = self.random.uniform(0.40, 0.95)
                size = self.random.uniform(2.0, 5.0)
                self.fire_particles.append(
                    FireParticle(
                        x=anchor_x + self.random.uniform(-4, 4),
                        y=anchor_y + self.random.uniform(-4, 4),
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=lifetime,
                        size=size,
                        p_type="spark",
                    )
                )
            else:
                # Dark smoke puff
                vx = self.random.uniform(-20.0, 20.0) + wind_x * 0.5
                vy = self.random.uniform(-80.0, -40.0)
                lifetime = self.random.uniform(0.60, 1.1)
                size = self.random.uniform(palm_scale * 0.20, palm_scale * 0.45)
                self.fire_particles.append(
                    FireParticle(
                        x=anchor_x + self.random.uniform(-8, 8),
                        y=anchor_y - palm_scale * 0.25,
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=lifetime,
                        size=size,
                        p_type="smoke",
                    )
                )

        # Update physics
        alive: list[FireParticle] = []
        for p in self.fire_particles:
            p.age += delta_time
            if p.age >= p.lifetime:
                continue

            if p.p_type == "flame":
                # Thermal upward buoyancy + lateral sine flicker
                p.vy -= 220.0 * delta_time
                p.vx += math.sin(p.flicker_phase + p.age * 18.0) * 8.0
            elif p.p_type == "spark":
                p.vy -= 120.0 * delta_time
                p.vx += math.sin(p.age * 25.0) * 14.0
            else:
                # Smoke drifts slowly
                p.vy -= 40.0 * delta_time

            p.x += p.vx * delta_time
            p.y += p.vy * delta_time
            alive.append(p)

        self.fire_particles = alive

    def _draw_fire(
        self,
        fx_overlay: np.ndarray,
        hand_mask: np.ndarray,
        core_mask: np.ndarray,
        anchors: list[tuple[int, int]],
        palm: tuple[int, int],
        elapsed: float,
        palm_scale: float,
    ) -> None:
        h, w = fx_overlay.shape[:2]

        # 1. Living Molten Fire Skin on Hand
        # Generate dynamic turbulent plasma noise across hand bounding box
        ys, xs = np.where(hand_mask > 0)
        if len(xs) > 0:
            min_x, max_x = max(0, xs.min()), min(w - 1, xs.max())
            min_y, max_y = max(0, ys.min()), min(h - 1, ys.max())
            sub_mask = hand_mask[min_y : max_y + 1, min_x : max_x + 1]

            sub_h, sub_w = sub_mask.shape
            grid_y, grid_x = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]

            # Harmonic fire noise
            wave = np.sin(grid_x * 0.045 + elapsed * 9.0) * np.cos(
                grid_y * 0.055 - elapsed * 8.0
            ) + 0.5 * np.sin((grid_x + grid_y) * 0.035 + elapsed * 11.0)
            norm_wave = np.clip((wave + 1.5) / 3.0, 0.0, 1.0)

            # Fiery color ramp: Searing Orange (0, 110, 255) -> Deep Red (0, 20, 210)
            b = (norm_wave * 15).astype(np.uint8)
            g = (norm_wave * 125 + 25).astype(np.uint8)
            r = (norm_wave * 60 + 195).astype(np.uint8)
            plasma_bgr = np.dstack((b, g, r))

            sub_overlay = fx_overlay[min_y : max_y + 1, min_x : max_x + 1]
            active_pixels = sub_mask > 0
            sub_overlay[active_pixels] = cv2.add(
                sub_overlay[active_pixels], plasma_bgr[active_pixels]
            )

        # 2. Incandescent White-Hot Core Channel
        # Intense golden-white heat running through bones and palm center
        core_pts = np.where(core_mask > 0)
        if len(core_pts[0]) > 0:
            core_bgr = np.zeros_like(fx_overlay)
            core_bgr[core_pts] = (100, 225, 255)
            # Add glowing core
            fx_overlay[:] = cv2.add(fx_overlay, (core_bgr * 0.75).astype(np.uint8))

        # 3. Outer Edge Aura (Licking Flame Boundary)
        dilated = cv2.dilate(hand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        edge_mask = cv2.subtract(dilated, hand_mask)
        edge_pts = np.where(edge_mask > 0)
        if len(edge_pts[0]) > 0:
            edge_bgr = np.zeros_like(fx_overlay)
            edge_bgr[edge_pts] = (0, 130, 255)
            fx_overlay[:] = cv2.add(fx_overlay, edge_bgr)

        # 4. Fingertip Flame Plumes
        # High-energy flame tongues erupting upward from fingertips
        tip_indices = [4, 8, 12, 16, 20]
        for idx in tip_indices:
            if idx < len(anchors):
                tx, ty = anchors[idx]
                for layer in range(3):
                    plume_phase = elapsed * 15.0 + idx * 2.3 + layer * 1.2
                    sway = math.sin(plume_phase) * (6 + layer * 3)
                    height = palm_scale * (0.55 + 0.25 * math.sin(plume_phase * 0.8)) * (1.0 - layer * 0.25)
                    top_pt = (int(tx + sway), int(ty - height))
                    thickness = max(2, int((palm_scale * 0.12) / (layer + 1)))
                    color = (
                        (0, 75, 255) if layer == 0
                        else (30, 175, 255) if layer == 1
                        else (180, 245, 255)
                    )
                    cv2.line(fx_overlay, (tx, ty), top_pt, color, thickness, cv2.LINE_AA)
                    cv2.circle(fx_overlay, top_pt, thickness // 2, color, -1, cv2.LINE_AA)

        # 5. Render Fire Particles
        for p in self.fire_particles:
            life_pct = p.age / p.lifetime
            rem = max(0.0, 1.0 - life_pct)
            px, py = int(p.x), int(p.y)
            if not (0 <= px < w and 0 <= py < h):
                continue

            if p.p_type == "flame":
                # Multi-color gradient: White -> Yellow -> Fiery Orange -> Crimson
                cur_size = max(1, int(p.size * (0.4 + 0.6 * rem)))
                if life_pct < 0.25:
                    color = (int(120 * rem), int(210 * rem + 40), 255)
                elif life_pct < 0.65:
                    color = (0, int(150 * rem + 20), int(235 * rem + 20))
                else:
                    color = (0, int(50 * rem), int(160 * rem))

                cv2.circle(fx_overlay, (px, py), cur_size, color, -1, cv2.LINE_AA)
                # Outer glow
                cv2.circle(fx_overlay, (px, py), cur_size + 2, (0, int(80 * rem), int(180 * rem)), 1, cv2.LINE_AA)

            elif p.p_type == "spark":
                # Blinding golden cinder spark with trail
                flicker = 0.75 + 0.25 * math.sin(p.age * 40.0)
                cur_size = max(1, int(p.size * rem * flicker))
                color = (int(100 * rem), int(230 * rem), 255)
                tail_x = int(p.x - p.vx * 0.018)
                tail_y = int(p.y - p.vy * 0.018)
                cv2.line(fx_overlay, (px, py), (tail_x, tail_y), color, max(1, cur_size), cv2.LINE_AA)
                cv2.circle(fx_overlay, (px, py), cur_size, (220, 245, 255), -1, cv2.LINE_AA)

            else:
                # Fading smoke puff
                smoke_size = max(2, int(p.size * (0.8 + 0.5 * life_pct)))
                alpha_smoke = int(45 * rem)
                cv2.circle(fx_overlay, (px, py), smoke_size, (alpha_smoke, alpha_smoke, alpha_smoke * 2), -1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # WATER SIMULATION
    # -------------------------------------------------------------------------
    def _update_water_particles(
        self,
        anchors: list[tuple[int, int]],
        palm: tuple[int, int],
        hand_velocity: np.ndarray,
        delta_time: float,
        palm_scale: float,
    ) -> None:
        spawn_count = int(12 * (palm_scale / 50.0))
        spawn_count = max(8, min(spawn_count, 20))

        wind_x = -hand_velocity[0] * 0.20
        wind_y = -hand_velocity[1] * 0.20

        for _ in range(spawn_count):
            anchor_x, anchor_y = self.random.choice(anchors)
            roll = self.random.random()

            if roll < 0.60:
                # Flowing / dripping water bead
                vx = self.random.uniform(-18.0, 18.0) + wind_x
                vy = self.random.uniform(40.0, 120.0) + wind_y
                lifetime = self.random.uniform(0.45, 0.85)
                size = self.random.uniform(palm_scale * 0.08, palm_scale * 0.18)
                self.water_particles.append(
                    WaterParticle(
                        x=anchor_x + self.random.uniform(-5, 5),
                        y=anchor_y + self.random.uniform(-4, 4),
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=lifetime,
                        size=size,
                        p_type="droplet",
                    )
                )
            elif roll < 0.85:
                # Splash droplet from hand movement
                speed = math.hypot(hand_velocity[0], hand_velocity[1])
                spread = max(30.0, min(speed * 0.5, 180.0))
                vx = self.random.uniform(-spread, spread) + wind_x * 1.4
                vy = self.random.uniform(-spread * 0.8, spread) + wind_y * 1.4
                lifetime = self.random.uniform(0.35, 0.75)
                size = self.random.uniform(2.5, 6.0)
                self.water_particles.append(
                    WaterParticle(
                        x=anchor_x + self.random.uniform(-4, 4),
                        y=anchor_y + self.random.uniform(-4, 4),
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=lifetime,
                        size=size,
                        p_type="splash",
                    )
                )
            else:
                # Ethereal hydro-mist
                vx = self.random.uniform(-15.0, 15.0)
                vy = self.random.uniform(-25.0, 25.0)
                lifetime = self.random.uniform(0.60, 1.1)
                size = self.random.uniform(palm_scale * 0.15, palm_scale * 0.35)
                self.water_particles.append(
                    WaterParticle(
                        x=anchor_x + self.random.uniform(-8, 8),
                        y=anchor_y + self.random.uniform(-8, 8),
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=lifetime,
                        size=size,
                        p_type="mist",
                    )
                )

        alive: list[WaterParticle] = []
        for p in self.water_particles:
            p.age += delta_time
            if p.age >= p.lifetime:
                continue

            if p.p_type == "droplet":
                # Fluid gravity acceleration + drag
                p.vy += 480.0 * delta_time
                p.vx *= 0.98
            elif p.p_type == "splash":
                p.vy += 520.0 * delta_time
                p.vx *= 0.96
            else:
                # Mist floats gently
                p.vy += 10.0 * delta_time

            p.x += p.vx * delta_time
            p.y += p.vy * delta_time
            alive.append(p)

        self.water_particles = alive

    def _draw_water(
        self,
        fx_overlay: np.ndarray,
        hand_mask: np.ndarray,
        core_mask: np.ndarray,
        anchors: list[tuple[int, int]],
        palm: tuple[int, int],
        elapsed: float,
        palm_scale: float,
    ) -> None:
        h, w = fx_overlay.shape[:2]

        # 1. Living Fluid Hydro-Skin (Caustic Refractions & Deep Azure Tone)
        ys, xs = np.where(hand_mask > 0)
        if len(xs) > 0:
            min_x, max_x = max(0, xs.min()), min(w - 1, xs.max())
            min_y, max_y = max(0, ys.min()), min(h - 1, ys.max())
            sub_mask = hand_mask[min_y : max_y + 1, min_x : max_x + 1]

            grid_y, grid_x = np.ogrid[min_y : max_y + 1, min_x : max_x + 1]

            # Shimmering aquatic caustics
            caustic1 = np.sin(grid_x * 0.065 + elapsed * 5.0) * np.cos(grid_y * 0.065 + elapsed * 4.0)
            caustic2 = np.sin((grid_x - grid_y) * 0.045 - elapsed * 6.0)
            combined_caustic = np.clip((caustic1 + caustic2 + 1.2) / 2.4, 0.0, 1.0)

            # Deep Oceanic Palette: Saturated Azure (220, 120, 15) -> Crystal Aqua (255, 235, 130)
            b = (combined_caustic * 50 + 195).astype(np.uint8)
            g = (combined_caustic * 105 + 115).astype(np.uint8)
            r = (combined_caustic * 95 + 15).astype(np.uint8)
            water_bgr = np.dstack((b, g, r))

            sub_overlay = fx_overlay[min_y : max_y + 1, min_x : max_x + 1]
            active_pixels = sub_mask > 0
            sub_overlay[active_pixels] = cv2.add(
                sub_overlay[active_pixels], water_bgr[active_pixels]
            )

        # 2. Specular Fresnel Rim Highlight (Crystal Edge Reflection)
        dilated = cv2.dilate(hand_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        edge_mask = cv2.subtract(dilated, hand_mask)
        edge_pts = np.where(edge_mask > 0)
        if len(edge_pts[0]) > 0:
            edge_bgr = np.zeros_like(fx_overlay)
            # Radiant cyan-white specular highlight
            edge_bgr[edge_pts] = (255, 240, 175)
            fx_overlay[:] = cv2.add(fx_overlay, edge_bgr)

        # 3. Swirling Hydro-Vortex (Orbital Water Ring around Palm)
        palm_x, palm_y = palm
        vortex_radius = palm_scale * 0.45
        for ring_idx in range(2):
            ring_pts = []
            steps = 28
            rot_speed = 4.2 if ring_idx == 0 else -3.5
            phase_offset = ring_idx * math.pi * 0.6
            for s in range(steps + 1):
                angle = (s / steps) * math.pi * 2 + elapsed * rot_speed + phase_offset
                pulse = 1.0 + 0.12 * math.sin(elapsed * 7.0 + angle * 3.0)
                r = vortex_radius * pulse * (0.85 + ring_idx * 0.3)
                rx = int(palm_x + math.cos(angle) * r)
                ry = int(palm_y + math.sin(angle) * (r * 0.68))  # Tilted orbital disc
                ring_pts.append((rx, ry))

            if len(ring_pts) > 1:
                poly_pts = np.array(ring_pts, dtype=np.int32)
                cv2.polylines(
                    fx_overlay,
                    [poly_pts],
                    True,
                    (255, 215, 80) if ring_idx == 0 else (255, 175, 40),
                    max(2, int(palm_scale * 0.05)),
                    cv2.LINE_AA,
                )

        # 4. Render Water Droplets & Mist
        for p in self.water_particles:
            rem = max(0.0, 1.0 - p.age / p.lifetime)
            px, py = int(p.x), int(p.y)
            if not (0 <= px < w and 0 <= py < h):
                continue

            if p.p_type == "droplet":
                # Realistic spherical water bead with 3D specular highlight
                rad = max(2, int(p.size * rem))
                # Translucent deep ocean body
                cv2.circle(fx_overlay, (px, py), rad, (240, 160, 30), -1, cv2.LINE_AA)
                # Cyan rim glow
                cv2.circle(fx_overlay, (px, py), rad, (255, 230, 120), 1, cv2.LINE_AA)
                # Off-center white specular glint (light reflection)
                glint_x = px - max(1, rad // 3)
                glint_y = py - max(1, rad // 3)
                cv2.circle(fx_overlay, (glint_x, glint_y), max(1, rad // 3), (255, 255, 255), -1, cv2.LINE_AA)

            elif p.p_type == "splash":
                # Fast water spray streak
                rad = max(1, int(p.size * rem))
                tail_x = int(p.x - p.vx * 0.015)
                tail_y = int(p.y - p.vy * 0.015)
                cv2.line(fx_overlay, (px, py), (tail_x, tail_y), (255, 225, 95), rad, cv2.LINE_AA)
                cv2.circle(fx_overlay, (px, py), rad, (255, 255, 240), -1, cv2.LINE_AA)

            else:
                # Ethereal hydro-mist puff
                mist_rad = max(2, int(p.size * (0.8 + 0.4 * rem)))
                alpha_mist = int(35 * rem)
                cv2.circle(fx_overlay, (px, py), mist_rad, (alpha_mist * 2, int(alpha_mist * 1.5), alpha_mist), -1, cv2.LINE_AA)

    # -------------------------------------------------------------------------
    # MAIN UPDATE & RENDER DISPATCHER
    # -------------------------------------------------------------------------
    def update_and_draw(
        self,
        fx_overlay: np.ndarray,
        points: list[tuple[int, int]],
        delta_time: float,
        elapsed: float,
    ) -> tuple[tuple[int, int], str, float]:
        """Processes and draws the active elemental effect onto the FX layer."""
        smoothed_points = self.filter.update(points)
        h, w = fx_overlay.shape[:2]
        hand_mask, core_mask, palm, palm_scale = build_hand_masks(smoothed_points, (h, w))

        if not self.mode:
            return palm, "", palm_scale

        if self.mode == "fire":
            self._update_fire_particles(
                smoothed_points, palm, self.filter.velocity, delta_time, palm_scale
            )
            self._draw_fire(
                fx_overlay, hand_mask, core_mask, smoothed_points, palm, elapsed, palm_scale
            )
        else:
            self._update_water_particles(
                smoothed_points, palm, self.filter.velocity, delta_time, palm_scale
            )
            self._draw_water(
                fx_overlay, hand_mask, core_mask, smoothed_points, palm, elapsed, palm_scale
            )

        return palm, self.mode, palm_scale


# =============================================================================
# CINEMATIC FUSION / INTERACTION SYSTEM
# =============================================================================

class FusionEngine:
    """Simulates high-energy interactions when two hands come together."""

    def __init__(self) -> None:
        self.particles: list[FusionParticle] = []
        self.random = random.Random()

    def update_and_draw(
        self,
        fx_overlay: np.ndarray,
        hand1: tuple[tuple[int, int], str, float],
        hand2: tuple[tuple[int, int], str, float],
        delta_time: float,
        elapsed: float,
    ) -> None:
        (p1, mode1, scale1), (p2, mode2, scale2) = hand1, hand2
        if not mode1 or not mode2:
            self.particles.clear()
            return

        dist = math.dist(p1, p2)
        merge_dist = fx_overlay.shape[1] * 0.42
        if dist >= merge_dist:
            self.particles.clear()
            return

        intensity = (1.0 - dist / merge_dist) ** 0.75
        mid_x = (p1[0] + p2[0]) // 2
        mid_y = (p1[1] + p2[1]) // 2
        midpoint = (mid_x, mid_y)
        avg_scale = (scale1 + scale2) * 0.5

        # ---------------------------------------------------------------------
        # CASE A: FIRE + WATER -> SUPERHEATED STEAM EXPLOSION
        # ---------------------------------------------------------------------
        if (mode1 == "fire" and mode2 == "water") or (mode1 == "water" and mode2 == "fire"):
            # Sizzling boiling steam particles
            spawn_count = int(12 * intensity)
            for _ in range(spawn_count):
                angle = self.random.uniform(0, math.pi * 2)
                speed = self.random.uniform(30.0, 160.0) * intensity
                vx = math.cos(angle) * speed
                vy = math.sin(angle) * speed - 40.0  # Convective rise
                lifetime = self.random.uniform(0.40, 0.90)
                size = self.random.uniform(avg_scale * 0.20, avg_scale * 0.55) * intensity
                p_type = "steam" if self.random.random() < 0.75 else "ember"
                self.particles.append(
                    FusionParticle(
                        x=mid_x + self.random.uniform(-10, 10),
                        y=mid_y + self.random.uniform(-10, 10),
                        vx=vx,
                        vy=vy,
                        age=0.0,
                        lifetime=lifetime,
                        size=size,
                        p_type=p_type,
                        color=(240, 245, 250) if p_type == "steam" else (40, 180, 255),
                    )
                )

            # Crackling thermal lightning arcs between hands
            arc_steps = 18
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            arc_pts = []
            for s in range(arc_steps + 1):
                t = s / arc_steps
                jitter = (self.random.uniform(-18.0, 18.0) * intensity) if (0 < s < arc_steps) else 0.0
                nx = -dy / (dist + 1e-4) * jitter
                ny = dx / (dist + 1e-4) * jitter
                arc_pts.append((int(p1[0] + dx * t + nx), int(p1[1] + dy * t + ny)))

            if len(arc_pts) > 1:
                poly = np.array(arc_pts, dtype=np.int32)
                cv2.polylines(fx_overlay, [poly], False, (255, 230, 180), max(2, int(5 * intensity)), cv2.LINE_AA)
                cv2.polylines(fx_overlay, [poly], False, (255, 255, 255), 1, cv2.LINE_AA)

            # Pulsing central reaction shockwave
            shock_radius = int(avg_scale * (0.4 + 0.4 * math.sin(elapsed * 18.0)) * intensity)
            cv2.circle(fx_overlay, midpoint, shock_radius, (245, 240, 230), max(1, int(3 * intensity)), cv2.LINE_AA)
            cv2.circle(fx_overlay, midpoint, max(2, shock_radius // 2), (255, 255, 255), -1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # CASE B: DUAL FIRE -> SOLAR CATACLYSM / INFERNO CONDUIT
        # ---------------------------------------------------------------------
        elif mode1 == "fire" and mode2 == "fire":
            # Solar plasma flare particles
            spawn_count = int(10 * intensity)
            for _ in range(spawn_count):
                angle = self.random.uniform(0, math.pi * 2)
                speed = self.random.uniform(40.0, 190.0) * intensity
                self.particles.append(
                    FusionParticle(
                        x=mid_x + self.random.uniform(-8, 8),
                        y=mid_y + self.random.uniform(-8, 8),
                        vx=math.cos(angle) * speed,
                        vy=math.sin(angle) * speed - 60.0,
                        age=0.0,
                        lifetime=self.random.uniform(0.35, 0.75),
                        size=self.random.uniform(3.0, 8.0),
                        p_type="ember",
                        color=(50, 210, 255),
                    )
                )

            # Twisting double-helix fire ribbon
            steps = 22
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = max(dist, 1.0)
            nx = -dy / length
            ny = dx / length

            for helix in (-1, 1):
                helix_pts = []
                for s in range(steps + 1):
                    t = s / steps
                    wave = math.sin(t * math.pi * 4.0 + elapsed * 14.0 * helix) * (18.0 * intensity)
                    helix_pts.append((int(p1[0] + dx * t + nx * wave), int(p1[1] + dy * t + ny * wave)))
                if len(helix_pts) > 1:
                    cv2.polylines(fx_overlay, [np.array(helix_pts, dtype=np.int32)], False, (0, 130, 255), max(2, int(6 * intensity)), cv2.LINE_AA)
                    cv2.polylines(fx_overlay, [np.array(helix_pts, dtype=np.int32)], False, (120, 235, 255), max(1, int(2 * intensity)), cv2.LINE_AA)

            # Searing white-hot solar orb at midpoint
            solar_rad = int(avg_scale * 0.48 * intensity)
            cv2.circle(fx_overlay, midpoint, solar_rad, (0, 140, 255), -1, cv2.LINE_AA)
            cv2.circle(fx_overlay, midpoint, int(solar_rad * 0.65), (50, 220, 255), -1, cv2.LINE_AA)
            cv2.circle(fx_overlay, midpoint, max(2, int(solar_rad * 0.35)), (255, 255, 255), -1, cv2.LINE_AA)

        # ---------------------------------------------------------------------
        # CASE C: DUAL WATER -> TIDAL SURGE / HYDRO-VORTEX
        # ---------------------------------------------------------------------
        else:
            spawn_count = int(10 * intensity)
            for _ in range(spawn_count):
                angle = self.random.uniform(0, math.pi * 2)
                speed = self.random.uniform(40.0, 160.0) * intensity
                self.particles.append(
                    FusionParticle(
                        x=mid_x + self.random.uniform(-8, 8),
                        y=mid_y + self.random.uniform(-8, 8),
                        vx=math.cos(angle) * speed,
                        vy=math.sin(angle) * speed + 30.0,
                        age=0.0,
                        lifetime=self.random.uniform(0.40, 0.85),
                        size=self.random.uniform(3.0, 7.0),
                        p_type="splash",
                        color=(255, 230, 100),
                    )
                )

            # High-pressure swirling water torrent bridge
            steps = 22
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = max(dist, 1.0)
            nx = -dy / length
            ny = dx / length

            for stream in (-1, 1):
                stream_pts = []
                for s in range(steps + 1):
                    t = s / steps
                    wave = math.cos(t * math.pi * 3.5 + elapsed * 10.0 * stream) * (16.0 * intensity)
                    stream_pts.append((int(p1[0] + dx * t + nx * wave), int(p1[1] + dy * t + ny * wave)))
                if len(stream_pts) > 1:
                    cv2.polylines(fx_overlay, [np.array(stream_pts, dtype=np.int32)], False, (255, 140, 25), max(2, int(6 * intensity)), cv2.LINE_AA)
                    cv2.polylines(fx_overlay, [np.array(stream_pts, dtype=np.int32)], False, (255, 235, 140), max(1, int(2 * intensity)), cv2.LINE_AA)

            # Central hydro-sphere with orbital splash ring
            water_rad = int(avg_scale * 0.42 * intensity)
            cv2.circle(fx_overlay, midpoint, water_rad, (245, 160, 35), -1, cv2.LINE_AA)
            cv2.circle(fx_overlay, midpoint, int(water_rad * 0.65), (255, 225, 120), -1, cv2.LINE_AA)
            cv2.circle(fx_overlay, midpoint, max(2, int(water_rad * 0.3)), (255, 255, 255), -1, cv2.LINE_AA)

        # Update & draw fusion particles
        alive_particles: list[FusionParticle] = []
        for p in self.particles:
            p.age += delta_time
            if p.age >= p.lifetime:
                continue

            p.x += p.vx * delta_time
            p.y += p.vy * delta_time

            rem = max(0.0, 1.0 - p.age / p.lifetime)
            px, py = int(p.x), int(p.y)
            if 0 <= px < fx_overlay.shape[1] and 0 <= py < fx_overlay.shape[0]:
                rad = max(1, int(p.size * rem))
                color = tuple(int(c * rem) for c in p.color)
                cv2.circle(fx_overlay, (px, py), rad, color, -1, cv2.LINE_AA)
            alive_particles.append(p)

        self.particles = alive_particles


# =============================================================================
# CINEMATIC BLOOM & HUD SYSTEM
# =============================================================================

def apply_cinematic_bloom(frame: np.ndarray, fx_layer: np.ndarray) -> np.ndarray:
    """
    High-performance optical bloom filter.
    Downsamples the FX layer to produce realistic luminous glow that bathes the scene.
    """
    h, w = frame.shape[:2]
    # Fast 2x downsample for bloom pass
    small_fx = cv2.resize(fx_layer, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    bloom_blur = cv2.GaussianBlur(small_fx, (15, 15), 0)
    bloom_upscaled = cv2.resize(bloom_blur, (w, h), interpolation=cv2.INTER_LINEAR)

    # Additive blend: Frame + Bloom Glow + Crisp Effect Layer
    rendered = cv2.add(frame, (bloom_upscaled * 0.85).astype(np.uint8))
    rendered = cv2.add(rendered, fx_layer)
    return rendered


def draw_hud(
    frame: np.ndarray,
    hand_status: list[tuple[str, int, str]],
    fps: float,
    show_hud: bool,
) -> None:
    if not show_hud:
        return

    h, w = frame.shape[:2]
    # Glassmorphic top bar
    bar_height = 86
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_height), (10, 12, 18), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Title & FPS
    cv2.putText(
        frame, "FIRE & WATER AR", (18, 30),
        cv2.FONT_HERSHEY_DUPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"FPS: {fps:.0f}", (w - 110, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 240, 255), 1, cv2.LINE_AA,
    )

    # Key shortcuts
    cv2.putText(
        frame, "[1 Finger: Fire]  [2+ Fingers: Water]  [Fist: Hold]  [H: HUD]  [C: Skeleton]  [Q: Quit]",
        (18, bar_height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (170, 185, 205), 1, cv2.LINE_AA,
    )

    # Hand status indicators
    if hand_status:
        x_offset = 260
        for label, fingers, mode in hand_status:
            if "FIRE" in mode:
                color = (40, 140, 255)
                badge = f"{label}: [FIRE]"
            elif "WATER" in mode:
                color = (255, 210, 50)
                badge = f"{label}: [WATER]"
            else:
                color = (160, 175, 190)
                badge = f"{label}: [IDLE]"

            cv2.putText(
                frame, badge, (x_offset, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
            x_offset += 160
    else:
        cv2.putText(
            frame, "Show hands to webcam...", (260, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 165, 185), 1, cv2.LINE_AA,
        )


def draw_debug_skeleton(image: np.ndarray, points: list[tuple[int, int]]) -> None:
    """Optional subtle skeletal guide if requested by user."""
    connections = (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17),
    )
    for start, end in connections:
        cv2.line(image, points[start], points[end], (210, 235, 255), 1, cv2.LINE_AA)
    for pt in points:
        cv2.circle(image, pt, 2, (255, 255, 255), -1, cv2.LINE_AA)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def create_landmarker() -> vision.HandLandmarker:
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return vision.HandLandmarker.create_from_options(options)


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
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    effects: dict[str, HandEffect] = {}
    fusion_engine = FusionEngine()

    show_hud = True
    show_skeleton = False

    previous_frame_time = time.perf_counter()
    start_time = time.monotonic()
    previous_timestamp_ms = -1

    try:
        with create_landmarker() as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read a frame from the webcam.", file=sys.stderr)
                    return 1

                # Flip horizontally for natural mirror feel
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = int((time.monotonic() - start_time) * 1000)
                timestamp_ms = max(timestamp_ms, previous_timestamp_ms + 1)
                previous_timestamp_ms = timestamp_ms

                detection = landmarker.detect_for_video(mp_image, timestamp_ms)

                now = time.perf_counter()
                delta_time = min(now - previous_frame_time, 0.08)
                previous_frame_time = now
                elapsed = time.monotonic() - start_time

                fx_overlay = np.zeros_like(frame)
                hand_status: list[tuple[str, int, str]] = []
                detected_hands: list[
                    tuple[str, int, list, list[tuple[int, int]], str]
                ] = []

                if detection.hand_landmarks:
                    for index, landmarks in enumerate(detection.hand_landmarks):
                        handedness = "Hand"
                        if index < len(detection.handedness) and detection.handedness[index]:
                            handedness = detection.handedness[index][0].category_name

                        fingers = count_extended_fingers(landmarks)
                        effect = effects.setdefault(handedness, HandEffect())
                        if fingers > 0:
                            effect.select_gesture(fingers)

                        mode = effect.mode
                        display_mode = mode.upper() if mode else "IDLE"
                        if fingers == 0 and mode:
                            display_mode += " (LOCKED)"

                        points = [
                            (int(p.x * width), int(p.y * height))
                            for p in landmarks
                        ]
                        detected_hands.append(
                            (handedness, fingers, landmarks, points, mode)
                        )
                        hand_status.append((handedness, fingers, display_mode))

                # Clear departed hands
                visible_hands = {h for h, *_ in detected_hands}
                for h_name, eff in list(effects.items()):
                    if h_name not in visible_hands:
                        eff.fire_particles.clear()
                        eff.water_particles.clear()

                active_hands_data: list[tuple[tuple[int, int], str, float]] = []

                # Render individual hand effects
                for handedness, _, _, points, mode in detected_hands:
                    if mode:
                        palm, active_mode, scale = effects[handedness].update_and_draw(
                            fx_overlay, points, delta_time, elapsed
                        )
                        active_hands_data.append((palm, active_mode, scale))

                    if show_skeleton:
                        draw_debug_skeleton(frame, points)

                # Render fusion interaction if both hands are active
                if len(active_hands_data) == 2:
                    fusion_engine.update_and_draw(
                        fx_overlay,
                        active_hands_data[0],
                        active_hands_data[1],
                        delta_time,
                        elapsed,
                    )

                # Cinematic Bloom & Additive Composition
                scene = apply_cinematic_bloom(frame, fx_overlay)

                # HUD
                fps = 1.0 / max(delta_time, 0.001)
                draw_hud(scene, hand_status, fps, show_hud)

                cv2.imshow(WINDOW_NAME, scene)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    break
                elif key == ord("h"):
                    show_hud = not show_hud
                elif key == ord("c"):
                    show_skeleton = not show_skeleton
                elif key == ord("r"):
                    for eff in effects.values():
                        eff.fire_particles.clear()
                        eff.water_particles.clear()
                    fusion_engine.particles.clear()

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
