from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraDenoiseConfig:
    enabled: bool = True
    clahe: bool = True
    bilateral: bool = True
    temporal_alpha: float = 0.0
    clip_limit: float = 2.0
    tile_grid_size: int = 8
    bilateral_d: int = 5
    bilateral_sigma_color: int = 45
    bilateral_sigma_space: int = 45


class CameraDenoiser:
    """Lightweight OpenCV denoising for drone/camera frames before detection."""

    def __init__(self, config: CameraDenoiseConfig | None = None) -> None:
        self.config = config or CameraDenoiseConfig()
        self._ema_frame: np.ndarray | None = None

    def reset(self) -> None:
        self._ema_frame = None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        return self.apply(frame)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if not self.config.enabled:
            return frame

        output = frame

        if self.config.temporal_alpha > 0:
            output = self._temporal_smooth(output)

        if self.config.bilateral:
            output = cv2.bilateralFilter(
                output,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )

        if self.config.clahe:
            output = self._apply_clahe(output)

        return output

    def _temporal_smooth(self, frame: np.ndarray) -> np.ndarray:
        alpha = float(np.clip(self.config.temporal_alpha, 0.0, 0.95))
        frame_f32 = frame.astype(np.float32)

        if self._ema_frame is None or self._ema_frame.shape != frame.shape:
            self._ema_frame = frame_f32
        else:
            self._ema_frame = alpha * self._ema_frame + (1.0 - alpha) * frame_f32

        return np.clip(self._ema_frame, 0, 255).astype(np.uint8)

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, channel_a, channel_b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clip_limit,
            tileGridSize=(self.config.tile_grid_size, self.config.tile_grid_size),
        )
        lightness = clahe.apply(lightness)
        merged = cv2.merge((lightness, channel_a, channel_b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
