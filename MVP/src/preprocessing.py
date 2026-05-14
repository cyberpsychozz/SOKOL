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


@dataclass(frozen=True)
class DronePreprocessConfig:
    enabled: bool = True
    clahe: bool = True
    bilateral: bool = False
    temporal_alpha: float = 0.0
    gamma: float = 1.0
    sharpen_amount: float = 0.0
    white_balance: bool = False
    bilateral_d: int = 5
    bilateral_sigma_color: int = 45
    bilateral_sigma_space: int = 45
    clip_limit: float = 2.0
    tile_grid_size: int = 8
    gaussian_sigma: float = 1.2


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


class DroneFramePreprocessor:
    """
    Lightweight frame enhancement for onboard inference.

    The default settings are conservative so they do not oversharpen smoke plumes
    or introduce too much latency on small edge devices.
    """

    def __init__(self, config: DronePreprocessConfig | None = None) -> None:
        self.config = config or DronePreprocessConfig()
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

        if self.config.white_balance:
            output = self._gray_world_white_balance(output)

        if self.config.bilateral:
            output = cv2.bilateralFilter(
                output,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )

        if self.config.clahe:
            output = self._apply_clahe(output)

        if abs(self.config.gamma - 1.0) > 1e-3:
            output = self._apply_gamma(output, self.config.gamma)

        if self.config.sharpen_amount > 0:
            output = self._unsharp_mask(output, self.config.sharpen_amount)

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

    def _apply_gamma(self, frame: np.ndarray, gamma: float) -> np.ndarray:
        gamma = max(gamma, 1e-3)
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((index / 255.0) ** inv_gamma) * 255 for index in range(256)],
            dtype=np.uint8,
        )
        return cv2.LUT(frame, table)

    def _unsharp_mask(self, frame: np.ndarray, amount: float) -> np.ndarray:
        blurred = cv2.GaussianBlur(frame, (0, 0), self.config.gaussian_sigma)
        return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)

    def _gray_world_white_balance(self, frame: np.ndarray) -> np.ndarray:
        balanced = frame.astype(np.float32)
        channel_means = balanced.mean(axis=(0, 1))
        mean_intensity = float(channel_means.mean())
        scales = mean_intensity / np.clip(channel_means, 1.0, None)
        balanced *= scales.reshape(1, 1, 3)
        return np.clip(balanced, 0, 255).astype(np.uint8)
