"""Image-only preprocessing and explicitly synthetic camera stress diagnostics.

Nothing here loads a model, reads a transcript/audio stream, or alters the app.
Select a whole-clip preset on development data and freeze it before evaluating
test data. Synthetic blur, resolution and exposure changes are not evidence of
real webcam or intentionally silent speech performance.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from numbers import Real
import re

import cv2
import numpy as np

ALGORITHM_VERSION = "visual-robustness-v1"


def _name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", value):
        raise ValueError("Preset name must be a short lowercase identifier")


def _bounded(name, value, lo, hi):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not lo <= value <= hi:
        raise ValueError(f"{name} must be finite in [{lo}, {hi}]")


@dataclass(frozen=True)
class StabilizationConfig:
    name: str = "identity"
    strength: float = 0.0
    max_gain: float = 1.0
    max_offset: float = 0.0
    min_brightness: float = 24.0
    max_brightness: float = 231.0
    min_contrast: float = 12.0  # robust frame P90-P10, not noise standard deviation
    max_saturated_fraction: float = 0.20

    def __post_init__(self):
        _name(self.name)
        for field, lo, hi in (("strength", 0, 1), ("max_gain", 1, 1.5), ("max_offset", 0, 24),
                              ("min_brightness", 8, 64), ("max_brightness", 191, 247),
                              ("min_contrast", 4, 32), ("max_saturated_fraction", 0, .5)):
            _bounded(field, getattr(self, field), lo, hi)
            # Canonical numeric types keep equivalent JSON configs/hash identities
            # stable (and avoid accepting a numpy scalar that cannot be serialized).
            object.__setattr__(self, field, float(getattr(self, field)))


@dataclass(frozen=True)
class StressConfig:
    name: str = "clean"
    downsample: float = 1.0
    blur_sigma: float = 0.0
    exposure_gain: float = 1.0
    exposure_offset: float = 0.0
    contrast_gain: float = 1.0
    temporal_exposure_amplitude: float = 0.0

    def __post_init__(self):
        _name(self.name)
        for field, lo, hi in (("downsample", .25, 1), ("blur_sigma", 0, 2),
                              ("exposure_gain", .5, 1.25), ("exposure_offset", -24, 24),
                              ("contrast_gain", .5, 1.25), ("temporal_exposure_amplitude", 0, .25)):
            _bounded(field, getattr(self, field), lo, hi)
            object.__setattr__(self, field, float(getattr(self, field)))


def stabilization_presets() -> tuple[StabilizationConfig, ...]:
    """Small predeclared grid: identity and two bounded pixel-only alternatives."""
    return (StabilizationConfig(),
            StabilizationConfig("stabilize_mild", strength=.5, max_gain=1.15, max_offset=8),
            StabilizationConfig("stabilize_moderate", strength=.75, max_gain=1.25, max_offset=16))


def stress_presets() -> tuple[StressConfig, ...]:
    """Fixed synthetic degradations; no claim of matching a particular camera."""
    return (StressConfig(),
            StressConfig("synthetic_mild", downsample=.75, blur_sigma=.55, exposure_gain=.92,
                         exposure_offset=-4, contrast_gain=.95, temporal_exposure_amplitude=.10),
            StressConfig("synthetic_moderate", downsample=.5, blur_sigma=1.0, exposure_gain=.80,
                         exposure_offset=-10, contrast_gain=.90, temporal_exposure_amplitude=.20))


def _resolve(preset, presets, cls):
    if isinstance(preset, cls):
        return preset
    if isinstance(preset, str):
        for config in presets:
            if config.name == preset:
                return config
    raise ValueError(f"Unknown {cls.__name__} preset")


def _images(rois):
    if (not isinstance(rois, np.ndarray) or rois.dtype != np.uint8 or rois.ndim != 3
            or rois.shape[0] < 1 or min(rois.shape[1:]) < 2):
        raise ValueError("Expected nonempty T,H,W uint8 grayscale mouth crops (H,W >= 2)")
    return rois


def _fingerprint(parameters):
    serialized = json.dumps(parameters, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _pixels_sha256(rois):
    return hashlib.sha256(np.ascontiguousarray(rois).tobytes()).hexdigest()


def _pixel_summary(rois):
    means = np.mean(rois, axis=(1, 2))
    return {"mean_brightness": float(np.mean(means)), "frame_brightness_std": float(np.std(means)),
            "pixel_contrast_std": float(np.std(rois)),
            "saturated_fraction": float(np.mean((rois == 0) | (rois == 255)))}


def _diagnostics(before, after, parameters):
    return {"algorithm_version": ALGORITHM_VERSION, "parameters": parameters,
            "configuration_sha256": _fingerprint({"algorithm_version": ALGORITHM_VERSION, **parameters}),
            "shape": list(before.shape), "dtype": "uint8", "frame_order_preserved": True,
            "input_pixels_sha256": _pixels_sha256(before), "output_pixels_sha256": _pixels_sha256(after),
            "input_quality": _pixel_summary(before), "output_quality": _pixel_summary(after),
            "changed_frames": int(np.count_nonzero(np.any(before != after, axis=(1, 2)))),
            "audio_used": False, "text_used": False, "claims_real_webcam": False}


def preprocess(rois: np.ndarray, preset="identity") -> tuple[np.ndarray, dict]:
    """Apply bounded, geometry-preserving temporal photometric stabilization.

    Each frame's median and P90-P10 contrast approach the medians of usable
    frames from this clip. Gains and offsets are bounded; pixels stay at their
    original coordinates. Blank, dark, low-contrast or heavily clipped frames
    are left untouched, because missing lip detail cannot be recovered safely.
    This is an offline whole-clip diagnostic, not a streaming algorithm.
    """
    rois = _images(rois)
    config = _resolve(preset, stabilization_presets(), StabilizationConfig)
    quantiles = np.percentile(rois, [10, 50, 90], axis=(1, 2))
    brightness, contrast = quantiles[1], quantiles[2] - quantiles[0]
    saturation = np.mean((rois == 0) | (rois == 255), axis=(1, 2))
    usable = ((brightness >= config.min_brightness) & (brightness <= config.max_brightness)
              & (contrast >= config.min_contrast) & (saturation <= config.max_saturated_fraction))
    gains = np.ones(len(rois), np.float64)
    offsets = np.zeros(len(rois), np.float64)
    target_brightness = float(np.median(brightness[usable])) if usable.any() else None
    target_contrast = float(np.median(contrast[usable])) if usable.any() else None
    out = rois.copy()
    if config.strength and usable.any():
        desired_gain = np.clip(target_contrast / contrast[usable], 1 / config.max_gain, config.max_gain)
        gains[usable] = 1 + config.strength * (desired_gain - 1)
        offsets[usable] = config.strength * np.clip(
            target_brightness - gains[usable] * brightness[usable], -config.max_offset, config.max_offset)
        # Frame-by-frame float32 workspace keeps long clips from needing another
        # complete float video; the output remains independent of the input array.
        for index in np.flatnonzero(usable):
            frame = rois[index].astype(np.float32) * gains[index] + offsets[index]
            out[index] = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    diagnostics = _diagnostics(rois, out, {"kind": "photometric_stabilization", **asdict(config)})
    diagnostics.update({"guarded_frames": int(np.count_nonzero(~usable)),
                        "target_brightness": target_brightness, "target_contrast": target_contrast,
                        "applied_gain_range": [float(gains.min()), float(gains.max())],
                        "applied_offset_range": [float(offsets.min()), float(offsets.max())],
                        "geometry_changed": False, "uses_whole_clip_statistics": True})
    return out, diagnostics


def stress(rois: np.ndarray, preset="clean", *, seed=0) -> tuple[np.ndarray, dict]:
    """Deterministic SYNTHETIC resolution, blur and exposure stress, retaining order.

    Downsample/restore preserves image extent and uses no affine crop, mirroring
    or frame interpolation. Resolution loss is synthetic; it cannot stand in for
    a real camera or recover image detail. The seed changes exposure phase only.
    """
    rois = _images(rois)
    config = _resolve(preset, stress_presets(), StressConfig)
    if type(seed) is not int or not 0 <= seed <= 2 ** 32 - 1:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    synthetic = any((config.downsample != 1, config.blur_sigma != 0, config.exposure_gain != 1,
                     config.exposure_offset != 0, config.contrast_gain != 1,
                     config.temporal_exposure_amplitude != 0))
    out = rois.copy()
    if synthetic:
        phase = np.random.default_rng(seed).uniform(0, 2 * np.pi)
        gain = config.exposure_gain * (1 + config.temporal_exposure_amplitude * np.sin(
            np.linspace(0, 2 * np.pi, len(rois), endpoint=False) + phase))
        height, width = rois.shape[1:]
        small_size = (max(2, int(round(width * config.downsample))),
                      max(2, int(round(height * config.downsample))))
        for index, original in enumerate(rois):
            frame = original
            if config.downsample != 1:
                frame = cv2.resize(frame, small_size, interpolation=cv2.INTER_AREA)
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            if config.blur_sigma:
                frame = cv2.GaussianBlur(frame, (0, 0), config.blur_sigma, borderType=cv2.BORDER_REFLECT_101)
            frame = ((frame.astype(np.float32) - 127.5) * config.contrast_gain + 127.5)
            out[index] = np.clip(np.rint(frame * gain[index] + config.exposure_offset), 0, 255).astype(np.uint8)
    diagnostics = _diagnostics(rois, out, {"kind": "synthetic_camera_stress", "seed": seed, **asdict(config)})
    diagnostics.update({"synthetic": synthetic,
                        "domain_label": "synthetic_camera_stress" if synthetic else "source_mouth_crop",
                        "real_webcam_measurement": False, "affine_geometry_changed": False})
    return out, diagnostics


def apply_visual(rois: np.ndarray, *, preprocessing="identity", stress_preset="clean", seed=0) -> tuple[np.ndarray, dict]:
    """Apply synthetic stress first, then stabilization; never select a preset."""
    degraded, stress_evidence = stress(rois, stress_preset, seed=seed)
    output, preprocess_evidence = preprocess(degraded, preprocessing)
    parameters = {"stress": stress_evidence["parameters"], "preprocessing": preprocess_evidence["parameters"]}
    diagnostics = _diagnostics(rois, output, parameters)
    diagnostics.update({"stress": stress_evidence, "preprocessing": preprocess_evidence,
                        "synthetic": stress_evidence["synthetic"],
                        "real_webcam_measurement": False, "operation_order": ["stress", "preprocessing"]})
    return output, diagnostics
