"""Pixel-only research transformations; no model, transcript, audio or hardware."""
from dataclasses import FrozenInstanceError
import inspect
import json

import numpy as np
import pytest

from research.visual_robustness import (StabilizationConfig, StressConfig, apply_visual,
                                       preprocess, stabilization_presets, stress, stress_presets)


def textured_clip():
    # Asymmetric moving marks expose shifts, mirroring and frame reordering.
    yy, xx = np.indices((96, 96))
    base = 60 + ((xx * 3 + yy * 7) % 100)
    frames = []
    for index, gain in enumerate([.85, 1.0, 1.15, .95, 1.10]):
        frame = np.rint(base * gain).astype(np.uint8)
        frame[13 + 4 * index, 27 + 2 * index] = 215
        frames.append(frame)
    return np.stack(frames)


def test_identity_is_bit_exact_without_mutating_or_aliasing_input():
    images = textured_clip()
    before = images.copy()
    for output, diagnostics in (preprocess(images), stress(images), apply_visual(images)):
        np.testing.assert_array_equal(output, before)
        assert output.shape == images.shape and output.dtype == np.uint8
        assert not np.shares_memory(output, images)
        assert diagnostics["input_pixels_sha256"] == diagnostics["output_pixels_sha256"]
        assert diagnostics["frame_order_preserved"] and diagnostics["changed_frames"] == 0
        json.dumps(diagnostics, allow_nan=False)
    np.testing.assert_array_equal(images, before)


def test_small_predeclared_grid_and_configs_are_frozen():
    assert [p.name for p in stabilization_presets()] == ["identity", "stabilize_mild", "stabilize_moderate"]
    assert [p.name for p in stress_presets()] == ["clean", "synthetic_mild", "synthetic_moderate"]
    with pytest.raises(FrozenInstanceError):
        stabilization_presets()[1].max_gain = 2


def test_equivalent_numeric_configs_have_the_same_parameter_identity():
    a, evidence_a = preprocess(textured_clip(), StabilizationConfig(max_offset=0))
    b, evidence_b = preprocess(textured_clip(), StabilizationConfig(max_offset=np.float64(0)))
    np.testing.assert_array_equal(a, b)
    assert evidence_a == evidence_b


def test_noncontiguous_input_retains_logical_geometry_and_frame_order():
    images = textured_clip()[:, :, ::-1]
    assert not images.flags.c_contiguous
    identity, _ = preprocess(images)
    stabilized, _ = preprocess(images, "stabilize_mild")
    np.testing.assert_array_equal(identity, images)
    for original, transformed in zip(images, stabilized):
        assert np.unravel_index(original.argmax(), original.shape) == np.unravel_index(transformed.argmax(), transformed.shape)


@pytest.mark.parametrize("preset", stabilization_presets())
def test_stabilization_is_deterministic_geometry_and_frame_order_are_preserved(preset):
    images = textured_clip()
    before = images.copy()
    output, evidence = preprocess(images, preset)
    repeated, repeat_evidence = preprocess(images, preset.name)
    np.testing.assert_array_equal(output, repeated)
    np.testing.assert_array_equal(images, before)
    assert evidence == repeat_evidence
    assert output.shape == images.shape and output.dtype == np.uint8
    assert not evidence["geometry_changed"] and evidence["frame_order_preserved"]
    for original, transformed in zip(images, output):
        assert np.unravel_index(original.argmax(), original.shape) == np.unravel_index(transformed.argmax(), transformed.shape)
    assert evidence["applied_gain_range"][0] >= 1 / preset.max_gain
    assert evidence["applied_gain_range"][1] <= preset.max_gain
    assert min(evidence["applied_offset_range"]) >= -preset.max_offset
    assert max(evidence["applied_offset_range"]) <= preset.max_offset
    if preset.strength:
        assert evidence["output_quality"]["frame_brightness_std"] < evidence["input_quality"]["frame_brightness_std"]
        assert evidence["changed_frames"] > 0


@pytest.mark.parametrize("preset", ["stabilize_mild", "stabilize_moderate"])
def test_dark_blank_low_contrast_and_clipped_frames_do_not_amplify_noise(preset):
    rng = np.random.default_rng(73)
    images = np.stack([np.zeros((96, 96), np.uint8),
                       rng.integers(2, 12, (96, 96), dtype=np.uint8),
                       rng.integers(98, 103, (96, 96), dtype=np.uint8),
                       np.full((96, 96), 255, np.uint8)])
    output, evidence = preprocess(images, preset)
    np.testing.assert_array_equal(output, images)
    assert evidence["guarded_frames"] == 4 and evidence["changed_frames"] == 0
    assert evidence["target_brightness"] is None and evidence["target_contrast"] is None
    json.dumps(evidence, allow_nan=False)


def test_guarded_frames_are_not_pulled_towards_good_clip_statistics():
    images = textured_clip()
    images[1] = 0
    images[3] = 5
    output, evidence = preprocess(images, "stabilize_moderate")
    np.testing.assert_array_equal(output[[1, 3]], images[[1, 3]])
    assert evidence["guarded_frames"] == 2


@pytest.mark.parametrize("preset", ["synthetic_mild", "synthetic_moderate"])
def test_synthetic_stress_is_seeded_and_explicitly_not_webcam_evidence(preset):
    images = textured_clip()
    before = images.copy()
    a, evidence = stress(images, preset, seed=42)
    b, evidence_b = stress(images, preset, seed=42)
    c, evidence_c = stress(images, preset, seed=43)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(images, before)
    assert evidence == evidence_b and not np.array_equal(a, c)
    assert evidence["configuration_sha256"] != evidence_c["configuration_sha256"]
    assert a.dtype == np.uint8 and a.shape == images.shape
    assert evidence["synthetic"] and evidence["domain_label"] == "synthetic_camera_stress"
    assert evidence["real_webcam_measurement"] is False and evidence["claims_real_webcam"] is False
    assert evidence["text_used"] is evidence["audio_used"] is False
    assert evidence["frame_order_preserved"] and not evidence["affine_geometry_changed"]
    assert evidence["output_quality"]["mean_brightness"] < evidence["input_quality"]["mean_brightness"]


def test_synthetic_spatial_degradation_retains_asymmetric_center_and_frame_order():
    images = np.full((4, 96, 96), 60, np.uint8)
    coordinates = [(20, 25), (30, 40), (50, 55), (65, 70)]
    for frame, (y, x) in zip(images, coordinates):
        frame[y - 4:y + 5, x - 4:x + 5] = 200
    output, _ = stress(images, "synthetic_moderate", seed=0)
    yy, xx = np.indices((96, 96))
    for frame, (y, x) in zip(output, coordinates):
        weights = frame.astype(float) - frame.min()
        assert float((yy * weights).sum() / weights.sum()) == pytest.approx(y, abs=1)
        assert float((xx * weights).sum() / weights.sum()) == pytest.approx(x, abs=1)


def test_composition_stresses_then_stabilizes_with_reproducible_diagnostics():
    images = textured_clip()
    degraded, stress_evidence = stress(images, "synthetic_mild", seed=7)
    expected, preprocessing_evidence = preprocess(degraded, "stabilize_mild")
    actual, evidence = apply_visual(images, preprocessing="stabilize_mild", stress_preset="synthetic_mild", seed=7)
    np.testing.assert_array_equal(actual, expected)
    assert evidence["operation_order"] == ["stress", "preprocessing"]
    assert evidence["stress"] == stress_evidence and evidence["preprocessing"] == preprocessing_evidence
    assert evidence["preprocessing"]["input_pixels_sha256"] == evidence["stress"]["output_pixels_sha256"]
    assert evidence["synthetic"] and not evidence["real_webcam_measurement"]
    json.dumps(evidence, allow_nan=False)


@pytest.mark.parametrize("factory, kwargs", [
    (StabilizationConfig, {"strength": float("nan")}),
    (StabilizationConfig, {"strength": True}),
    (StabilizationConfig, {"strength": -1}),
    (StabilizationConfig, {"max_gain": 2}),
    (StabilizationConfig, {"max_offset": 25}),
    (StabilizationConfig, {"min_brightness": 0}),
    (StabilizationConfig, {"max_saturated_fraction": .8}),
    (StressConfig, {"downsample": 0}),
    (StressConfig, {"blur_sigma": -1}),
    (StressConfig, {"exposure_gain": float("inf")}),
    (StressConfig, {"contrast_gain": "1"}),
    (StressConfig, {"temporal_exposure_amplitude": .5}),
    (StressConfig, {"name": "../bad"}),
])
def test_malformed_or_unbounded_parameters_are_rejected(factory, kwargs):
    with pytest.raises(ValueError):
        factory(**kwargs)


@pytest.mark.parametrize("images", [[], np.zeros((0, 96, 96), np.uint8), np.zeros((2, 96, 96, 3), np.uint8),
                                   np.zeros((2, 96, 96), np.float32), np.zeros((2, 1, 96), np.uint8)])
def test_malformed_pixels_are_rejected(images):
    for transform in (preprocess, stress, apply_visual):
        with pytest.raises(ValueError, match="uint8 grayscale"):
            transform(images)


@pytest.mark.parametrize("seed", [True, -1, 2 ** 32, 0.5, "1"])
def test_malformed_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        stress(textured_clip(), seed=seed)


def test_unknown_presets_are_not_silently_replaced_and_api_has_no_labels():
    with pytest.raises(ValueError, match="Unknown"):
        preprocess(textured_clip(), "missing")
    with pytest.raises(ValueError, match="Unknown"):
        stress(textured_clip(), "missing")
    for transform in (preprocess, stress, apply_visual):
        assert not {"reference", "text", "audio", "transcript", "sample"} & set(inspect.signature(transform).parameters)
