"""Frozen final-v2 cue-guided instance reconstruction and small-object rescue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.segmentation import watershed


@dataclass(frozen=True)
class RawInstanceConfig:
    foreground_threshold: float = 0.50
    boundary_threshold: float = 0.45
    center_keep_threshold: float = 0.35
    minimum_area: int = 30


@dataclass(frozen=True)
class ReconstructionConfig:
    foreground_threshold: float = 0.45
    boundary_threshold: float = 0.40
    center_threshold: float = 0.25
    center_sigma: float = 1.0
    watershed_min_distance: int = 10
    distance_threshold: float = 3.0
    minimum_area: int = 10
    marker_dilate: int = 3
    center_weight: float = 0.70
    watershed_line: bool = False


@dataclass(frozen=True)
class RescueConfig:
    rescue_iou_threshold: float = 0.05
    rescue_containment_threshold: float = 0.30
    minimum_small_area: int = 5
    maximum_small_area: int = 500
    minimum_final_area: int = 5
    foreground_confidence_threshold: float = 0.0


def _label_areas(inst: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(inst, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if int(i) != 0}


def relabel_sequential(inst: np.ndarray, minimum_area: int = 1) -> np.ndarray:
    out = np.zeros(inst.shape, dtype=np.uint32)
    next_id = 1
    for label_id, area in _label_areas(inst).items():
        if area < minimum_area:
            continue
        out[inst == label_id] = next_id
        next_id += 1
    return out


def _connected_components_binary(binary: np.ndarray, minimum_area: int) -> np.ndarray:
    binary = binary.astype(bool)
    height, width = binary.shape
    instances = np.zeros((height, width), dtype=np.int32)
    visited = np.zeros((height, width), dtype=bool)
    next_label = 0
    neighbors = ((-1, 0), (1, 0), (0, -1), (0, 1))
    ys, xs = np.where(binary)
    for start_y, start_x in zip(ys, xs):
        if visited[start_y, start_x]:
            continue
        stack = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        coords: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            coords.append((y, x))
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if (
                    0 <= yy < height
                    and 0 <= xx < width
                    and binary[yy, xx]
                    and not visited[yy, xx]
                ):
                    visited[yy, xx] = True
                    stack.append((yy, xx))
        if len(coords) < minimum_area:
            continue
        next_label += 1
        for y, x in coords:
            instances[y, x] = next_label
    return instances


def reconstruct_raw(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    center_probability: np.ndarray,
    config: RawInstanceConfig = RawInstanceConfig(),
) -> np.ndarray:
    """Reproduce the retained raw connected-component candidate map."""
    foreground = foreground_probability >= config.foreground_threshold
    boundary = boundary_probability >= config.boundary_threshold
    center_keep = center_probability >= config.center_keep_threshold
    separated = foreground & (~boundary | center_keep)
    instances = _connected_components_binary(separated, config.minimum_area)
    if int(instances.max()) == 0 and foreground.any():
        instances = _connected_components_binary(foreground, config.minimum_area)
    return instances


def reconstruct_instances(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    center_probability: np.ndarray,
    config: ReconstructionConfig = ReconstructionConfig(),
) -> np.ndarray:
    """Run the frozen final-v2 marker-controlled watershed reconstruction."""
    foreground = foreground_probability >= config.foreground_threshold
    interior = foreground & (boundary_probability < config.boundary_threshold)
    if int(interior.sum()) < config.minimum_area:
        interior = foreground.copy()

    center_smooth = ndi.gaussian_filter(
        center_probability.astype(np.float32), sigma=config.center_sigma
    )
    coords = peak_local_max(
        center_smooth,
        min_distance=config.watershed_min_distance,
        threshold_abs=config.center_threshold,
        labels=foreground.astype(np.uint8),
        exclude_border=False,
    )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    for marker_id, (y, x) in enumerate(coords, start=1):
        markers[y, x] = marker_id

    # Boundary is reserved for this image-global no-center-marker fallback.
    if int(markers.max()) == 0:
        distance_interior = ndi.distance_transform_edt(interior)
        coords = peak_local_max(
            distance_interior,
            min_distance=config.watershed_min_distance,
            threshold_abs=config.distance_threshold,
            labels=interior.astype(np.uint8),
            exclude_border=False,
        )
        for marker_id, (y, x) in enumerate(coords, start=1):
            markers[y, x] = marker_id

    if int(markers.max()) == 0:
        return relabel_sequential(
            label(foreground).astype(np.uint16), config.minimum_area
        )

    if config.marker_dilate > 0:
        markers = ndi.grey_dilation(
            markers, size=(config.marker_dilate, config.marker_dilate)
        )
        markers = markers * foreground.astype(np.int32)

    distance = ndi.distance_transform_edt(foreground)
    distance_normalized = distance / (distance.max() + 1e-6)
    score = (
        config.center_weight * center_smooth
        + (1.0 - config.center_weight) * distance_normalized
    )
    instances = watershed(
        -score,
        markers=markers,
        mask=foreground,
        watershed_line=config.watershed_line,
    )
    return relabel_sequential(instances.astype(np.uint16), config.minimum_area)


def _max_iou_and_containment(
    mask: np.ndarray,
    reconstruction: np.ndarray,
    reconstruction_areas: dict[int, int],
) -> tuple[float, float]:
    raw_area = int(mask.sum())
    if raw_area == 0 or not reconstruction_areas:
        return 0.0, 0.0
    labels, counts = np.unique(reconstruction[mask], return_counts=True)
    max_iou = 0.0
    max_containment = 0.0
    for label_id, intersection in zip(labels, counts):
        label_id = int(label_id)
        if label_id == 0:
            continue
        intersection = int(intersection)
        union = raw_area + reconstruction_areas[label_id] - intersection
        max_iou = max(max_iou, intersection / union if union else 0.0)
        max_containment = max(max_containment, intersection / raw_area)
    return max_iou, max_containment


def rescue_small_objects(
    raw_instances: np.ndarray,
    reconstructed_instances: np.ndarray,
    foreground_probability: np.ndarray | None = None,
    config: RescueConfig = RescueConfig(),
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Restore eligible raw small objects and relabel the final map sequentially."""
    final = relabel_sequential(
        np.asarray(reconstructed_instances), config.minimum_final_area
    )
    next_label = int(final.max()) + 1
    reconstruction_areas = _label_areas(final)
    audit_rows: list[dict[str, object]] = []
    for raw_label, raw_area in _label_areas(np.asarray(raw_instances)).items():
        row: dict[str, object] = {
            "raw_label": raw_label,
            "raw_area": raw_area,
            "action": "skipped",
        }
        if not config.minimum_small_area <= raw_area <= config.maximum_small_area:
            row["action"] = "skipped_area"
            audit_rows.append(row)
            continue
        raw_mask = raw_instances == raw_label
        max_iou, max_containment = _max_iou_and_containment(
            raw_mask, final, reconstruction_areas
        )
        row["max_iou_with_reconstruction"] = max_iou
        row["max_containment_in_reconstruction"] = max_containment
        if (
            max_iou >= config.rescue_iou_threshold
            or max_containment >= config.rescue_containment_threshold
        ):
            row["action"] = "skipped_already_covered"
            audit_rows.append(row)
            continue
        if foreground_probability is not None:
            confidence = float(np.mean(foreground_probability[raw_mask]))
            row["foreground_confidence"] = confidence
            if confidence < config.foreground_confidence_threshold:
                row["action"] = "skipped_low_fg"
                audit_rows.append(row)
                continue
        free_mask = raw_mask & (final == 0)
        if int(free_mask.sum()) < config.minimum_final_area:
            row["action"] = "skipped_no_free_area"
            audit_rows.append(row)
            continue
        final[free_mask] = next_label
        row.update(
            action="rescued",
            new_label=next_label,
            rescued_area=int(free_mask.sum()),
        )
        audit_rows.append(row)
        next_label += 1
    return relabel_sequential(final, config.minimum_final_area), audit_rows


def reconstruct_final(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    center_probability: np.ndarray,
    raw_config: RawInstanceConfig = RawInstanceConfig(),
    reconstruction_config: ReconstructionConfig = ReconstructionConfig(),
    rescue_config: RescueConfig = RescueConfig(),
) -> tuple[np.ndarray, dict[str, object]]:
    raw = reconstruct_raw(
        foreground_probability,
        boundary_probability,
        center_probability,
        raw_config,
    )
    reconstructed = reconstruct_instances(
        foreground_probability,
        boundary_probability,
        center_probability,
        reconstruction_config,
    )
    final, rescue_audit = rescue_small_objects(
        raw,
        reconstructed,
        foreground_probability=foreground_probability,
        config=rescue_config,
    )
    return final, {
        "raw_instances": raw,
        "reconstructed_instances": reconstructed,
        "rescue_events": rescue_audit,
    }
