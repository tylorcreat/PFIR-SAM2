"""Figure-5 error taxonomy without publication-layout plotting code."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from .unified_evaluator import infer_source, normalize_stem, read_size_groups


ERROR_TYPES = (
    "Small FN",
    "Med/Large FN",
    "FP",
    "Boundary",
    "Merge",
    "Split",
)


def load_map(path: Path | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if path is None:
        if shape is None:
            raise ValueError("shape is required for a missing prediction")
        return np.zeros(shape, dtype=np.int64)
    value = np.asarray(Image.open(path))
    if value.ndim == 3:
        value = value[..., 0]
    return value.astype(np.int64)


def areas(label_map: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(label_map, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if int(i) != 0}


def overlap_tables(gt: np.ndarray, pred: np.ndarray):
    gt_areas, pred_areas = areas(gt), areas(pred)
    candidates: list[tuple[float, int, int]] = []
    pred_to_gt: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    gt_to_pred: dict[int, list[tuple[int, float]]] = defaultdict(list)
    if not gt_areas or not pred_areas:
        return gt_areas, pred_areas, candidates, pred_to_gt, gt_to_pred
    base = int(gt.max()) + 1
    codes, counts = np.unique(pred.astype(np.int64).ravel() * base + gt.ravel(), return_counts=True)
    for code, intersection in zip(codes, counts):
        pred_id, gt_id = int(code // base), int(code % base)
        if pred_id == 0 or gt_id == 0:
            continue
        intersection = int(intersection)
        union = gt_areas[gt_id] + pred_areas[pred_id] - intersection
        iou = intersection / union if union else 0.0
        gt_coverage = intersection / gt_areas[gt_id]
        candidates.append((iou, gt_id, pred_id))
        pred_to_gt[pred_id].append((gt_id, iou, gt_coverage))
        gt_to_pred[gt_id].append((pred_id, gt_coverage))
    candidates.sort(reverse=True)
    return gt_areas, pred_areas, candidates, pred_to_gt, gt_to_pred


def greedy_match(candidates: list[tuple[float, int, int]], threshold: float = 0.5):
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[float, int, int]] = []
    for iou, gt_id, pred_id in candidates:
        if iou < threshold:
            break
        if gt_id in used_gt or pred_id in used_pred:
            continue
        used_gt.add(gt_id)
        used_pred.add(pred_id)
        matches.append((iou, gt_id, pred_id))
    return matches, used_gt, used_pred


def analyze_image(
    gt: np.ndarray,
    pred: np.ndarray,
    image_id: str,
    size_groups: dict[tuple[str, int], str],
) -> dict[str, int]:
    gt_areas, pred_areas, candidates, pred_to_gt, gt_to_pred = overlap_tables(gt, pred)
    matches, matched_gt, matched_pred = greedy_match(candidates)
    false_negative_ids = set(gt_areas) - matched_gt
    false_positive_ids = set(pred_areas) - matched_pred
    merge_ids = {
        pred_id
        for pred_id, overlaps in pred_to_gt.items()
        if len({gt_id for gt_id, iou, coverage in overlaps if iou > 0.1 or coverage > 0.2}) >= 2
    }
    split_ids = {
        gt_id
        for gt_id, overlaps in gt_to_pred.items()
        if len({pred_id for pred_id, coverage in overlaps if coverage > 0.1}) >= 2
    }
    small_fn = sum(
        size_groups.get((image_id, gt_id), "") == "small"
        for gt_id in false_negative_ids
        if gt_id not in split_ids
    )
    medium_large_fn = sum(
        size_groups.get((image_id, gt_id), "") != "small"
        for gt_id in false_negative_ids
        if gt_id not in split_ids
    )
    false_positive = 0
    boundary = 0
    for pred_id in false_positive_ids - merge_ids:
        maximum_iou = max(
            (iou for _gt_id, iou, _coverage in pred_to_gt.get(pred_id, [])),
            default=0.0,
        )
        if maximum_iou < 0.1:
            false_positive += 1
        elif maximum_iou < 0.5:
            boundary += 1
        else:
            false_positive += 1
    boundary += sum(1 for iou, _gt_id, _pred_id in matches if 0.5 <= iou < 0.75)
    return {
        "Small FN": int(small_fn),
        "Med/Large FN": int(medium_large_fn),
        "FP": int(false_positive),
        "Boundary": int(boundary),
        "Merge": len(merge_ids),
        "Split": len(split_ids),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-mask-dir", required=True)
    parser.add_argument("--gt-boxes-csv", required=True)
    parser.add_argument("--method", action="append", required=True, help="Name=prediction_dir")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    gt_dir = Path(args.gt_mask_dir)
    size_groups = read_size_groups(args.gt_boxes_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for item in args.method:
        name, directory = item.split("=", 1)
        pred_files = {
            normalize_stem(path): path
            for path in Path(directory).iterdir()
            if path.suffix.lower() in {".png", ".tif", ".tiff"}
        }
        for gt_path in sorted(gt_dir.iterdir()):
            if gt_path.suffix.lower() not in {".png", ".tif", ".tiff"}:
                continue
            image_id = normalize_stem(gt_path)
            gt = load_map(gt_path)
            pred = load_map(pred_files.get(image_id), gt.shape)
            counts = analyze_image(gt, pred, image_id, size_groups)
            rows.append(
                {
                    "method": name,
                    "source": infer_source(gt_path.name),
                    "image_id": image_id,
                    **counts,
                    "total event count": sum(counts.values()),
                }
            )
    output_path = output_dir / "error_counts_per_image.csv"
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[DONE] wrote {len(rows)} rows -> {output_path}")


if __name__ == "__main__":
    main()
