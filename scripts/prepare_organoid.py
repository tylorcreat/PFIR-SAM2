#!/usr/bin/env python
"""Validate released OrganoIDNetData folders and write a local manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def normalized_stem(path: Path) -> str:
    stem = path.stem.lower()
    for suffix in ("_mask", "_masks", "_masks_organoid"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", default="outputs/organoid_dataset_manifest.csv")
    parser.add_argument(
        "--size-groups-output-dir",
        default="outputs/organoid_size_groups",
        help="Directory for split-specific instance-area size-group CSV files.",
    )
    args = parser.parse_args()
    root = Path(args.data_root)
    rows = []
    instances_by_split: dict[str, list[dict[str, object]]] = {}
    for split in ("Train", "Val", "Test"):
        image_dir = root / split / "Images"
        mask_dir = root / split / "Masks"
        if not image_dir.is_dir() or not mask_dir.is_dir():
            raise FileNotFoundError(
                f"Expected author-distributed folders {image_dir} and {mask_dir}"
            )
        masks = {
            normalized_stem(path): path
            for path in mask_dir.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        }
        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        split_instances: list[dict[str, object]] = []
        for image_path in images:
            mask_path = masks.get(normalized_stem(image_path))
            if mask_path is None:
                raise RuntimeError(f"No matching mask for {image_path.name}")
            with Image.open(image_path) as image:
                width, height = image.size
            rows.append(
                {
                    "split": split,
                    "image_name": image_path.name,
                    "mask_name": mask_path.name,
                    "width": width,
                    "height": height,
                }
            )
            mask = np.asarray(Image.open(mask_path))
            if mask.ndim == 3:
                mask = mask[..., 0]
            ids, counts = np.unique(mask, return_counts=True)
            for instance_id, area in zip(ids, counts):
                if int(instance_id) == 0:
                    continue
                split_instances.append(
                    {
                        "split": split.lower(),
                        "image_id": normalized_stem(image_path),
                        "image_name": image_path.name,
                        "instance_id": int(instance_id),
                        "area": int(area),
                        "size_group": "",
                    }
                )
        instances_by_split[split.lower()] = split_instances
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    train_areas = np.asarray(
        [float(row["area"]) for row in instances_by_split.get("train", [])],
        dtype=float,
    )
    train_areas = train_areas[train_areas >= 10]
    if train_areas.size == 0:
        raise RuntimeError("No training instances were found for size-group thresholds")
    small_threshold = float(np.quantile(train_areas, 1 / 3))
    medium_threshold = float(np.quantile(train_areas, 2 / 3))
    size_output = Path(args.size_groups_output_dir)
    size_output.mkdir(parents=True, exist_ok=True)
    for split, instance_rows in instances_by_split.items():
        for row in instance_rows:
            area = float(row["area"])
            row["size_group"] = (
                "small"
                if area <= small_threshold
                else "medium"
                if area <= medium_threshold
                else "large"
            )
        path = size_output / f"bbox_instances_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(instance_rows[0]))
            writer.writeheader()
            writer.writerows(instance_rows)
    print(f"[DONE] validated {len(rows)} image-mask pairs -> {output}")
    print(
        "[DONE] size groups from retained Train areas (area >= 10 pixels): "
        f"small<={small_threshold:.6f}, medium<={medium_threshold:.6f} -> {size_output}"
    )


if __name__ == "__main__":
    main()
