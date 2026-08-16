#!/usr/bin/env python
"""Benchmark the same strict-load inference path used by scripts/infer.py."""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfir_sam2.config import load_yaml
from pfir_sam2.inference import infer_sliding, load_model_strict, read_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="outputs/benchmark_inference.csv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    config = load_yaml(args.config)
    model_config = config["model"]
    inference_config = config["inference"]
    model, _ = load_model_strict(
        args.checkpoint,
        model_config["sam2_config"],
        model_config["sam2_checkpoint"],
        sam2_root=model_config.get("sam2_root", ""),
        device=args.device,
        lazy_init_size=int(model_config.get("lazy_init_size", 256)),
    )
    files = sorted(
        path for path in Path(args.input).iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    )
    rows = []
    for index, image_path in enumerate(files):
        image = read_rgb(image_path)
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started = time.perf_counter()
        infer_sliding(
            model,
            image,
            args.device,
            tile_size=int(inference_config.get("tile_size", 800)),
            overlap=int(inference_config.get("overlap", 200)),
            normalize=str(inference_config.get("normalize", "sam")),
            amp=bool(inference_config.get("amp", True)),
        )
        if torch.cuda.is_available() and args.device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if index >= args.warmup:
            rows.append({"image_name": image_path.name, "runtime_seconds": elapsed})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_name", "runtime_seconds"])
        writer.writeheader()
        writer.writerows(rows)
    runtimes = [float(row["runtime_seconds"]) for row in rows]
    if runtimes:
        print(
            f"[DONE] n={len(runtimes)} mean={statistics.mean(runtimes):.4f}s "
            f"median={statistics.median(runtimes):.4f}s -> {output}"
        )


if __name__ == "__main__":
    main()
