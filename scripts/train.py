#!/usr/bin/env python
"""YAML entry point for the retained full-model training protocol."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfir_sam2.config import flatten_sections, load_yaml


SUPPORTED = {
    "variant",
    "reference_ckpt",
    "save_root",
    "run_name",
    "resume",
    "overwrite",
    "dry_run",
    "deterministic",
    "data_root",
    "sam2_config",
    "sam2_ckpt",
    "sam2_root",
    "epochs",
    "batch_size",
    "num_workers",
    "lr",
    "lora_lr",
    "weight_decay",
    "seed",
    "expected_seed",
    "amp",
    "grad_clip",
    "print_freq",
    "crop_size",
    "whole_size",
    "mixed_scale",
    "whole_prob",
    "normalize",
    "min_area",
    "small_area",
    "small_boost",
    "center_sigma",
    "foreground_crop_prob",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "lora_keywords",
    "cnn_base_ch",
    "dec_ch",
    "lazy_init_size",
    "lambda_boundary",
    "lambda_center",
    "use_focal",
    "mask_thresh",
    "save_debug_every",
    "debug_items",
}
BOOLEAN_OPTIONAL = {"deterministic", "amp", "mixed_scale", "use_focal"}
STORE_TRUE = {"overwrite", "dry_run"}


def build_runtime_argv(config: dict, dry_run: bool) -> list[str]:
    flat = flatten_sections(config)
    flat["dry_run"] = bool(dry_run or flat.get("dry_run", False))
    argv: list[str] = []
    for key, value in flat.items():
        if key not in SUPPORTED or value is None:
            continue
        option = "--" + key
        if key in BOOLEAN_OPTIONAL:
            argv.append(option if bool(value) else "--no-" + key)
        elif key in STORE_TRUE:
            if bool(value):
                argv.append(option)
        else:
            argv.extend([option, str(value)])
    return argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--dry-run", action="store_true", help="Build the model and run one dry step only."
    )
    args = parser.parse_args()
    config = load_yaml(args.config)
    runtime_argv = build_runtime_argv(config, args.dry_run)
    from pfir_sam2 import formal_training

    original_argv = sys.argv
    try:
        sys.argv = ["train.py", *runtime_argv]
        formal_training.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
