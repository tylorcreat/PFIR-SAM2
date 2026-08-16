#!/usr/bin/env python
"""Run strict PFIR-SAM2 inference and final-v2 instance reconstruction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pfir_sam2.config import load_yaml
from pfir_sam2.inference import load_model_strict, run_inference_directory
from pfir_sam2.reconstruction import (
    RawInstanceConfig,
    ReconstructionConfig,
    RescueConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="")
    parser.add_argument("--sam2-root", default="")
    parser.add_argument("--sam2-config", default="")
    parser.add_argument("--sam2-checkpoint", default="")
    parser.add_argument("--save-dense-cues", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    model_config = config.get("model", {})
    inference_config = config.get("inference", {})
    checkpoint = args.checkpoint or model_config.get("checkpoint")
    sam2_config = args.sam2_config or model_config.get("sam2_config")
    sam2_checkpoint = args.sam2_checkpoint or model_config.get("sam2_checkpoint")
    sam2_root = args.sam2_root or model_config.get("sam2_root", "")
    device = args.device or model_config.get("device", "cuda")
    for label, value in (
        ("checkpoint", checkpoint),
        ("sam2_config", sam2_config),
        ("sam2_checkpoint", sam2_checkpoint),
    ):
        if not value:
            parser.error(f"{label} is required through the config or CLI")

    model, load_audit = load_model_strict(
        checkpoint,
        sam2_config,
        sam2_checkpoint,
        sam2_root=sam2_root,
        device=device,
        lazy_init_size=int(model_config.get("lazy_init_size", 256)),
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "model_load_audit.json").write_text(
        json.dumps(load_audit, indent=2), encoding="utf-8"
    )
    run_inference_directory(
        model,
        args.input,
        output,
        device=device,
        tile_size=int(inference_config.get("tile_size", 800)),
        overlap=int(inference_config.get("overlap", 200)),
        normalize=str(inference_config.get("normalize", "sam")),
        amp=bool(inference_config.get("amp", True)),
        save_dense_cues=bool(
            args.save_dense_cues or inference_config.get("save_dense_cues", False)
        ),
        raw_config=RawInstanceConfig(**config.get("raw_instance", {})),
        reconstruction_config=ReconstructionConfig(**config.get("reconstruction", {})),
        rescue_config=RescueConfig(**config.get("rescue", {})),
    )


if __name__ == "__main__":
    main()
