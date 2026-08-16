"""Strict checkpoint loading and retained 800/200 Hann-window inference."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch

from .core_training import (
    LoRALinear,
    PromptFreeOrganoidSAM2,
    image_to_tensor,
    initialize_lazy_modules,
    parse_keywords,
)
from .reconstruction import (
    RawInstanceConfig,
    ReconstructionConfig,
    RescueConfig,
    reconstruct_final,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def read_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def save_instance_tiff(path: str | Path, instances: np.ndarray) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    maximum = int(np.max(instances)) if instances.size else 0
    dtype = np.uint32 if maximum > np.iinfo(np.uint16).max else np.uint16
    Image.fromarray(np.asarray(instances, dtype=dtype)).save(output)


def _torch_load(path: str | Path, device: torch.device) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(
            path, map_location=device, weights_only=False, mmap=device.type == "cpu"
        )
    except (TypeError, RuntimeError):
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("PFIR checkpoint must be a mapping")
    return checkpoint


def _extract_state_dict(checkpoint: Mapping[str, Any]) -> Mapping[str, torch.Tensor]:
    for key in ("model", "model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping) and value:
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint  # type: ignore[return-value]
    raise RuntimeError("No model state dictionary was found in the checkpoint")


def load_model_strict(
    checkpoint_path: str | Path,
    sam2_config: str | Path,
    sam2_checkpoint: str | Path,
    sam2_root: str | Path | None = None,
    device: str | torch.device = "cuda",
    lazy_init_size: int = 256,
) -> tuple[PromptFreeOrganoidSAM2, dict[str, Any]]:
    device = torch.device(device)
    checkpoint = _torch_load(checkpoint_path, torch.device("cpu"))
    state = _extract_state_dict(checkpoint)
    saved_args = checkpoint.get("args", {})
    if not isinstance(saved_args, Mapping):
        saved_args = vars(saved_args) if hasattr(saved_args, "__dict__") else {}

    model = PromptFreeOrganoidSAM2(
        sam2_config=str(sam2_config),
        sam2_ckpt=str(sam2_checkpoint),
        device=device,
        sam2_root="" if sam2_root is None else str(sam2_root),
        lora_rank=int(saved_args.get("lora_rank", 8)),
        lora_alpha=float(saved_args.get("lora_alpha", 16.0)),
        lora_dropout=float(saved_args.get("lora_dropout", 0.0)),
        lora_keywords=parse_keywords(
            str(saved_args.get("lora_keywords", "q,k,v,qkv,proj,attn"))
        ),
        cnn_base_ch=int(saved_args.get("cnn_base_ch", 32)),
        dec_ch=int(saved_args.get("dec_ch", 128)),
    ).to(device)
    initialize_lazy_modules(model, device, img_size=lazy_init_size)

    model_state = model.state_dict()
    missing = sorted(set(model_state) - set(state))
    unexpected = sorted(set(state) - set(model_state))
    shape_mismatches = [
        key
        for key in sorted(set(model_state) & set(state))
        if tuple(model_state[key].shape) != tuple(state[key].shape)
    ]
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Strict checkpoint validation failed: "
            f"missing={len(missing)}, unexpected={len(unexpected)}, "
            f"shape_mismatches={len(shape_mismatches)}"
        )
    model.load_state_dict(state, strict=True)
    model.eval()
    lora_modules = sum(1 for module in model.modules() if isinstance(module, LoRALinear))
    if lora_modules != 195:
        raise RuntimeError(f"Expected 195 LoRA-wrapped Linear modules, found {lora_modules}")
    audit = {
        "checkpoint_top_level_keys": sorted(checkpoint.keys()),
        "state_dict_key_count": len(state),
        "missing_keys_count": 0,
        "unexpected_keys_count": 0,
        "shape_mismatches_count": 0,
        "lora_linear_module_count": lora_modules,
        "epoch": checkpoint.get("epoch"),
        "best_dice": checkpoint.get("best_dice"),
    }
    return model, audit


def make_2d_hann(height: int, width: int) -> np.ndarray:
    wy = np.hanning(height) if height > 1 else np.ones(1)
    wx = np.hanning(width) if width > 1 else np.ones(1)
    return np.maximum(np.outer(wy, wx).astype(np.float32), 1e-3)


@torch.no_grad()
def infer_tensor(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    amp: bool = True,
) -> dict[str, torch.Tensor]:
    enabled = amp and tensor.device.type == "cuda"
    with torch.autocast(device_type=tensor.device.type, enabled=enabled):
        prediction = model(tensor)
    return {key: torch.sigmoid(value.float()) for key, value in prediction.items()}


@torch.no_grad()
def infer_sliding(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str | torch.device,
    tile_size: int = 800,
    overlap: int = 200,
    normalize: str = "sam",
    amp: bool = True,
) -> dict[str, np.ndarray]:
    device = torch.device(device)
    height, width = image.shape[:2]
    stride = max(1, tile_size - overlap)
    padded_height = max(
        tile_size,
        int(math.ceil(max(height, tile_size) / stride) * stride + overlap),
    )
    padded_width = max(
        tile_size,
        int(math.ceil(max(width, tile_size) / stride) * stride + overlap),
    )
    canvas = np.zeros((padded_height, padded_width, 3), dtype=np.uint8)
    canvas[:height, :width] = image
    accumulators = {
        key: np.zeros((padded_height, padded_width), dtype=np.float32)
        for key in ("fg", "boundary", "center")
    }
    weight = np.zeros((padded_height, padded_width), dtype=np.float32)
    window = make_2d_hann(tile_size, tile_size)

    ys = list(range(0, max(1, padded_height - tile_size + 1), stride))
    xs = list(range(0, max(1, padded_width - tile_size + 1), stride))
    if ys[-1] != padded_height - tile_size:
        ys.append(padded_height - tile_size)
    if xs[-1] != padded_width - tile_size:
        xs.append(padded_width - tile_size)
    for y in ys:
        for x in xs:
            patch = canvas[y : y + tile_size, x : x + tile_size]
            tensor = image_to_tensor(patch, normalize=normalize)[None].to(device)
            probabilities = infer_tensor(model, tensor, amp=amp)
            for key in accumulators:
                value = probabilities[key][0, 0].detach().cpu().numpy()
                accumulators[key][y : y + tile_size, x : x + tile_size] += (
                    value * window
                )
            weight[y : y + tile_size, x : x + tile_size] += window
    weight = np.maximum(weight, 1e-6)
    return {
        key: (value / weight)[:height, :width].astype(np.float32)
        for key, value in accumulators.items()
    }


def run_inference_directory(
    model: torch.nn.Module,
    input_dir: str | Path,
    output_dir: str | Path,
    device: str | torch.device,
    tile_size: int = 800,
    overlap: int = 200,
    normalize: str = "sam",
    amp: bool = True,
    save_dense_cues: bool = False,
    raw_config: RawInstanceConfig = RawInstanceConfig(),
    reconstruction_config: ReconstructionConfig = ReconstructionConfig(),
    rescue_config: RescueConfig = RescueConfig(),
) -> list[dict[str, object]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    instance_dir = output_dir / "instances"
    instance_dir.mkdir(parents=True, exist_ok=True)
    cue_dir = output_dir / "dense_cues"
    if save_dense_cues:
        cue_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    files = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    for image_path in files:
        started = time.perf_counter()
        image = read_rgb(image_path)
        cues = infer_sliding(
            model,
            image,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            normalize=normalize,
            amp=amp,
        )
        final, intermediate = reconstruct_final(
            cues["fg"],
            cues["boundary"],
            cues["center"],
            raw_config,
            reconstruction_config,
            rescue_config,
        )
        output_path = instance_dir / f"{image_path.stem}.tif"
        save_instance_tiff(output_path, final)
        if save_dense_cues:
            for key, value in cues.items():
                np.save(cue_dir / f"{image_path.stem}_{key}.npy", value)
        elapsed = time.perf_counter() - started
        row = {
            "image_name": image_path.name,
            "output_instance": str(output_path),
            "n_instances": int(final.max()),
            "n_raw_instances": int(intermediate["raw_instances"].max()),
            "n_reconstructed_instances": int(
                intermediate["reconstructed_instances"].max()
            ),
            "n_rescued": sum(
                event.get("action") == "rescued"
                for event in intermediate["rescue_events"]
            ),
            "runtime_seconds": elapsed,
        }
        manifest.append(row)
        print(
            f"[INFER] {image_path.name}: instances={row['n_instances']} "
            f"runtime={elapsed:.3f}s"
        )
    manifest_path = output_dir / "inference_manifest.csv"
    if manifest:
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
            writer.writeheader()
            writer.writerows(manifest)
    return manifest


def namespace_from_config(config: Mapping[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**dict(config))
