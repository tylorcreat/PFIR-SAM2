# -*- coding: utf-8 -*-
"""
train_promptfree_organoid_sam2_full.py

Prompt-free Organoid-SAM2 full training script.

核心框架：
1) 真实加载 SAM2 checkpoint/config；
2) SAM2 image encoder / Hiera 主干大部分冻结；
3) 对 SAM2 image encoder 内部 Linear 层注入 LoRA；
4) 不使用 prompt box / point / prompt encoder；
5) 额外加入 CNN/U-Net local branch，补充局部纹理、弱边界、小目标细节；
6) 使用 SAM2 多尺度特征 + CNN 多尺度特征进入 Multi-scale Residual Fusion Decoder；
7) 输出三类 dense prediction：
   - foreground mask
   - boundary map
   - center / instance cue
8) 损失：
   - Dice + weighted BCE for foreground
   - BCE for boundary
   - MSE for center heatmap
   - small-object reweighting

重要说明：
- 这是“无 prompt box 版”，训练时不使用 box prompt，不调用 SAM2 prompt encoder，也不调用原生 SAM2 mask decoder。
- SAM2 在这里作为强 image encoder / foundation backbone；分割由新的 dense decoder 完成。
- 如果你的 sam2 版本中 image_encoder 输出字段不同，脚本里 SAM2FeatureExtractor 已做多种兼容解析：
  backbone_fpn / vision_features / dict / list / tensor 都会尝试处理。
- Install the official facebookresearch/sam2 package and provide its config and
  checkpoint paths through the command line or public YAML configuration.

推荐数据结构：
DATA_ROOT/
  train/
    images/*.png|jpg|tif
    masks/*.png|tif       # 最好是 instance id mask；二值 mask 也可跑，但 center/instance cue 会弱
  val/
    images/*.png|jpg|tif
    masks/*.png|tif

The public entry point is ``python scripts/train.py --config ...``.

如果显存不足：
- batch_size=1
- crop_size=384 或 512
- whole_size=640 或 768
- num_workers=0
- dec_ch=96
"""

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Default paths: 修改这里后，也可以不在命令行输入路径
# ============================================================
DEFAULT_DATA_ROOT = "data/OrganoID"
DEFAULT_SAVE_DIR = "outputs/training"
DEFAULT_SAM2_CONFIG = "sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CKPT = "weights/sam2.1_hiera_large.pt"


# ============================================================
# Basic utilities
# ============================================================
IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 对医学图像训练，更优先复现；如果想更快，可以关掉 deterministic
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    return [p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMG_EXTS]


def read_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def read_mask(path: Path) -> np.ndarray:
    m = Image.open(path)
    arr = np.asarray(m)
    if arr.ndim == 3:
        arr = arr[..., 0]
    # 保留 instance id；不要归一化到 0/1
    if np.issubdtype(arr.dtype, np.floating):
        arr = np.rint(arr).astype(np.int32)
    else:
        arr = arr.astype(np.int32)
    return arr


def resize_image_np(img: np.ndarray, out_hw: Tuple[int, int], is_mask: bool = False) -> np.ndarray:
    oh, ow = out_hw
    if is_mask:
        arr = img.astype(np.int32)
        # PIL 不支持 int32 label 图，转 uint16/uint32 可能不同平台行为不同；这里用 uint16 更稳
        maxv = int(arr.max()) if arr.size else 0
        if maxv <= 65535:
            pil = Image.fromarray(arr.astype(np.uint16))
        else:
            pil = Image.fromarray(arr.astype(np.int32), mode="I")
        out = pil.resize((ow, oh), resample=Image.Resampling.NEAREST)
        return np.asarray(out).astype(np.int32)
    pil = Image.fromarray(img.astype(np.uint8))
    out = pil.resize((ow, oh), resample=Image.Resampling.BILINEAR)
    return np.asarray(out, dtype=np.uint8)


def image_to_tensor(img: np.ndarray, normalize: str = "imagenet") -> torch.Tensor:
    """
    RGB uint8 HWC -> float CHW
    """
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    if normalize == "imagenet":
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        x = (x - mean) / std
    elif normalize == "sam":
        # SAM/SAM2 常用 pixel mean/std；输入若已经是 0-1，转换成近似范围
        mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)[:, None, None] / 255.0
        std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)[:, None, None] / 255.0
        x = (x - mean) / std
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize type: {normalize}")
    return x


def safe_stem(stem: str) -> str:
    s = stem
    for suffix in [
        "_img",
        "_image",
        "_images",
        "_mask",
        "_masks",
        "_masks_organoid",
        "_label",
        "_labels",
        "_instance",
        "_instances",
        "_semantic",
    ]:
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def match_image_mask_pairs(image_dir: Path, mask_dir: Path) -> List[Tuple[Path, Path]]:
    images = list_images(image_dir)
    masks = list_images(mask_dir)

    mask_by_stem: Dict[str, Path] = {}
    for mp in masks:
        mask_by_stem[safe_stem(mp.stem)] = mp
        mask_by_stem[mp.stem] = mp

    pairs: List[Tuple[Path, Path]] = []
    for ip in images:
        candidates = [
            ip.stem,
            safe_stem(ip.stem),
            ip.stem.replace("_img", ""),
            ip.stem + "_mask",
            ip.stem + "_masks",
            ip.stem + "_masks_organoid",
        ]
        found = None
        for c in candidates:
            if c in mask_by_stem:
                found = mask_by_stem[c]
                break
        if found is None:
            # 再尝试文件路径模式
            direct = [
                mask_dir / f"{ip.stem}.png",
                mask_dir / f"{ip.stem}_mask.png",
                mask_dir / f"{ip.stem}_masks.png",
                mask_dir / f"{ip.stem}_masks_organoid.png",
                mask_dir / f"{safe_stem(ip.stem)}.png",
                mask_dir / f"{safe_stem(ip.stem)}_mask.png",
                mask_dir / f"{safe_stem(ip.stem)}_masks_organoid.png",
            ]
            for dp in direct:
                if dp.exists():
                    found = dp
                    break
        if found is not None:
            pairs.append((ip, found))

    if len(pairs) == 0:
        raise RuntimeError(
            f"No image/mask pairs found.\n"
            f"image_dir={image_dir}\nmask_dir={mask_dir}\n"
            f"Expected masks with same stem or *_mask / *_masks_organoid."
        )
    return pairs


# ============================================================
# Target generation: foreground / boundary / center / weights
# ============================================================
def mask_to_foreground(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.float32)


def mask_to_boundary(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    """
    Morphological boundary from foreground, no cv2/scipy dependency.
    """
    fg = torch.from_numpy((mask > 0).astype(np.float32))[None, None]
    k = 2 * radius + 1
    dil = F.max_pool2d(fg, kernel_size=k, stride=1, padding=radius)
    ero = 1.0 - F.max_pool2d(1.0 - fg, kernel_size=k, stride=1, padding=radius)
    b = (dil - ero).clamp(0, 1)[0, 0].numpy()
    return b.astype(np.float32)


def get_instance_ids(mask: np.ndarray) -> np.ndarray:
    ids = np.unique(mask)
    return ids[ids > 0]


def extract_centers_boxes_areas(
    mask: np.ndarray,
    min_area: int = 5,
) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int, int, int]], List[int]]:
    centers: List[Tuple[float, float]] = []
    boxes: List[Tuple[int, int, int, int]] = []
    areas: List[int] = []
    ids = get_instance_ids(mask)
    for lab in ids:
        ys, xs = np.where(mask == lab)
        area = int(len(xs))
        if area < min_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        centers.append((float(xs.mean()), float(ys.mean())))
        boxes.append((x1, y1, x2, y2))
        areas.append(area)

    # 二值 mask fallback：只有一个连通整体中心。此时 center supervision 不如 instance map 准。
    if len(centers) == 0 and (mask > 0).any():
        ys, xs = np.where(mask > 0)
        area = int(len(xs))
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        centers.append((float(xs.mean()), float(ys.mean())))
        boxes.append((x1, y1, x2, y2))
        areas.append(area)
    return centers, boxes, areas


def make_center_heatmap(
    centers: Sequence[Tuple[float, float]],
    shape_hw: Tuple[int, int],
    sigma: float = 5.0,
) -> np.ndarray:
    h, w = shape_hw
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros((h, w), dtype=np.float32)
    sigma = max(float(sigma), 1.0)
    denom = 2.0 * sigma * sigma
    for cx, cy in centers:
        g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / denom).astype(np.float32)
        heat = np.maximum(heat, g)
    return heat


def make_small_object_weight_map(
    mask: np.ndarray,
    small_area: int = 256,
    boost: float = 3.0,
    boundary: Optional[np.ndarray] = None,
    boundary_boost: float = 1.5,
) -> np.ndarray:
    w = np.ones(mask.shape, dtype=np.float32)
    ids = get_instance_ids(mask)
    for lab in ids:
        area = int((mask == lab).sum())
        if 0 < area <= small_area:
            w[mask == lab] = boost
    if boundary is not None:
        w[boundary > 0.5] = np.maximum(w[boundary > 0.5], boundary_boost)
    return w


# ============================================================
# Dataset
# ============================================================
class OrganoidDenseDataset(Dataset):
    """
    输出：
      image:    [3,H,W]
      fg:       [1,H,W]
      boundary: [1,H,W]
      center:   [1,H,W]
      weight:   [1,H,W]
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str,
        crop_size: int = 512,
        whole_size: int = 768,
        mixed_scale: bool = True,
        whole_prob: float = 0.35,
        normalize: str = "imagenet",
        min_area: int = 5,
        small_area: int = 256,
        small_boost: float = 3.0,
        center_sigma: float = 5.0,
        augment: bool = True,
        foreground_crop_prob: float = 0.75,
        brightness_jitter: float = 0.10,
        contrast_jitter: float = 0.10,
    ):
        self.data_root = Path(data_root)
        self.split = split
        #self.image_dir = self.data_root / split / "images"
        #self.mask_dir = self.data_root / split / "masks"
        split_map = {
            "train": "Train",
            "val": "Val",
            "test": "Test"
        }
        split_name = split_map.get(split.lower(), split)

        p1_img = self.data_root / split / "images"
        p1_mask = self.data_root / split / "masks"

        p2_img = self.data_root / split_name / "images"
        p2_mask = self.data_root / split_name / "masks"

        p3_img = self.data_root / "images" / split_name
        p3_mask = self.data_root / "masks" / split_name

        p4_img = self.data_root / split_name
        p4_mask = self.data_root / f"{split_name}_masks"

        candidates = [
            (p1_img, p1_mask),
            (p2_img, p2_mask),
            (p3_img, p3_mask),
            (p4_img, p4_mask),
        ]

        for img_dir, mask_dir in candidates:
            if img_dir.exists() and mask_dir.exists():
                self.image_dir = img_dir
                self.mask_dir = mask_dir
                break
        else:
            raise FileNotFoundError(
                f"Cannot find image/mask folders for split={split} under {self.data_root}"
            )
        self.pairs = match_image_mask_pairs(self.image_dir, self.mask_dir)

        self.crop_size = int(crop_size)
        self.whole_size = int(whole_size)
        self.mixed_scale = bool(mixed_scale)
        self.whole_prob = float(whole_prob)
        self.normalize = normalize
        self.min_area = int(min_area)
        self.small_area = int(small_area)
        self.small_boost = float(small_boost)
        self.center_sigma = float(center_sigma)
        self.augment = bool(augment)
        self.foreground_crop_prob = float(foreground_crop_prob)
        self.brightness_jitter = float(brightness_jitter)
        self.contrast_jitter = float(contrast_jitter)

    def __len__(self) -> int:
        return len(self.pairs)

    def _photometric_aug(self, img: np.ndarray) -> np.ndarray:
        if not self.augment:
            return img
        pil = Image.fromarray(img)
        if self.brightness_jitter > 0:
            fac = 1.0 + random.uniform(-self.brightness_jitter, self.brightness_jitter)
            pil = ImageEnhance.Brightness(pil).enhance(fac)
        if self.contrast_jitter > 0:
            fac = 1.0 + random.uniform(-self.contrast_jitter, self.contrast_jitter)
            pil = ImageEnhance.Contrast(pil).enhance(fac)
        return np.asarray(pil, dtype=np.uint8)

    def _geometric_aug(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return img, mask
        # flip
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        if random.random() < 0.5:
            img = np.ascontiguousarray(img[::-1, :])
            mask = np.ascontiguousarray(mask[::-1, :])
        # rotate 90*k
        if random.random() < 0.5:
            k = random.randint(0, 3)
            img = np.ascontiguousarray(np.rot90(img, k))
            mask = np.ascontiguousarray(np.rot90(mask, k))
        return img, mask

    def _pad_if_needed(self, img: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        pad_h = max(0, size - h)
        pad_w = max(0, size - w)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant")
        return img, mask

    def _random_crop(self, img: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        img, mask = self._pad_if_needed(img, mask, size)
        h, w = mask.shape
        if h == size and w == size:
            return img, mask

        use_fg = self.augment and (random.random() < self.foreground_crop_prob) and (mask > 0).any()
        if use_fg:
            ys, xs = np.where(mask > 0)
            idx = random.randrange(len(xs))
            cx, cy = int(xs[idx]), int(ys[idx])
            # 让前景不总在正中心，增加鲁棒性
            off_x = random.randint(size // 4, 3 * size // 4)
            off_y = random.randint(size // 4, 3 * size // 4)
            x1 = int(np.clip(cx - off_x, 0, w - size))
            y1 = int(np.clip(cy - off_y, 0, h - size))
        else:
            x1 = random.randint(0, w - size)
            y1 = random.randint(0, h - size)

        img_c = img[y1:y1 + size, x1:x1 + size]
        mask_c = mask[y1:y1 + size, x1:x1 + size]
        return img_c, mask_c

    def _resize_whole_to_square(self, img: np.ndarray, mask: np.ndarray, size: int) -> Tuple[np.ndarray, np.ndarray]:
        h, w = mask.shape
        scale = size / float(max(h, w))
        nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        img_r = resize_image_np(img, (nh, nw), is_mask=False)
        mask_r = resize_image_np(mask, (nh, nw), is_mask=True)

        out_img = np.zeros((size, size, 3), dtype=np.uint8)
        out_mask = np.zeros((size, size), dtype=np.int32)
        out_img[:nh, :nw] = img_r
        out_mask[:nh, :nw] = mask_r
        return out_img, out_mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ip, mp = self.pairs[idx]
        img = read_rgb(ip)
        mask = read_mask(mp)

        img, mask = self._geometric_aug(img, mask)

        if self.mixed_scale and self.augment and random.random() < self.whole_prob:
            img, mask = self._resize_whole_to_square(img, mask, self.whole_size)
        else:
            img, mask = self._random_crop(img, mask, self.crop_size)

        img = self._photometric_aug(img)

        fg = mask_to_foreground(mask)
        boundary = mask_to_boundary(mask, radius=2)
        centers, _, areas = extract_centers_boxes_areas(mask, min_area=self.min_area)
        sigma = self.center_sigma
        if len(areas) > 0:
            # 根据图像尺寸轻微自适应，避免大图中心点太尖
            sigma = max(self.center_sigma, min(mask.shape) / 160.0)
        center = make_center_heatmap(centers, mask.shape, sigma=sigma)
        weight = make_small_object_weight_map(
            mask, small_area=self.small_area, boost=self.small_boost,
            boundary=boundary, boundary_boost=1.5
        )

        return {
            "image": image_to_tensor(img, normalize=self.normalize),
            "fg": torch.from_numpy(fg)[None].float(),
            "boundary": torch.from_numpy(boundary)[None].float(),
            "center": torch.from_numpy(center)[None].float(),
            "weight": torch.from_numpy(weight)[None].float(),
            "name": ip.stem,
            "num_instances": len(centers),
        }


# ============================================================
# LoRA for true SAM2 Linear layers
# ============================================================
class LoRALinear(nn.Module):
    """
    Wrap an existing nn.Linear as:
      y = frozen_linear(x) + scale * B(A(dropout(x)))
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.lora_A = nn.Linear(self.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, self.out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(self.dropout(x)))


def _get_parent_module(root: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def inject_lora_to_sam2_image_encoder(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_keywords: Sequence[str] = ("q", "k", "v", "qkv", "proj", "attn"),
    verbose: bool = True,
) -> int:
    """
    在 SAM2 image_encoder 内部 Linear 层注入 LoRA。
    为了兼容不同 sam2 版本，不强依赖固定层名。
    默认只处理 image_encoder 下、名字包含 attention 相关关键词的 Linear。
    """
    if hasattr(model, "image_encoder"):
        root = model.image_encoder
        prefix = "image_encoder"
    else:
        root = model
        prefix = ""

    to_replace: List[str] = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear):
            lname = name.lower()
            hit = any(k.lower() in lname for k in target_keywords)
            # 如果名字里没有关键词，但在 blocks/trunk 中，也可作为候选；这里保持保守
            if hit:
                to_replace.append(name)

    replaced = 0
    for name in to_replace:
        parent, child_name = _get_parent_module(root, name)
        old = getattr(parent, child_name)
        if isinstance(old, LoRALinear):
            continue
        if not isinstance(old, nn.Linear):
            continue
        setattr(parent, child_name, LoRALinear(old, rank=rank, alpha=alpha, dropout=dropout))
        replaced += 1

    if verbose:
        print(f"[LoRA] Injected LoRA into {replaced} Linear layers under {prefix or 'model'}")
        if replaced == 0:
            print("[LoRA][Warning] No Linear layers matched. You may need to change --lora_keywords.")
    return replaced


def freeze_all_params(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False


def unfreeze_lora_params(model: nn.Module) -> int:
    n = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            n += p.numel()
    return n


# ============================================================
# SAM2 loading and feature extraction
# ============================================================
def try_import_build_sam2():
    """
    兼容官方 sam2 与部分本地版本。
    """
    errors = []
    candidates = [
        ("sam2.build_sam", "build_sam2"),
        ("sam2.build_sam2", "build_sam2"),
    ]
    for mod_name, fn_name in candidates:
        try:
            mod = __import__(mod_name, fromlist=[fn_name])
            fn = getattr(mod, fn_name)
            return fn
        except Exception as e:
            errors.append(f"{mod_name}.{fn_name}: {repr(e)}")
    raise ImportError(
        "Cannot import build_sam2. Please run this script inside your sam2 environment "
        "and make sure the official SAM2 package is installed or --sam2_root is set.\n"
        + "\n".join(errors)
    )


def maybe_add_sam2_root_to_syspath(sam2_root: str = "") -> None:
    if sam2_root:
        root = Path(sam2_root)
        if root.exists():
            sys.path.insert(0, str(root))
    for p in [Path.cwd()]:
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def load_sam2_model(
    config_file: str,
    ckpt_path: str,
    device: torch.device,
    sam2_root: str = "",
) -> nn.Module:
    maybe_add_sam2_root_to_syspath(sam2_root)
    build_sam2 = try_import_build_sam2()

    # 不同版本 build_sam2 参数略有差别，逐个尝试
    errors = []
    call_patterns = [
        lambda: build_sam2(config_file, ckpt_path, device=device),
        lambda: build_sam2(config_file, ckpt_path),
        lambda: build_sam2(config_file=config_file, ckpt_path=ckpt_path, device=device),
        lambda: build_sam2(config_file=config_file, ckpt_path=ckpt_path),
    ]

    model = None
    for fn in call_patterns:
        try:
            model = fn()
            break
        except Exception as e:
            errors.append(repr(e))

    if model is None:
        raise RuntimeError(
            "Failed to build SAM2 model. Tried multiple build_sam2 signatures:\n" +
            "\n".join(errors)
        )

    model.to(device)
    model.eval()
    return model


def _collect_tensors_from_any(obj: Any) -> List[torch.Tensor]:
    """
    从 dict/list/tuple/tensor 中递归收集 4D feature tensors。
    """
    out: List[torch.Tensor] = []
    if torch.is_tensor(obj):
        if obj.ndim == 4:
            out.append(obj)
    elif isinstance(obj, dict):
        # 优先常见字段
        for key in ["backbone_fpn", "vision_features", "features", "fpn_features"]:
            if key in obj:
                out.extend(_collect_tensors_from_any(obj[key]))
        # 再兜底收集所有字段
        for k, v in obj.items():
            if k in ["backbone_fpn", "vision_features", "features", "fpn_features"]:
                continue
            out.extend(_collect_tensors_from_any(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_collect_tensors_from_any(v))
    return out


def _sort_features_by_resolution(feats: List[torch.Tensor]) -> List[torch.Tensor]:
    # 去重：有些 dict 会重复收集同一个 tensor
    unique: List[torch.Tensor] = []
    seen = set()
    for f in feats:
        key = (id(f), tuple(f.shape))
        if key not in seen:
            unique.append(f)
            seen.add(key)
    # 按 spatial size 从大到小排序
    unique.sort(key=lambda t: int(t.shape[-2]) * int(t.shape[-1]), reverse=True)
    return unique


class SAM2FeatureExtractor(nn.Module):
    """
    真实 SAM2 特征提取包装器：
    - 加载完整 SAM2；
    - 冻结除 LoRA 外的所有参数；
    - forward 时仅取 image encoder / forward_image 输出的 feature maps；
    - 自动解析 backbone_fpn / vision_features 等输出。
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_ckpt: str,
        device: torch.device,
        sam2_root: str = "",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_keywords: Sequence[str] = ("q", "k", "v", "qkv", "proj", "attn"),
        use_no_grad_for_frozen: bool = False,
        verbose: bool = True,
    ):
        super().__init__()
        self.device_ref = device
        self.sam2 = load_sam2_model(sam2_config, sam2_ckpt, device=device, sam2_root=sam2_root)

        freeze_all_params(self.sam2)
        inject_lora_to_sam2_image_encoder(
            self.sam2,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_keywords=lora_keywords,
            verbose=verbose,
        )
        lora_n = unfreeze_lora_params(self.sam2)
        if verbose:
            print(f"[LoRA] Trainable LoRA params in SAM2: {lora_n/1e6:.3f}M")

        self.use_no_grad_for_frozen = bool(use_no_grad_for_frozen)

    def forward_raw(self, x: torch.Tensor) -> Any:
        """
        x: normalized Bx3xHxW
        """
        # 优先官方 SAM2Base.forward_image
        if hasattr(self.sam2, "forward_image"):
            return self.sam2.forward_image(x)
        if hasattr(self.sam2, "image_encoder"):
            return self.sam2.image_encoder(x)
        return self.sam2(x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # 不能整体 no_grad，否则 LoRA 也没梯度；这里只是保留接口。
        out = self.forward_raw(x)
        feats = _collect_tensors_from_any(out)
        feats = _sort_features_by_resolution(feats)

        if len(feats) == 0:
            raise RuntimeError(
                "SAM2 image encoder returned no 4D feature tensors. "
                "Please inspect your SAM2 output and modify SAM2FeatureExtractor.forward()."
            )

        # 只保留合理 feature：batch 维一致，空间维不超过输入
        b, _, h, w = x.shape
        valid = []
        for f in feats:
            if f.shape[0] == b and f.shape[-2] >= 2 and f.shape[-1] >= 2:
                valid.append(f)
        feats = valid

        # 目标需要 1/4, 1/8, 1/16 三层。SAM2 若输出更多，取分辨率最高的三层；
        # 若只有一层，则用插值构造 pyramid，保证训练不直接崩。
        if len(feats) >= 3:
            use = feats[:3]
        elif len(feats) == 2:
            f1, f2 = feats
            f3 = F.avg_pool2d(f2, kernel_size=2, stride=2) if min(f2.shape[-2:]) >= 4 else f2
            use = [f1, f2, f3]
        else:
            f = feats[0]
            # 如果只有最终低分辨率 feature，则上采样/下采样构造 3 层
            f1 = F.interpolate(f, size=(max(4, h // 4), max(4, w // 4)), mode="bilinear", align_corners=False)
            f2 = F.interpolate(f, size=(max(2, h // 8), max(2, w // 8)), mode="bilinear", align_corners=False)
            f3 = F.interpolate(f, size=(max(1, h // 16), max(1, w // 16)), mode="bilinear", align_corners=False)
            use = [f1, f2, f3]

        return use


# ============================================================
# Model blocks: CNN local branch + residual fusion decoder
# ============================================================
class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, norm: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=not norm)
        self.bn = nn.BatchNorm2d(out_ch) if norm else nn.Identity()
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResidualConvBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = ConvBNAct(ch, ch, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x + self.conv2(self.conv1(x)), inplace=True)


class CNNUNetLocalBranch(nn.Module):
    """
    CNN / U-Net local branch:
    返回 c1(1/4), c2(1/8), c3(1/16)
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 32):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(in_ch, base_ch, 3, 1, 1),
            ResidualConvBlock(base_ch),
        )
        self.down1 = ConvBNAct(base_ch, base_ch * 2, 3, 2, 1)       # 1/2
        self.enc1 = nn.Sequential(ResidualConvBlock(base_ch * 2), ResidualConvBlock(base_ch * 2))

        self.down2 = ConvBNAct(base_ch * 2, base_ch * 4, 3, 2, 1)   # 1/4
        self.enc2 = nn.Sequential(ResidualConvBlock(base_ch * 4), ResidualConvBlock(base_ch * 4))

        self.down3 = ConvBNAct(base_ch * 4, base_ch * 8, 3, 2, 1)   # 1/8
        self.enc3 = nn.Sequential(ResidualConvBlock(base_ch * 8), ResidualConvBlock(base_ch * 8))

        self.down4 = ConvBNAct(base_ch * 8, base_ch * 16, 3, 2, 1)  # 1/16
        self.enc4 = nn.Sequential(ResidualConvBlock(base_ch * 16), ResidualConvBlock(base_ch * 16))

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x0 = self.stem(x)
        x1 = self.enc1(self.down1(x0))
        c1 = self.enc2(self.down2(x1))
        c2 = self.enc3(self.down3(c1))
        c3 = self.enc4(self.down4(c2))
        return [c1, c2, c3]


class LazyFeatureProjector(nn.Module):
    """
    Lazy 1x1 projection. 真实 SAM2 的 feature channel 可能是 256/768/1024 等，
    用 LazyConv2d 避免手工写死通道。
    """
    def __init__(self, out_ch: int):
        super().__init__()
        self.proj = nn.LazyConv2d(out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.proj(x)))


class SmallOrganoidEnhancementHead(nn.Module):
    """
    浅层 1/4 feature 的小目标增强门控。
    """
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNAct(ch, ch, 3, 1, 1),
            nn.Conv2d(ch, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, shallow: torch.Tensor) -> torch.Tensor:
        return self.net(shallow)


class MultiScaleResidualFusionDecoder(nn.Module):
    """
    SAM2 multi-level features + CNN multi-level features -> dense heads.
    """
    def __init__(self, cnn_chs: Sequence[int], dec_ch: int = 128):
        super().__init__()
        self.dec_ch = int(dec_ch)

        # SAM channel unknown -> lazy projectors
        self.sam_proj4 = LazyFeatureProjector(dec_ch)
        self.sam_proj8 = LazyFeatureProjector(dec_ch)
        self.sam_proj16 = LazyFeatureProjector(dec_ch)

        self.cnn_proj4 = nn.Conv2d(cnn_chs[0], dec_ch, 1, bias=False)
        self.cnn_proj8 = nn.Conv2d(cnn_chs[1], dec_ch, 1, bias=False)
        self.cnn_proj16 = nn.Conv2d(cnn_chs[2], dec_ch, 1, bias=False)

        self.fuse16 = nn.Sequential(ConvBNAct(dec_ch, dec_ch), ResidualConvBlock(dec_ch))
        self.fuse8 = nn.Sequential(ConvBNAct(dec_ch, dec_ch), ResidualConvBlock(dec_ch))
        self.fuse4 = nn.Sequential(ConvBNAct(dec_ch, dec_ch), ResidualConvBlock(dec_ch))

        self.small_enhance = SmallOrganoidEnhancementHead(dec_ch)

        self.refine2 = nn.Sequential(
            ConvBNAct(dec_ch, dec_ch // 2),
            ResidualConvBlock(dec_ch // 2),
        )
        self.refine1 = nn.Sequential(
            ConvBNAct(dec_ch // 2, dec_ch // 4),
            ResidualConvBlock(dec_ch // 4),
        )
        out_ch = dec_ch // 4

        self.fg_head = nn.Conv2d(out_ch, 1, 1)
        self.boundary_head = nn.Conv2d(out_ch, 1, 1)
        self.center_head = nn.Conv2d(out_ch, 1, 1)

    @staticmethod
    def _resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, sam_feats: Sequence[torch.Tensor], cnn_feats: Sequence[torch.Tensor], out_hw: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        # SAM feats 可能不是刚好 1/4,1/8,1/16，但按分辨率从大到小排列。
        s4, s8, s16 = sam_feats[:3]
        c4, c8, c16 = cnn_feats[:3]

        p4 = self.sam_proj4(self._resize_to(s4, c4)) + self.cnn_proj4(c4)
        p8 = self.sam_proj8(self._resize_to(s8, c8)) + self.cnn_proj8(c8)
        p16 = self.sam_proj16(self._resize_to(s16, c16)) + self.cnn_proj16(c16)

        x16 = self.fuse16(p16)
        x8 = self.fuse8(p8 + F.interpolate(x16, size=p8.shape[-2:], mode="bilinear", align_corners=False))
        x4 = self.fuse4(p4 + F.interpolate(x8, size=p4.shape[-2:], mode="bilinear", align_corners=False))

        gate = self.small_enhance(x4)
        x4 = x4 * (1.0 + gate)

        x = F.interpolate(x4, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.refine2(x)
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        x = self.refine1(x)

        if x.shape[-2:] != out_hw:
            x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)

        return {
            "fg": self.fg_head(x),
            "boundary": self.boundary_head(x),
            "center": self.center_head(x),
        }


class PromptFreeOrganoidSAM2(nn.Module):
    """
    完整无 prompt box 版 Organoid-SAM2。
    """

    def __init__(
        self,
        sam2_config: str,
        sam2_ckpt: str,
        device: torch.device,
        sam2_root: str = "",
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_keywords: Sequence[str] = ("q", "k", "v", "qkv", "proj", "attn"),
        cnn_base_ch: int = 32,
        dec_ch: int = 128,
    ):
        super().__init__()
        self.sam2_encoder = SAM2FeatureExtractor(
            sam2_config=sam2_config,
            sam2_ckpt=sam2_ckpt,
            device=device,
            sam2_root=sam2_root,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_keywords=lora_keywords,
            verbose=True,
        )
        self.cnn_branch = CNNUNetLocalBranch(in_ch=3, base_ch=cnn_base_ch)
        cnn_chs = [cnn_base_ch * 4, cnn_base_ch * 8, cnn_base_ch * 16]
        self.decoder = MultiScaleResidualFusionDecoder(cnn_chs=cnn_chs, dec_ch=dec_ch)

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        out_hw = tuple(image.shape[-2:])
        sam_feats = self.sam2_encoder(image)
        cnn_feats = self.cnn_branch(image)
        return self.decoder(sam_feats, cnn_feats, out_hw=out_hw)


# ============================================================
# Loss and metrics
# ============================================================
def dice_loss_with_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (prob * target).sum(dim=dims)
    den = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (den + eps)
    return 1.0 - dice.mean()


def weighted_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def focal_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.25,
    weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    prob = torch.sigmoid(logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * ((1.0 - p_t).clamp(min=1e-6) ** gamma) * bce
    if weight is not None:
        loss = loss * weight
    return loss.mean()


def total_loss_fn(
    pred: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    lambda_boundary: float = 0.5,
    lambda_center: float = 0.25,
    use_focal: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    fg_target = batch["fg"]
    bd_target = batch["boundary"]
    center_target = batch["center"]
    weight = batch["weight"]

    if use_focal:
        fg_bce = focal_bce_with_logits(pred["fg"], fg_target, weight=weight)
    else:
        fg_bce = weighted_bce_with_logits(pred["fg"], fg_target, weight=weight)

    fg_dice = dice_loss_with_logits(pred["fg"], fg_target)
    fg_loss = fg_dice + fg_bce

    boundary_loss = weighted_bce_with_logits(pred["boundary"], bd_target, weight=None)
    center_loss = F.mse_loss(torch.sigmoid(pred["center"]), center_target)

    total = fg_loss + lambda_boundary * boundary_loss + lambda_center * center_loss
    info = {
        "total": float(total.detach().cpu()),
        "fg": float(fg_loss.detach().cpu()),
        "fg_dice": float(fg_dice.detach().cpu()),
        "fg_bce": float(fg_bce.detach().cpu()),
        "boundary": float(boundary_loss.detach().cpu()),
        "center": float(center_loss.detach().cpu()),
    }
    return total, info


@torch.no_grad()
def semantic_metrics_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    thresh: float = 0.5,
) -> Dict[str, float]:
    prob = torch.sigmoid(logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()

    dims = (1, 2, 3)
    tp = (pred * target).sum(dim=dims)
    fp = (pred * (1.0 - target)).sum(dim=dims)
    fn = ((1.0 - pred) * target).sum(dim=dims)

    dice = ((2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)).mean().item()
    iou = ((tp + 1e-6) / (tp + fp + fn + 1e-6)).mean().item()
    precision = ((tp + 1e-6) / (tp + fp + 1e-6)).mean().item()
    recall = ((tp + 1e-6) / (tp + fn + 1e-6)).mean().item()
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(precision),
        "recall": float(recall),
    }


# ============================================================
# Training / validation
# ============================================================
def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def optimizer_parameter_groups(model: nn.Module, lr: float, lora_lr: float, weight_decay: float):
    lora_params = []
    other_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "lora_A" in name or "lora_B" in name:
            lora_params.append(p)
        else:
            other_params.append(p)

    groups = []
    if other_params:
        groups.append({"params": other_params, "lr": lr, "weight_decay": weight_decay})
    if lora_params:
        groups.append({"params": lora_params, "lr": lora_lr, "weight_decay": weight_decay})
    return groups


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def initialize_lazy_modules(model: nn.Module, device: torch.device, img_size: int = 256) -> None:
    """
    LazyConv2d 需要先前向一次后才能创建参数，再构建 optimizer。
    """
    model.eval()
    with torch.no_grad():
        dummy = torch.zeros(1, 3, img_size, img_size, device=device)
        _ = model(dummy)
    model.train()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    running = {
        "loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    n = 0
    t0 = time.time()

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        image = batch["image"]

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(image)
            loss, loss_info = total_loss_fn(
                pred, batch,
                lambda_boundary=args.lambda_boundary,
                lambda_center=args.lambda_center,
                use_focal=args.use_focal,
            )

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        metrics = semantic_metrics_from_logits(pred["fg"].detach(), batch["fg"].detach(), thresh=args.mask_thresh)
        bs = image.size(0)
        running["loss"] += loss.item() * bs
        for k in ["iou", "dice", "precision", "recall"]:
            running[k] += metrics[k] * bs
        n += bs

        if step % args.print_freq == 0:
            dt = time.time() - t0
            print(
                f"[Train] epoch={epoch:03d} step={step:04d}/{len(loader)} "
                f"loss={running['loss']/max(n,1):.4f} "
                f"IoU={running['iou']/max(n,1):.4f} Dice={running['dice']/max(n,1):.4f} "
                f"Prec={running['precision']/max(n,1):.4f} Rec={running['recall']/max(n,1):.4f} "
                f"time={dt:.1f}s"
            )

    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.eval()
    running = {
        "loss": 0.0,
        "iou": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        image = batch["image"]
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(image)
            loss, _ = total_loss_fn(
                pred, batch,
                lambda_boundary=args.lambda_boundary,
                lambda_center=args.lambda_center,
                use_focal=args.use_focal,
            )

        metrics = semantic_metrics_from_logits(pred["fg"], batch["fg"], thresh=args.mask_thresh)
        bs = image.size(0)
        running["loss"] += loss.item() * bs
        for k in ["iou", "dice", "precision", "recall"]:
            running[k] += metrics[k] * bs
        n += bs

    return {k: v / max(n, 1) for k, v in running.items()}


@torch.no_grad()
def save_debug_visuals(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    args: argparse.Namespace,
    max_items: int = 4,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        image = batch["image"]
        pred = model(image)
        prob = torch.sigmoid(pred["fg"]).detach().cpu()
        bd = torch.sigmoid(pred["boundary"]).detach().cpu()
        cen = torch.sigmoid(pred["center"]).detach().cpu()
        gt = batch["fg"].detach().cpu()

        # 反归一化只做近似可视化
        img = image.detach().cpu()
        for i in range(img.size(0)):
            if count >= max_items:
                return
            im = img[i].clone()
            if args.normalize == "imagenet":
                mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
                std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
                im = im * std + mean
            elif args.normalize == "sam":
                mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None] / 255.0
                std = torch.tensor([58.395, 57.12, 57.375])[:, None, None] / 255.0
                im = im * std + mean
            im = im.clamp(0, 1)

            im_np = (im.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt_np = (gt[i, 0].numpy() * 255).astype(np.uint8)
            pr_np = (prob[i, 0].numpy() * 255).astype(np.uint8)
            bd_np = (bd[i, 0].numpy() * 255).astype(np.uint8)
            cen_np = (cen[i, 0].numpy() * 255).astype(np.uint8)

            # 横向拼接：image, gt, pred, boundary, center
            h, w = gt_np.shape
            panels = [
                im_np,
                np.repeat(gt_np[..., None], 3, axis=2),
                np.repeat(pr_np[..., None], 3, axis=2),
                np.repeat(bd_np[..., None], 3, axis=2),
                np.repeat(cen_np[..., None], 3, axis=2),
            ]
            canvas = np.concatenate(panels, axis=1)
            Image.fromarray(canvas).save(out_dir / f"debug_{count:03d}.png")
            count += 1


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_dice: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "best_dice": best_dice,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    if scheduler is not None:
        ckpt["scheduler"] = scheduler.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    device: Union[str, torch.device] = "cpu",
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"[Resume] missing keys: {len(missing)}")
    if unexpected:
        print(f"[Resume] unexpected keys: {len(unexpected)}")
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    return int(ckpt.get("epoch", 0)) + 1, float(ckpt.get("best_dice", -1.0))


# ============================================================
# Argument parsing
# ============================================================
def parse_keywords(s: str) -> Tuple[str, ...]:
    return tuple([x.strip() for x in s.split(",") if x.strip()])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Prompt-free Organoid-SAM2 full training")

    # paths
    p.add_argument("--data_root", type=str, default=DEFAULT_DATA_ROOT)
    p.add_argument("--save_dir", type=str, default=DEFAULT_SAVE_DIR)
    p.add_argument("--sam2_config", type=str, default=DEFAULT_SAM2_CONFIG)
    p.add_argument("--sam2_ckpt", type=str, default=DEFAULT_SAM2_CKPT)
    p.add_argument("--sam2_root", type=str, default="")

    # train
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--print_freq", type=int, default=20)
    p.add_argument("--resume", type=str, default="")

    # data
    p.add_argument("--crop_size", type=int, default=512)
    p.add_argument("--whole_size", type=int, default=768)
    p.add_argument("--mixed_scale", action="store_true", default=True)
    p.add_argument("--no_mixed_scale", action="store_true")
    p.add_argument("--whole_prob", type=float, default=0.35)
    p.add_argument("--normalize", type=str, default="sam", choices=["sam", "imagenet", "none"])
    p.add_argument("--min_area", type=int, default=5)
    p.add_argument("--small_area", type=int, default=256)
    p.add_argument("--small_boost", type=float, default=3.0)
    p.add_argument("--center_sigma", type=float, default=5.0)
    p.add_argument("--foreground_crop_prob", type=float, default=0.75)

    # model
    p.add_argument("--lora_rank", type=int, default=8)
    p.add_argument("--lora_alpha", type=float, default=16.0)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_keywords", type=str, default="q,k,v,qkv,proj,attn")
    p.add_argument("--cnn_base_ch", type=int, default=32)
    p.add_argument("--dec_ch", type=int, default=128)
    p.add_argument("--lazy_init_size", type=int, default=256)

    # loss
    p.add_argument("--lambda_boundary", type=float, default=0.5)
    p.add_argument("--lambda_center", type=float, default=0.25)
    p.add_argument("--use_focal", action="store_true")
    p.add_argument("--mask_thresh", type=float, default=0.5)

    # debug
    p.add_argument("--save_debug_every", type=int, default=5)
    p.add_argument("--debug_items", type=int, default=4)

    args = p.parse_args()
    if args.no_mixed_scale:
        args.mixed_scale = False
    return args


# ============================================================
# Main
# ============================================================
def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    save_dir = ensure_dir(args.save_dir)
    (save_dir / "args.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"[Start] {now_str()}")
    print(f"[Device] {device}")
    print(f"[Data root] {args.data_root}")
    print(f"[Save dir] {save_dir}")
    print(f"[SAM2 config] {args.sam2_config}")
    print(f"[SAM2 ckpt] {args.sam2_ckpt}")
    print("[Mode] Prompt-free dense prediction: no box prompt, no point prompt, no prompt encoder.")
    print("=" * 80)

    train_ds = OrganoidDenseDataset(
        data_root=args.data_root,
        split="train",
        crop_size=args.crop_size,
        whole_size=args.whole_size,
        mixed_scale=args.mixed_scale,
        whole_prob=args.whole_prob,
        normalize=args.normalize,
        min_area=args.min_area,
        small_area=args.small_area,
        small_boost=args.small_boost,
        center_sigma=args.center_sigma,
        augment=True,
        foreground_crop_prob=args.foreground_crop_prob,
    )
    val_ds = OrganoidDenseDataset(
        data_root=args.data_root,
        split="val",
        crop_size=args.crop_size,
        whole_size=args.whole_size,
        mixed_scale=False,
        whole_prob=0.0,
        normalize=args.normalize,
        min_area=args.min_area,
        small_area=args.small_area,
        small_boost=args.small_boost,
        center_sigma=args.center_sigma,
        augment=False,
        foreground_crop_prob=0.0,
    )

    print(f"[Dataset] train={len(train_ds)} val={len(val_ds)}")
    print(f"[Dataset] first train pair: {train_ds.pairs[0][0].name} | {train_ds.pairs[0][1].name}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = PromptFreeOrganoidSAM2(
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
        device=device,
        sam2_root=args.sam2_root,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_keywords=parse_keywords(args.lora_keywords),
        cnn_base_ch=args.cnn_base_ch,
        dec_ch=args.dec_ch,
    ).to(device)

    print("[Init] Initializing lazy projection layers...")
    initialize_lazy_modules(model, device=device, img_size=args.lazy_init_size)
    total_params, trainable_params = count_parameters(model)
    print(f"[Params] total={total_params/1e6:.2f}M trainable={trainable_params/1e6:.2f}M")

    groups = optimizer_parameter_groups(
        model,
        lr=args.lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    best_dice = -1.0
    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        start_epoch, best_dice = load_checkpoint(args.resume, model, optimizer, scheduler, device=device)
        print(f"[Resume] start_epoch={start_epoch}, best_dice={best_dice:.4f}")

    log_path = save_dir / "train_log.csv"
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "epoch",
                "train_loss", "train_iou", "train_dice", "train_precision", "train_recall",
                "val_loss", "val_iou", "val_dice", "val_precision", "val_recall",
                "best_dice", "lr", "time"
            ])

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()

        train_stats = train_one_epoch(model, train_loader, optimizer, scaler, device, epoch, args)
        val_stats = validate(model, val_loader, device, args)

        scheduler.step()

        improved = val_stats["dice"] > best_dice
        if improved:
            best_dice = val_stats["dice"]
            save_checkpoint(save_dir / "best_model.pth", model, optimizer, scheduler, epoch, best_dice, args)

        save_checkpoint(save_dir / "last_model.pth", model, optimizer, scheduler, epoch, best_dice, args)

        if args.save_debug_every > 0 and (epoch == 1 or epoch % args.save_debug_every == 0):
            debug_dir = save_dir / "debug_vis" / f"epoch_{epoch:03d}"
            try:
                save_debug_visuals(model, val_loader, device, debug_dir, args, max_items=args.debug_items)
            except Exception as e:
                print(f"[DebugVis][Warning] failed to save debug visuals: {repr(e)}")

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_t0

        print(
            f"\nEpoch [{epoch:03d}/{args.epochs}] "
            f"Train: loss={train_stats['loss']:.4f}, IoU={train_stats['iou']:.4f}, "
            f"Dice={train_stats['dice']:.4f}, P={train_stats['precision']:.4f}, R={train_stats['recall']:.4f} | "
            f"Val: loss={val_stats['loss']:.4f}, IoU={val_stats['iou']:.4f}, "
            f"Dice={val_stats['dice']:.4f}, P={val_stats['precision']:.4f}, R={val_stats['recall']:.4f} | "
            f"BestDice={best_dice:.4f}{' *' if improved else ''} | "
            f"lr={lr_now:.2e} | time={epoch_time:.1f}s\n"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_stats["loss"], train_stats["iou"], train_stats["dice"], train_stats["precision"], train_stats["recall"],
                val_stats["loss"], val_stats["iou"], val_stats["dice"], val_stats["precision"], val_stats["recall"],
                best_dice, lr_now, epoch_time,
            ])

        # 清理显存碎片
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print("=" * 80)
    print(f"[Done] {now_str()}")
    print(f"[Best Dice] {best_dice:.4f}")
    print(f"[Checkpoints] {save_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
