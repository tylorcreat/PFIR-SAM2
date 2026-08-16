# -*- coding: utf-8 -*-
"""
train_pfir_frontend_ablation.py

PFIR-SAM2 前端结构消融训练脚本。

依赖：
    与 train_promptfree_organoid_sam2_full.py 放在同一目录运行。

支持的变体：
    sam2_only   : LoRA-SAM2 分支 + 原有层级残差解码/三线索头；关闭 CNN 分支
    cnn_only    : CNN 局部分支 + 原有层级残差解码/三线索头；关闭 SAM2 分支
    dual_simple : LoRA-SAM2 + CNN；用 concat + 1x1 conv 和非残差尺度融合替代原多尺度残差融合
    full        : 原始完整 PFIR-SAM2，仅用于复现核对，通常无需重新训练

公平性原则：
1. 自动读取正式 full-model checkpoint 中保存的 args，并作为共同训练配置默认值；
2. 除指定前端结构外，数据、损失、训练轮数、优化器、LoRA、验证规则保持一致；
3. 每个变体独立从同一 SAM2 预训练 checkpoint 初始化，不加载完整 PFIR-SAM2 的已训练权重；
4. best_model.pth 仍按验证集 foreground Dice 选择，与原脚本一致；
5. 训练只输出 dense foreground/boundary/center cues。实例重建与最终统一评估必须使用
   已冻结的 Val 参数和统一 evaluator，不能在 Test 上重新调参。

推荐先运行 dry-run：
    python train_pfir_frontend_ablation.py --variant sam2_only --dry_run --num_workers 0
    python train_pfir_frontend_ablation.py --variant cnn_only --dry_run --num_workers 0
    python train_pfir_frontend_ablation.py --variant dual_simple --dry_run --num_workers 0

正式训练：
    python train_pfir_frontend_ablation.py --variant sam2_only --num_workers 0
    python train_pfir_frontend_ablation.py --variant cnn_only --num_workers 0
    python train_pfir_frontend_ablation.py --variant dual_simple --num_workers 0
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from . import core_training as core
except Exception as exc:
    raise ImportError(
        "Cannot import the formal PFIR-SAM2 training implementation."
    ) from exc


DEFAULT_REFERENCE_CKPT = "weights/PFIR-SAM2_OrganoID_best_model.pth"
DEFAULT_SAVE_ROOT = "outputs/training"
VARIANTS = ("sam2_only", "cnn_only", "dual_simple", "full")


# ============================================================
# 通用工具
# ============================================================
def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def torch_load_full(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """兼容 PyTorch 2.6 的 weights_only 默认行为。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def bool_default(ref: Mapping[str, Any], key: str, fallback: bool) -> bool:
    value = ref.get(key, fallback)
    return bool(value)


def ref_default(ref: Mapping[str, Any], key: str, fallback: Any) -> Any:
    value = ref.get(key, fallback)
    return fallback if value is None else value


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def set_reproducibility(seed: int, deterministic: bool) -> None:
    core.seed_everything(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


# ============================================================
# 正式 full checkpoint 审计
# ============================================================
def audit_reference_checkpoint(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"正式参考 checkpoint 不存在：{path}")

    ckpt = torch_load_full(path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("参考 checkpoint 不是 dict，无法确认其来源结构。")

    state = ckpt.get("model")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("参考 checkpoint 缶少非空的 'model' state_dict。")

    keys = list(state.keys())
    required_groups = {
        "SAM2 encoder": (
            "sam2_encoder.",
        ),
        "CNN local branch": (
            "cnn_branch.",
        ),
        "SAM projection": (
            "decoder.sam_proj4.",
            "decoder.sam_proj8.",
            "decoder.sam_proj16.",
        ),
        "CNN projection": (
            "decoder.cnn_proj4.",
            "decoder.cnn_proj8.",
            "decoder.cnn_proj16.",
        ),
        "foreground head": (
            "decoder.fg_head.",
        ),
        "boundary head": (
            "decoder.boundary_head.",
        ),
        "center head": (
            "decoder.center_head.",
        ),
    }

    missing_groups = []
    matched = {}
    for group, prefixes in required_groups.items():
        found = [p for p in prefixes if any(k.startswith(p) for k in keys)]
        matched[group] = found
        if not found:
            missing_groups.append(group)

    if missing_groups:
        raise RuntimeError(
            "参考 checkpoint 的 state_dict 与上传的原始训练脚本结构不匹配。\n"
            f"缺失结构组：{missing_groups}\n"
            "请不要继续消融训练，先确认 checkpoint 来源。"
        )

    saved_args = ckpt.get("args", {})
    if not isinstance(saved_args, dict):
        saved_args = {}

    audit = {
        "status": "PASS",
        "checkpoint": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "modified_time": time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)
        ),
        "top_level_keys": sorted(ckpt.keys()),
        "epoch": ckpt.get("epoch"),
        "best_dice": ckpt.get("best_dice"),
        "state_dict_key_count": len(keys),
        "matched_architecture_groups": matched,
        "saved_args": saved_args,
    }
    return saved_args, audit


# ============================================================
# 解码器公共尾部
# ============================================================
class DenseCueOutputTail(nn.Module):
    """
    保持原始完整模型的小目标增强、两级上采样和三个 dense cue heads。
    """

    def __init__(self, dec_ch: int):
        super().__init__()
        self.small_enhance = core.SmallOrganoidEnhancementHead(dec_ch)
        self.refine2 = nn.Sequential(
            core.ConvBNAct(dec_ch, dec_ch // 2),
            core.ResidualConvBlock(dec_ch // 2),
        )
        self.refine1 = nn.Sequential(
            core.ConvBNAct(dec_ch // 2, dec_ch // 4),
            core.ResidualConvBlock(dec_ch // 4),
        )
        out_ch = dec_ch // 4
        self.fg_head = nn.Conv2d(out_ch, 1, 1)
        self.boundary_head = nn.Conv2d(out_ch, 1, 1)
        self.center_head = nn.Conv2d(out_ch, 1, 1)

    def forward(
        self,
        x4: torch.Tensor,
        out_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        gate = self.small_enhance(x4)
        x4 = x4 * (1.0 + gate)

        x = F.interpolate(
            x4, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        x = self.refine2(x)
        x = F.interpolate(
            x, scale_factor=2.0, mode="bilinear", align_corners=False
        )
        x = self.refine1(x)

        if x.shape[-2:] != out_hw:
            x = F.interpolate(
                x, size=out_hw, mode="bilinear", align_corners=False
            )

        return {
            "fg": self.fg_head(x),
            "boundary": self.boundary_head(x),
            "center": self.center_head(x),
        }


# ============================================================
# SAM2-only / CNN-only
# ============================================================
class SingleBranchResidualDecoder(nn.Module):
    """
    单分支消融：
    - 只移除另一前端分支；
    - 保留原模型的 coarse-to-fine 残差解码、small enhancement 和三线索头；
    - 因此差异集中在 SAM2/CNN 分支是否存在。
    """

    def __init__(
        self,
        branch: str,
        cnn_chs: Sequence[int],
        dec_ch: int = 128,
    ):
        super().__init__()
        if branch not in ("sam2", "cnn"):
            raise ValueError(f"Unknown branch: {branch}")
        self.branch = branch
        self.dec_ch = int(dec_ch)

        if branch == "sam2":
            self.proj4 = core.LazyFeatureProjector(dec_ch)
            self.proj8 = core.LazyFeatureProjector(dec_ch)
            self.proj16 = core.LazyFeatureProjector(dec_ch)
        else:
            self.proj4 = nn.Conv2d(cnn_chs[0], dec_ch, 1, bias=False)
            self.proj8 = nn.Conv2d(cnn_chs[1], dec_ch, 1, bias=False)
            self.proj16 = nn.Conv2d(cnn_chs[2], dec_ch, 1, bias=False)

        # 与 full decoder 的层级残差路径保持一致
        self.fuse16 = nn.Sequential(
            core.ConvBNAct(dec_ch, dec_ch),
            core.ResidualConvBlock(dec_ch),
        )
        self.fuse8 = nn.Sequential(
            core.ConvBNAct(dec_ch, dec_ch),
            core.ResidualConvBlock(dec_ch),
        )
        self.fuse4 = nn.Sequential(
            core.ConvBNAct(dec_ch, dec_ch),
            core.ResidualConvBlock(dec_ch),
        )
        self.tail = DenseCueOutputTail(dec_ch)

    def forward(
        self,
        feats: Sequence[torch.Tensor],
        out_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if len(feats) < 3:
            raise RuntimeError(f"{self.branch} branch returned fewer than 3 feature maps")

        f4, f8, f16 = feats[:3]
        p4 = self.proj4(f4)
        p8 = self.proj8(f8)
        p16 = self.proj16(f16)

        x16 = self.fuse16(p16)
        x8 = self.fuse8(
            p8
            + F.interpolate(
                x16, size=p8.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        x4 = self.fuse4(
            p4
            + F.interpolate(
                x8, size=p4.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return self.tail(x4, out_hw=out_hw)


# ============================================================
# Dual branch + simple fusion
# ============================================================
class DualSimpleFusionDecoder(nn.Module):
    """
    双分支简单融合消融：
    1. 每个尺度分别将 SAM2/CNN 投影到 dec_ch；
    2. 对同尺度特征 concat，再用 1x1 conv 压缩；
    3. coarse-to-fine 融合仅使用普通 ConvBNAct，不使用 ResidualConvBlock；
    4. 输出 refinement、small enhancement 和三个 heads 与 full 保持一致。

    因此主要对照：
    simple concat fusion vs. 原始 multi-scale residual fusion。
    """

    def __init__(self, cnn_chs: Sequence[int], dec_ch: int = 128):
        super().__init__()
        self.dec_ch = int(dec_ch)

        self.sam_proj4 = core.LazyFeatureProjector(dec_ch)
        self.sam_proj8 = core.LazyFeatureProjector(dec_ch)
        self.sam_proj16 = core.LazyFeatureProjector(dec_ch)

        self.cnn_proj4 = nn.Conv2d(cnn_chs[0], dec_ch, 1, bias=False)
        self.cnn_proj8 = nn.Conv2d(cnn_chs[1], dec_ch, 1, bias=False)
        self.cnn_proj16 = nn.Conv2d(cnn_chs[2], dec_ch, 1, bias=False)

        self.pair_fuse4 = nn.Sequential(
            nn.Conv2d(dec_ch * 2, dec_ch, 1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.SiLU(inplace=True),
        )
        self.pair_fuse8 = nn.Sequential(
            nn.Conv2d(dec_ch * 2, dec_ch, 1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.SiLU(inplace=True),
        )
        self.pair_fuse16 = nn.Sequential(
            nn.Conv2d(dec_ch * 2, dec_ch, 1, bias=False),
            nn.BatchNorm2d(dec_ch),
            nn.SiLU(inplace=True),
        )

        # 明确使用非残差的尺度融合
        self.fuse16 = core.ConvBNAct(dec_ch, dec_ch)
        self.fuse8 = core.ConvBNAct(dec_ch, dec_ch)
        self.fuse4 = core.ConvBNAct(dec_ch, dec_ch)

        self.tail = DenseCueOutputTail(dec_ch)

    @staticmethod
    def _resize_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(
            x, size=ref.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self,
        sam_feats: Sequence[torch.Tensor],
        cnn_feats: Sequence[torch.Tensor],
        out_hw: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if len(sam_feats) < 3 or len(cnn_feats) < 3:
            raise RuntimeError("dual_simple requires three SAM2 and three CNN feature maps")

        s4, s8, s16 = sam_feats[:3]
        c4, c8, c16 = cnn_feats[:3]

        s4 = self.sam_proj4(self._resize_to(s4, c4))
        s8 = self.sam_proj8(self._resize_to(s8, c8))
        s16 = self.sam_proj16(self._resize_to(s16, c16))

        c4 = self.cnn_proj4(c4)
        c8 = self.cnn_proj8(c8)
        c16 = self.cnn_proj16(c16)

        p4 = self.pair_fuse4(torch.cat([s4, c4], dim=1))
        p8 = self.pair_fuse8(torch.cat([s8, c8], dim=1))
        p16 = self.pair_fuse16(torch.cat([s16, c16], dim=1))

        x16 = self.fuse16(p16)
        x8 = self.fuse8(
            p8
            + F.interpolate(
                x16, size=p8.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        x4 = self.fuse4(
            p4
            + F.interpolate(
                x8, size=p4.shape[-2:], mode="bilinear", align_corners=False
            )
        )
        return self.tail(x4, out_hw=out_hw)


# ============================================================
# 消融模型
# ============================================================
class FrontendAblationModel(nn.Module):
    def __init__(
        self,
        variant: str,
        sam2_config: str,
        sam2_ckpt: str,
        device: torch.device,
        sam2_root: str,
        lora_rank: int,
        lora_alpha: float,
        lora_dropout: float,
        lora_keywords: Sequence[str],
        cnn_base_ch: int,
        dec_ch: int,
    ):
        super().__init__()
        if variant not in ("sam2_only", "cnn_only", "dual_simple"):
            raise ValueError(f"Unsupported ablation variant: {variant}")

        self.variant = variant
        self.sam2_encoder: Optional[nn.Module] = None
        self.cnn_branch: Optional[nn.Module] = None

        cnn_chs = [
            cnn_base_ch * 4,
            cnn_base_ch * 8,
            cnn_base_ch * 16,
        ]

        if variant in ("sam2_only", "dual_simple"):
            self.sam2_encoder = core.SAM2FeatureExtractor(
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

        if variant in ("cnn_only", "dual_simple"):
            self.cnn_branch = core.CNNUNetLocalBranch(
                in_ch=3,
                base_ch=cnn_base_ch,
            )

        if variant == "sam2_only":
            self.decoder = SingleBranchResidualDecoder(
                branch="sam2",
                cnn_chs=cnn_chs,
                dec_ch=dec_ch,
            )
        elif variant == "cnn_only":
            self.decoder = SingleBranchResidualDecoder(
                branch="cnn",
                cnn_chs=cnn_chs,
                dec_ch=dec_ch,
            )
        else:
            self.decoder = DualSimpleFusionDecoder(
                cnn_chs=cnn_chs,
                dec_ch=dec_ch,
            )

    def forward(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        out_hw = tuple(image.shape[-2:])

        if self.variant == "sam2_only":
            assert self.sam2_encoder is not None
            sam_feats = self.sam2_encoder(image)
            return self.decoder(sam_feats, out_hw=out_hw)

        if self.variant == "cnn_only":
            assert self.cnn_branch is not None
            cnn_feats = self.cnn_branch(image)
            return self.decoder(cnn_feats, out_hw=out_hw)

        assert self.sam2_encoder is not None
        assert self.cnn_branch is not None
        sam_feats = self.sam2_encoder(image)
        cnn_feats = self.cnn_branch(image)
        return self.decoder(sam_feats, cnn_feats, out_hw=out_hw)


def build_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    if args.variant == "full":
        return core.PromptFreeOrganoidSAM2(
            sam2_config=args.sam2_config,
            sam2_ckpt=args.sam2_ckpt,
            device=device,
            sam2_root=args.sam2_root,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lora_keywords=core.parse_keywords(args.lora_keywords),
            cnn_base_ch=args.cnn_base_ch,
            dec_ch=args.dec_ch,
        )

    return FrontendAblationModel(
        variant=args.variant,
        sam2_config=args.sam2_config,
        sam2_ckpt=args.sam2_ckpt,
        device=device,
        sam2_root=args.sam2_root,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_keywords=core.parse_keywords(args.lora_keywords),
        cnn_base_ch=args.cnn_base_ch,
        dec_ch=args.dec_ch,
    )


# ============================================================
# 参数解析：默认值从正式 full checkpoint 的 args 恢复
# ============================================================
def build_parser(ref: Mapping[str, Any]) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "PFIR-SAM2 frontend ablation training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--variant", required=True, choices=VARIANTS)
    p.add_argument("--reference_ckpt", default=DEFAULT_REFERENCE_CKPT)
    p.add_argument("--save_root", default=DEFAULT_SAVE_ROOT)
    p.add_argument("--run_name", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)

    # paths
    p.add_argument(
        "--data_root",
        default=ref_default(ref, "data_root", "data/OrganoID"),
    )
    p.add_argument(
        "--sam2_config",
        default=ref_default(
            ref,
            "sam2_config",
            "sam2/configs/sam2.1/sam2.1_hiera_l.yaml",
        ),
    )
    p.add_argument(
        "--sam2_ckpt",
        default=ref_default(
            ref,
            "sam2_ckpt",
            "weights/sam2.1_hiera_large.pt",
        ),
    )
    p.add_argument(
        "--sam2_root",
        default=ref_default(ref, "sam2_root", ""),
    )

    # train
    p.add_argument("--epochs", type=int, default=int(ref_default(ref, "epochs", 100)))
    p.add_argument("--batch_size", type=int, default=int(ref_default(ref, "batch_size", 1)))
    p.add_argument("--num_workers", type=int, default=int(ref_default(ref, "num_workers", 0)))
    p.add_argument("--lr", type=float, default=float(ref_default(ref, "lr", 1e-4)))
    p.add_argument("--lora_lr", type=float, default=float(ref_default(ref, "lora_lr", 1e-4)))
    p.add_argument(
        "--weight_decay",
        type=float,
        default=float(ref_default(ref, "weight_decay", 1e-4)),
    )
    p.add_argument("--seed", type=int, default=int(ref_default(ref, "seed", 42)))
    p.add_argument("--expected_seed", type=int, default=None)
    p.add_argument("--source_config_path", default="")
    p.add_argument("--runtime_args_audit_path", default="")
    p.add_argument("--runtime_args_audit_only", action="store_true")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=bool_default(ref, "amp", True),
    )
    p.add_argument(
        "--grad_clip",
        type=float,
        default=float(ref_default(ref, "grad_clip", 1.0)),
    )
    p.add_argument(
        "--print_freq",
        type=int,
        default=int(ref_default(ref, "print_freq", 20)),
    )

    # data
    p.add_argument(
        "--crop_size",
        type=int,
        default=int(ref_default(ref, "crop_size", 512)),
    )
    p.add_argument(
        "--whole_size",
        type=int,
        default=int(ref_default(ref, "whole_size", 768)),
    )
    p.add_argument(
        "--mixed_scale",
        action=argparse.BooleanOptionalAction,
        default=bool_default(ref, "mixed_scale", True),
    )
    p.add_argument(
        "--whole_prob",
        type=float,
        default=float(ref_default(ref, "whole_prob", 0.35)),
    )
    p.add_argument(
        "--normalize",
        choices=("sam", "imagenet", "none"),
        default=ref_default(ref, "normalize", "sam"),
    )
    p.add_argument("--min_area", type=int, default=int(ref_default(ref, "min_area", 5)))
    p.add_argument(
        "--small_area",
        type=int,
        default=int(ref_default(ref, "small_area", 256)),
    )
    p.add_argument(
        "--small_boost",
        type=float,
        default=float(ref_default(ref, "small_boost", 3.0)),
    )
    p.add_argument(
        "--center_sigma",
        type=float,
        default=float(ref_default(ref, "center_sigma", 5.0)),
    )
    p.add_argument(
        "--foreground_crop_prob",
        type=float,
        default=float(ref_default(ref, "foreground_crop_prob", 0.75)),
    )

    # model
    p.add_argument(
        "--lora_rank",
        type=int,
        default=int(ref_default(ref, "lora_rank", 8)),
    )
    p.add_argument(
        "--lora_alpha",
        type=float,
        default=float(ref_default(ref, "lora_alpha", 16.0)),
    )
    p.add_argument(
        "--lora_dropout",
        type=float,
        default=float(ref_default(ref, "lora_dropout", 0.0)),
    )
    p.add_argument(
        "--lora_keywords",
        default=ref_default(ref, "lora_keywords", "q,k,v,qkv,proj,attn"),
    )
    p.add_argument(
        "--cnn_base_ch",
        type=int,
        default=int(ref_default(ref, "cnn_base_ch", 32)),
    )
    p.add_argument(
        "--dec_ch",
        type=int,
        default=int(ref_default(ref, "dec_ch", 128)),
    )
    p.add_argument(
        "--lazy_init_size",
        type=int,
        default=int(ref_default(ref, "lazy_init_size", 256)),
    )

    # loss
    p.add_argument(
        "--lambda_boundary",
        type=float,
        default=float(ref_default(ref, "lambda_boundary", 0.5)),
    )
    p.add_argument(
        "--lambda_center",
        type=float,
        default=float(ref_default(ref, "lambda_center", 0.25)),
    )
    p.add_argument(
        "--use_focal",
        action=argparse.BooleanOptionalAction,
        default=bool_default(ref, "use_focal", False),
    )
    p.add_argument(
        "--mask_thresh",
        type=float,
        default=float(ref_default(ref, "mask_thresh", 0.5)),
    )

    # debug
    p.add_argument(
        "--save_debug_every",
        type=int,
        default=int(ref_default(ref, "save_debug_every", 5)),
    )
    p.add_argument(
        "--debug_items",
        type=int,
        default=int(ref_default(ref, "debug_items", 4)),
    )
    return p


def parse_args_and_reference() -> Tuple[argparse.Namespace, Dict[str, Any]]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--reference_ckpt", default=DEFAULT_REFERENCE_CKPT)
    known, _ = pre.parse_known_args()

    saved_args, audit = audit_reference_checkpoint(Path(known.reference_ckpt))
    parser = build_parser(saved_args)
    args = parser.parse_args()

    # 用户若在第二阶段传了不同 reference path，重新审计
    if Path(args.reference_ckpt) != Path(known.reference_ckpt):
        saved_args, audit = audit_reference_checkpoint(Path(args.reference_ckpt))

    args.reference_checkpoint_sha256 = audit["sha256"]
    return args, audit


# ============================================================
# 实现审计与 dry-run
# ============================================================
def variant_description(variant: str) -> Dict[str, Any]:
    table = {
        "sam2_only": {
            "sam2_branch": True,
            "cnn_branch": False,
            "fusion": "single SAM2 pyramid + original residual top-down decoding",
            "changed_from_full": "CNN local branch and cross-branch addition removed",
        },
        "cnn_only": {
            "sam2_branch": False,
            "cnn_branch": True,
            "fusion": "single CNN pyramid + original residual top-down decoding",
            "changed_from_full": "SAM2/LoRA branch and cross-branch addition removed",
        },
        "dual_simple": {
            "sam2_branch": True,
            "cnn_branch": True,
            "fusion": "same-scale concat + 1x1 conv; non-residual top-down fusion",
            "changed_from_full": "multi-scale residual fusion replaced by simple fusion",
        },
        "full": {
            "sam2_branch": True,
            "cnn_branch": True,
            "fusion": "original projected addition + multi-scale residual fusion",
            "changed_from_full": "none",
        },
    }
    return table[variant]


def write_model_audit(
    out_dir: Path,
    args: argparse.Namespace,
    reference_audit: Mapping[str, Any],
    model: nn.Module,
    total_params: int,
    trainable_params: int,
    train_ds: core.OrganoidDenseDataset,
    val_ds: core.OrganoidDenseDataset,
    device: torch.device,
) -> None:
    desc = variant_description(args.variant)
    source_path = Path(core.__file__).resolve()

    lines = [
        "# MODEL IMPLEMENTATION AUDIT",
        "",
        f"- generated_at: `{now_str()}`",
        f"- variant: `{args.variant}`",
        f"- sam2_branch: `{desc['sam2_branch']}`",
        f"- cnn_branch: `{desc['cnn_branch']}`",
        f"- fusion: `{desc['fusion']}`",
        f"- changed_from_full: `{desc['changed_from_full']}`",
        f"- original_training_script: `{source_path}`",
        f"- original_training_script_sha256: `{sha256_file(source_path)}`",
        f"- reference_full_checkpoint: `{args.reference_ckpt}`",
        f"- reference_checkpoint_sha256: `{reference_audit['sha256']}`",
        f"- reference_checkpoint_audit: `{reference_audit['status']}`",
        f"- reference_epoch: `{reference_audit.get('epoch')}`",
        f"- reference_best_dice: `{reference_audit.get('best_dice')}`",
        f"- data_root: `{args.data_root}`",
        f"- train_samples: `{len(train_ds)}`",
        f"- val_samples: `{len(val_ds)}`",
        f"- total_params: `{total_params}`",
        f"- trainable_params: `{trainable_params}`",
        f"- device: `{device}`",
        f"- torch_version: `{torch.__version__}`",
        f"- cuda_version: `{torch.version.cuda}`",
        "",
        "## Controlled training configuration",
        "",
        "The common defaults were recovered from the formal full-model checkpoint.",
        "Only the declared frontend structure differs between variants.",
        "",
        "## Important evaluation boundary",
        "",
        "This script trains dense cue predictors only. Reconstruction/rescue parameters",
        "must remain frozen from validation and the same unified evaluator must be used",
        "for all final test comparisons.",
        "",
    ]
    (out_dir / "MODEL_IMPLEMENTATION_AUDIT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_dry_step(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    model.train()
    batch = next(iter(loader))
    batch = core.move_batch_to_device(batch, device)
    image = batch["image"]

    model.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=args.amp):
        pred = model(image)
        loss, info = core.total_loss_fn(
            pred,
            batch,
            lambda_boundary=args.lambda_boundary,
            lambda_center=args.lambda_center,
            use_focal=args.use_focal,
        )

    loss.backward()

    finite_outputs = {
        k: bool(torch.isfinite(v).all().item())
        for k, v in pred.items()
    }
    grad_tensors = 0
    finite_grad_tensors = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_tensors += 1
            if torch.isfinite(p.grad).all():
                finite_grad_tensors += 1

    result = {
        "status": "PASS"
        if all(finite_outputs.values())
        and grad_tensors > 0
        and grad_tensors == finite_grad_tensors
        else "FAIL",
        "loss": float(loss.detach().cpu()),
        "loss_components": info,
        "output_shapes": {k: list(v.shape) for k, v in pred.items()},
        "finite_outputs": finite_outputs,
        "gradient_tensor_count": grad_tensors,
        "finite_gradient_tensor_count": finite_grad_tensors,
    }
    model.zero_grad(set_to_none=True)
    return result


# ============================================================
# 主训练流程
# ============================================================
def main() -> None:
    args, reference_audit = parse_args_and_reference()
    run_name = args.run_name.strip() or args.variant
    if args.dry_run and not args.run_name.strip():
        run_name = f"{args.variant}_dryrun"

    save_dir = Path(args.save_root) / run_name
    seed_matches = args.expected_seed is None or args.seed == args.expected_seed
    runtime_args_audit = {
        "status": "PASS" if seed_matches else "FAIL_SEED_MISMATCH",
        "source_config_path": args.source_config_path,
        "expected_seed": args.expected_seed,
        "actual_runtime_seed": args.seed,
        "architecture": args.variant,
        "output_directory": str(save_dir),
        "num_workers": args.num_workers,
        "save_debug_every": args.save_debug_every,
        "test_used_for_selection": False,
        "checkpoint_selection_metric": "validation foreground Dice",
        "model_built": False,
        "optimizer_step_executed": False,
        "formal_training_started": False,
        "runtime_args_audit_only": args.runtime_args_audit_only,
    }
    print(f"[RuntimeArgs] source_config_path={args.source_config_path}")
    print(f"[RuntimeArgs] expected_seed={args.expected_seed}")
    print(f"[RuntimeArgs] actual_runtime_seed={args.seed}")
    print(f"[RuntimeArgs] architecture={args.variant}")
    print(f"[RuntimeArgs] output_directory={save_dir}")
    if args.runtime_args_audit_path:
        write_json(Path(args.runtime_args_audit_path), runtime_args_audit)
        print(f"[RuntimeArgs] audit={args.runtime_args_audit_path}")
    if not seed_matches:
        raise RuntimeError(
            f"Runtime seed mismatch before model construction: "
            f"expected={args.expected_seed}, actual={args.seed}"
        )
    if args.runtime_args_audit_only:
        print("[RuntimeArgs] parser-only audit complete; no training started.")
        return

    set_reproducibility(args.seed, args.deterministic)
    if save_dir.exists() and (save_dir / "train_log.csv").exists():
        if not args.resume and not args.overwrite and not args.dry_run:
            raise FileExistsError(
                f"输出目录已有训练日志：{save_dir}\n"
                "继续训练请使用 --resume <last_model.pth>；"
                "确认重新运行则显式添加 --overwrite。"
            )
    save_dir.mkdir(parents=True, exist_ok=True)

    write_json(save_dir / "REFERENCE_CHECKPOINT_AUDIT.json", reference_audit)
    write_json(save_dir / "TRAIN_CONFIG.json", vars(args))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 88)
    print(f"[Start] {now_str()}")
    print(f"[Variant] {args.variant}")
    print(f"[Device] {device}")
    print(f"[Reference full checkpoint] {args.reference_ckpt}")
    print(f"[Reference audit] {reference_audit['status']}")
    print(f"[Reference SHA256] {reference_audit['sha256']}")
    print(f"[Data root] {args.data_root}")
    print(f"[Save dir] {save_dir}")
    print("=" * 88)

    train_ds = core.OrganoidDenseDataset(
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
    val_ds = core.OrganoidDenseDataset(
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
    print(
        "[Dataset] first train pair: "
        f"{train_ds.pairs[0][0].name} | {train_ds.pairs[0][1].name}"
    )

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

    model = build_model(args, device=device).to(device)

    print("[Init] Initializing lazy projection layers...")
    core.initialize_lazy_modules(
        model,
        device=device,
        img_size=args.lazy_init_size,
    )
    total_params, trainable_params = core.count_parameters(model)
    print(
        f"[Params] total={total_params / 1e6:.3f}M "
        f"trainable={trainable_params / 1e6:.3f}M"
    )

    write_model_audit(
        out_dir=save_dir,
        args=args,
        reference_audit=reference_audit,
        model=model,
        total_params=total_params,
        trainable_params=trainable_params,
        train_ds=train_ds,
        val_ds=val_ds,
        device=device,
    )

    if args.dry_run:
        print("[DryRun] Running one forward/backward step...")
        result = run_dry_step(
            model=model,
            loader=train_loader,
            device=device,
            args=args,
        )
        write_json(save_dir / "DRY_RUN_RESULT.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if result["status"] != "PASS":
            raise RuntimeError("Dry-run failed. Do not start formal training.")
        print(f"[DryRun] PASS: {save_dir / 'DRY_RUN_RESULT.json'}")
        return

    groups = core.optimizer_parameter_groups(
        model,
        lr=args.lr,
        lora_lr=args.lora_lr,
        weight_decay=args.weight_decay,
    )
    optimizer = torch.optim.AdamW(
        groups,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.05,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    start_epoch = 1
    best_dice = -1.0
    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        start_epoch, best_dice = core.load_checkpoint(
            args.resume,
            model,
            optimizer,
            scheduler,
            device=device,
        )
        print(
            f"[Resume] start_epoch={start_epoch}, "
            f"best_dice={best_dice:.6f}"
        )

    log_path = save_dir / "train_log.csv"
    if not log_path.exists() or (args.overwrite and not args.resume):
        with log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "variant",
                    "epoch",
                    "train_loss",
                    "train_iou",
                    "train_dice",
                    "train_precision",
                    "train_recall",
                    "val_loss",
                    "val_iou",
                    "val_dice",
                    "val_precision",
                    "val_recall",
                    "best_dice",
                    "lr",
                    "time_seconds",
                ]
            )

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_t0 = time.time()

        train_stats = core.train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            args,
        )
        val_stats = core.validate(
            model,
            val_loader,
            device,
            args,
        )
        scheduler.step()

        improved = val_stats["dice"] > best_dice
        if improved:
            best_dice = val_stats["dice"]
            core.save_checkpoint(
                save_dir / "best_model.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_dice,
                args,
            )

        core.save_checkpoint(
            save_dir / "last_model.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_dice,
            args,
        )

        if args.save_debug_every > 0 and (
            epoch == 1 or epoch % args.save_debug_every == 0
        ):
            debug_dir = save_dir / "debug_vis" / f"epoch_{epoch:03d}"
            try:
                core.save_debug_visuals(
                    model,
                    val_loader,
                    device,
                    debug_dir,
                    args,
                    max_items=args.debug_items,
                )
            except Exception as exc:
                print(
                    "[DebugVis][Warning] failed to save debug visuals: "
                    f"{repr(exc)}"
                )

        lr_now = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_t0

        print(
            f"\nEpoch [{epoch:03d}/{args.epochs}] variant={args.variant} | "
            f"Train loss={train_stats['loss']:.4f}, "
            f"IoU={train_stats['iou']:.4f}, "
            f"Dice={train_stats['dice']:.4f}, "
            f"P={train_stats['precision']:.4f}, "
            f"R={train_stats['recall']:.4f} | "
            f"Val loss={val_stats['loss']:.4f}, "
            f"IoU={val_stats['iou']:.4f}, "
            f"Dice={val_stats['dice']:.4f}, "
            f"P={val_stats['precision']:.4f}, "
            f"R={val_stats['recall']:.4f} | "
            f"BestDice={best_dice:.4f}"
            f"{' *' if improved else ''} | "
            f"lr={lr_now:.2e} | time={epoch_time:.1f}s\n"
        )

        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    args.variant,
                    epoch,
                    train_stats["loss"],
                    train_stats["iou"],
                    train_stats["dice"],
                    train_stats["precision"],
                    train_stats["recall"],
                    val_stats["loss"],
                    val_stats["iou"],
                    val_stats["dice"],
                    val_stats["precision"],
                    val_stats["recall"],
                    best_dice,
                    lr_now,
                    epoch_time,
                ]
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    completion = {
        "status": "TRAINING_COMPLETED",
        "variant": args.variant,
        "best_dice": best_dice,
        "best_checkpoint": str(save_dir / "best_model.pth"),
        "last_checkpoint": str(save_dir / "last_model.pth"),
        "reference_full_checkpoint": args.reference_ckpt,
        "reference_checkpoint_sha256": reference_audit["sha256"],
        "finished_at": now_str(),
        "next_step": (
            "Use the frozen Val reconstruction/rescue configuration and the "
            "same unified evaluator. Do not tune on Test."
        ),
    }
    write_json(save_dir / "TRAINING_COMPLETION.json", completion)

    print("=" * 88)
    print(f"[Done] {now_str()}")
    print(f"[Variant] {args.variant}")
    print(f"[Best Dice] {best_dice:.6f}")
    print(f"[Checkpoints] {save_dir}")
    print("=" * 88)


if __name__ == "__main__":
    main()
