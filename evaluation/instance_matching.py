"""Greedy one-to-one fixed-IoU instance matching used in the manuscript."""

from .unified_evaluator import (
    greedy_from_candidates,
    label_areas,
    pair_iou_candidates,
    precision_at_iou,
)

__all__ = [
    "greedy_from_candidates",
    "label_areas",
    "pair_iou_candidates",
    "precision_at_iou",
]
