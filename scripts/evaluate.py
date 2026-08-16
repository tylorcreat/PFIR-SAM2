#!/usr/bin/env python
"""Unified fixed-IoU instance-segmentation evaluation entry point."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.unified_evaluator import build_parser, main


if __name__ == "__main__":
    main(build_parser().parse_args())
