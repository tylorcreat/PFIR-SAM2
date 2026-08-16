from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


METRIC_FIELDS = [
    "Method",
    "Source",
    "GT Count",
    "Pred Count",
    "TP",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
    "AP50",
    "AP75",
    "mAP50-95",
    "Mean IoU",
    "Mean Dice",
    "Matched IoU",
    "Count MAE",
    "Count MAPE",
    "Total Area MAE",
    "Total Area MAPE",
    "Mean Area MAE",
    "Mean Area MAPE",
    "Small Recall",
    "Medium Recall",
    "Large Recall",
    "Runtime per image",
]

PER_IMAGE_FIELDS = [
    "Method",
    "Source",
    "image_id",
    "image_name",
    "time_h",
    "GT Count",
    "Pred Count",
    "Count AE",
    "Count APE",
    "GT Total Area",
    "Pred Total Area",
    "Total Area AE",
    "Total Area APE",
    "GT Mean Instance Area",
    "Pred Mean Instance Area",
    "Mean Area AE",
    "Mean Area APE",
    "TP",
    "FP",
    "FN",
    "Precision",
    "Recall",
    "F1",
    "AP50",
    "AP75",
    "mAP50-95",
    "Mean IoU",
    "Mean Dice",
    "Matched IoU",
    "runtime_seconds",
]

TIME_SERIES_FIELDS = [
    "Method",
    "Source",
    "image_id",
    "image_name",
    "time_h",
    "GT Count",
    "Pred Count",
    "Count AE",
    "Count APE",
    "GT Total Area",
    "Pred Total Area",
    "Total Area AE",
    "Total Area APE",
    "GT Mean Instance Area",
    "Pred Mean Instance Area",
    "Mean Area AE",
    "Mean Area APE",
]

TIME_SERIES_SUMMARY_FIELDS = [
    "Method",
    "Source",
    "n_timepoints",
    "Count curve MAE",
    "Count curve RMSE",
    "Count curve MAPE",
    "Count Pearson correlation",
    "Count Spearman correlation",
    "Count AUC error",
    "Total area curve MAE",
    "Total area curve RMSE",
    "Total area curve MAPE",
    "Total area Pearson correlation",
    "Total area Spearman correlation",
    "Total area AUC error",
    "Mean area curve MAE",
    "Mean area curve RMSE",
    "Mean area curve MAPE",
    "Mean area Pearson correlation",
    "Mean area Spearman correlation",
    "Mean area AUC error",
]


def normalize_stem(value: str | Path) -> str:
    stem = Path(str(value)).stem.lower()
    for suffix in [
        "_instance_map",
        "_pred_instance",
        "_instances",
        "_mask",
        "_masks",
        "_masks_organoid",
        "_binary",
        "_label",
    ]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def infer_source(image_name: str) -> str:
    lower = image_name.lower()
    if "human" in lower:
        return "Human"
    if "mouse" in lower:
        return "Mouse"
    return "Unknown"


def parse_time_h(image_name: str) -> int | None:
    match = re.search(r"(?<!\d)(\d{1,4})h(?![a-zA-Z])", image_name)
    if not match:
        return None
    return int(match.group(1))


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_instance_map(path: str | Path | None, shape: tuple[int, int] | None = None) -> np.ndarray:
    if path is None:
        if shape is None:
            raise ValueError("shape is required when loading a missing prediction map")
        return np.zeros(shape, dtype=np.uint16)
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.int64)


def label_areas(label_map: np.ndarray) -> dict[int, int]:
    ids, counts = np.unique(label_map, return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts) if int(i) != 0}


def pair_iou_candidates(gt_map: np.ndarray, pred_map: np.ndarray) -> tuple[list[tuple[float, float, int, int]], dict[int, int], dict[int, int]]:
    gt_areas = label_areas(gt_map)
    pred_areas = label_areas(pred_map)
    if not gt_areas or not pred_areas:
        return [], gt_areas, pred_areas
    gt_base = int(gt_map.max()) + 1
    pairs = pred_map.astype(np.int64).ravel() * gt_base + gt_map.astype(np.int64).ravel()
    pair_ids, counts = np.unique(pairs, return_counts=True)
    candidates = []
    for pair_id, inter in zip(pair_ids, counts):
        pred_id = int(pair_id // gt_base)
        gt_id = int(pair_id % gt_base)
        if pred_id == 0 or gt_id == 0:
            continue
        union = pred_areas[pred_id] + gt_areas[gt_id] - int(inter)
        if union <= 0:
            continue
        iou = float(inter / union)
        dice = float(2 * int(inter) / (pred_areas[pred_id] + gt_areas[gt_id]))
        candidates.append((iou, dice, gt_id, pred_id))
    candidates.sort(reverse=True, key=lambda item: item[0])
    return candidates, gt_areas, pred_areas


def greedy_from_candidates(candidates: list[tuple[float, float, int, int]], gt_ids: set[int], pred_ids: set[int], iou_thr: float):
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches = []
    for iou, dice, gt_id, pred_id in candidates:
        if iou < iou_thr:
            break
        if gt_id in used_gt or pred_id in used_pred:
            continue
        used_gt.add(gt_id)
        used_pred.add(pred_id)
        matches.append({"gt_instance_id": gt_id, "pred_instance_id": pred_id, "iou": iou, "dice": dice})
    fn = sorted(gt_ids - used_gt)
    fp = sorted(pred_ids - used_pred)
    return matches, fn, fp


def precision_at_iou(candidates, gt_ids: set[int], pred_ids: set[int], iou_thr: float) -> float:
    matches, _, fp = greedy_from_candidates(candidates, gt_ids, pred_ids, iou_thr)
    denom = len(matches) + len(fp)
    return len(matches) / denom if denom else 0.0


def read_size_groups(path: str | Path | None) -> dict[tuple[str, int], str]:
    groups: dict[tuple[str, int], str] = {}
    if not path or not Path(path).exists():
        return groups
    for row in read_csv_rows(path):
        image_id = normalize_stem(row.get("image_id") or row.get("image_name") or row.get("image_path", ""))
        raw = row.get("instance_id") or row.get("box_id", "").split("_")[-1]
        try:
            instance_id = int(float(raw))
        except Exception:
            continue
        group = row.get("size_group", "")
        if group:
            groups[(image_id, instance_id)] = group
    return groups


def new_acc(method: str, source: str) -> dict:
    return {
        "Method": method,
        "Source": source,
        "GT Count": 0,
        "Pred Count": 0,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "ious": [],
        "dices": [],
        "ap50": [],
        "ap75": [],
        "map5095": [],
        "count_abs_errors": [],
        "count_apes": [],
        "total_area_abs_errors": [],
        "total_area_apes": [],
        "mean_area_abs_errors": [],
        "mean_area_apes": [],
        "runtimes": [],
        "size_gt": {"small": 0, "medium": 0, "large": 0},
        "size_tp": {"small": 0, "medium": 0, "large": 0},
    }


def update_acc(acc: dict, image_stats: dict, matches: list[dict], fn: list[int], size_groups: dict, image_id: str) -> None:
    acc["GT Count"] += image_stats["GT Count"]
    acc["Pred Count"] += image_stats["Pred Count"]
    acc["TP"] += image_stats["TP"]
    acc["FP"] += image_stats["FP"]
    acc["FN"] += image_stats["FN"]
    acc["ious"].extend(m["iou"] for m in matches)
    acc["dices"].extend(m["dice"] for m in matches)
    acc["ap50"].append(image_stats["AP50"])
    acc["ap75"].append(image_stats["AP75"])
    acc["map5095"].append(image_stats["mAP50-95"])
    acc["count_abs_errors"].append(image_stats["Count AE"])
    acc["count_apes"].append(image_stats["Count APE"])
    acc["total_area_abs_errors"].append(image_stats["Total Area AE"])
    acc["total_area_apes"].append(image_stats["Total Area APE"])
    acc["mean_area_abs_errors"].append(image_stats["Mean Area AE"])
    acc["mean_area_apes"].append(image_stats["Mean Area APE"])
    if image_stats.get("runtime_seconds") not in (None, ""):
        acc["runtimes"].append(float(image_stats["runtime_seconds"]))
    for match in matches:
        group = size_groups.get((image_id, match["gt_instance_id"]), "")
        if group in acc["size_gt"]:
            acc["size_gt"][group] += 1
            acc["size_tp"][group] += 1
    for gt_id in fn:
        group = size_groups.get((image_id, gt_id), "")
        if group in acc["size_gt"]:
            acc["size_gt"][group] += 1


def finalize_acc(acc: dict) -> dict[str, object]:
    tp, fp, fn = acc["TP"], acc["FP"], acc["FN"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    row = {
        "Method": acc["Method"],
        "Source": acc["Source"],
        "GT Count": acc["GT Count"],
        "Pred Count": acc["Pred Count"],
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Precision": round(precision, 6),
        "Recall": round(recall, 6),
        "F1": round(f1, 6),
        "AP50": round(float(np.mean(acc["ap50"])) if acc["ap50"] else 0.0, 6),
        "AP75": round(float(np.mean(acc["ap75"])) if acc["ap75"] else 0.0, 6),
        "mAP50-95": round(float(np.mean(acc["map5095"])) if acc["map5095"] else 0.0, 6),
        "Mean IoU": round(float(np.mean(acc["ious"])) if acc["ious"] else 0.0, 6),
        "Mean Dice": round(float(np.mean(acc["dices"])) if acc["dices"] else 0.0, 6),
        "Matched IoU": round(float(np.mean(acc["ious"])) if acc["ious"] else 0.0, 6),
        "Count MAE": round(float(np.mean(acc["count_abs_errors"])) if acc["count_abs_errors"] else 0.0, 6),
        "Count MAPE": round(float(np.mean(acc["count_apes"])) if acc["count_apes"] else 0.0, 6),
        "Total Area MAE": round(float(np.mean(acc["total_area_abs_errors"])) if acc["total_area_abs_errors"] else 0.0, 6),
        "Total Area MAPE": round(float(np.mean(acc["total_area_apes"])) if acc["total_area_apes"] else 0.0, 6),
        "Mean Area MAE": round(float(np.mean(acc["mean_area_abs_errors"])) if acc["mean_area_abs_errors"] else 0.0, 6),
        "Mean Area MAPE": round(float(np.mean(acc["mean_area_apes"])) if acc["mean_area_apes"] else 0.0, 6),
        "Runtime per image": round(float(np.mean(acc["runtimes"])), 6) if acc["runtimes"] else "",
    }
    for group, field in [("small", "Small Recall"), ("medium", "Medium Recall"), ("large", "Large Recall")]:
        gt = acc["size_gt"][group]
        row[field] = round(acc["size_tp"][group] / gt, 6) if gt else 0.0
    return row


def pred_files_by_id(pred_dir: Path) -> dict[str, Path]:
    files = [p for p in pred_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff"}]
    return {normalize_stem(p): p for p in files}


def evaluate_method(
    method: str,
    pred_dir: Path,
    gt_dir: Path,
    size_groups: dict,
    iou_thr: float,
    runtime_by_id: dict[str, float] | None = None,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    gt_files = sorted(p for p in gt_dir.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".tif", ".tiff"})
    pred_by_id = pred_files_by_id(pred_dir)
    accs = {source: new_acc(method, source) for source in ["All", "Human", "Mouse", "Unknown"]}
    per_image = []
    matched_rows = []
    size_rows = []
    thresholds = [round(x, 2) for x in np.arange(0.50, 1.00, 0.05)]

    for gt_path in gt_files:
        image_id = normalize_stem(gt_path)
        source = infer_source(gt_path.name)
        time_h = parse_time_h(gt_path.name)
        gt_map = load_instance_map(gt_path)
        pred_path = pred_by_id.get(image_id)
        pred_map = load_instance_map(pred_path, shape=gt_map.shape)
        if pred_map.shape != gt_map.shape:
            raise ValueError(f"Shape mismatch for {image_id}: gt={gt_map.shape}, pred={pred_map.shape}, path={pred_path}")

        candidates, gt_areas, pred_areas = pair_iou_candidates(gt_map, pred_map)
        gt_ids = set(gt_areas)
        pred_ids = set(pred_areas)
        matches, fn, fp = greedy_from_candidates(candidates, gt_ids, pred_ids, iou_thr)
        precision = len(matches) / (len(matches) + len(fp)) if matches or fp else 0.0
        recall = len(matches) / (len(matches) + len(fn)) if matches or fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ap50 = precision_at_iou(candidates, gt_ids, pred_ids, 0.50)
        ap75 = precision_at_iou(candidates, gt_ids, pred_ids, 0.75)
        map5095 = float(np.mean([precision_at_iou(candidates, gt_ids, pred_ids, thr) for thr in thresholds])) if thresholds else 0.0
        count_ae = abs(len(pred_areas) - len(gt_areas))
        count_ape = count_ae / len(gt_areas) if gt_areas else 0.0
        gt_total_area = int(sum(gt_areas.values()))
        pred_total_area = int(sum(pred_areas.values()))
        total_area_ae = abs(pred_total_area - gt_total_area)
        total_area_ape = total_area_ae / gt_total_area if gt_total_area else 0.0
        gt_mean_area = float(np.mean(list(gt_areas.values()))) if gt_areas else 0.0
        pred_mean_area = float(np.mean(list(pred_areas.values()))) if pred_areas else 0.0
        mean_area_ae = abs(pred_mean_area - gt_mean_area)
        mean_area_ape = mean_area_ae / gt_mean_area if gt_mean_area else 0.0
        mean_iou = float(np.mean([m["iou"] for m in matches])) if matches else 0.0
        mean_dice = float(np.mean([m["dice"] for m in matches])) if matches else 0.0
        image_stats = {
            "Method": method,
            "Source": source,
            "image_id": image_id,
            "image_name": gt_path.name,
            "time_h": "" if time_h is None else time_h,
            "GT Count": len(gt_areas),
            "Pred Count": len(pred_areas),
            "Count AE": count_ae,
            "Count APE": round(count_ape, 6),
            "GT Total Area": gt_total_area,
            "Pred Total Area": pred_total_area,
            "Total Area AE": total_area_ae,
            "Total Area APE": round(total_area_ape, 6),
            "GT Mean Instance Area": round(gt_mean_area, 6),
            "Pred Mean Instance Area": round(pred_mean_area, 6),
            "Mean Area AE": round(mean_area_ae, 6),
            "Mean Area APE": round(mean_area_ape, 6),
            "TP": len(matches),
            "FP": len(fp),
            "FN": len(fn),
            "Precision": round(precision, 6),
            "Recall": round(recall, 6),
            "F1": round(f1, 6),
            "AP50": round(ap50, 6),
            "AP75": round(ap75, 6),
            "mAP50-95": round(map5095, 6),
            "Mean IoU": round(mean_iou, 6),
            "Mean Dice": round(mean_dice, 6),
            "Matched IoU": round(mean_iou, 6),
            "runtime_seconds": "" if runtime_by_id is None or image_id not in runtime_by_id else runtime_by_id[image_id],
        }
        per_image.append(image_stats)
        for acc_name in ["All", source]:
            update_acc(accs[acc_name], image_stats, matches, fn, size_groups, image_id)
        for match in matches:
            matched_rows.append(
                {
                    "Method": method,
                    "Source": source,
                    "image_id": image_id,
                    "gt_instance_id": match["gt_instance_id"],
                    "pred_instance_id": match["pred_instance_id"],
                    "IoU": round(match["iou"], 6),
                    "Dice": round(match["dice"], 6),
                    "size_group": size_groups.get((image_id, match["gt_instance_id"]), ""),
                }
            )

    summary_all = [finalize_acc(accs["All"])]
    summary_by_source = [finalize_acc(accs[source]) for source in ["All", "Human", "Mouse"]]
    if accs["Unknown"]["GT Count"] or accs["Unknown"]["Pred Count"]:
        summary_by_source.append(finalize_acc(accs["Unknown"]))
    for source in ["All", "Human", "Mouse"]:
        acc = accs[source]
        for group in ["small", "medium", "large"]:
            gt = acc["size_gt"][group]
            tp = acc["size_tp"][group]
            size_rows.append(
                {
                    "Method": method,
                    "Source": source,
                    "size_group": group,
                    "GT Count": gt,
                    "TP": tp,
                    "FN": gt - tp,
                    "Recall": round(tp / gt, 6) if gt else 0.0,
                }
            )
    return summary_all, summary_by_source, per_image, matched_rows, size_rows


def pearson_corr(gt: np.ndarray, pred: np.ndarray) -> float:
    if len(gt) < 2 or float(np.std(gt)) == 0.0 or float(np.std(pred)) == 0.0:
        return 0.0
    return float(np.corrcoef(gt, pred)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman_corr(gt: np.ndarray, pred: np.ndarray) -> float:
    if len(gt) < 2:
        return 0.0
    return pearson_corr(rankdata(gt), rankdata(pred))


def auc_error(times: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> float:
    if len(times) == 0:
        return 0.0
    if len(times) == 1:
        return float(abs(pred[0] - gt[0]))
    gt_auc = float(np.trapezoid(gt, times))
    pred_auc = float(np.trapezoid(pred, times))
    return abs(pred_auc - gt_auc)


def curve_metrics(times: list[int], gt_values: list[float], pred_values: list[float]) -> dict[str, float]:
    if not times:
        return {"MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "Pearson": 0.0, "Spearman": 0.0, "AUC error": 0.0}
    order = np.argsort(np.asarray(times, dtype=float))
    t = np.asarray(times, dtype=float)[order]
    gt = np.asarray(gt_values, dtype=float)[order]
    pred = np.asarray(pred_values, dtype=float)[order]
    err = pred - gt
    denom = np.where(gt != 0, np.abs(gt), np.nan)
    ape = np.abs(err) / denom
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(math.sqrt(np.mean(err * err))),
        "MAPE": float(np.nanmean(ape)) if not np.all(np.isnan(ape)) else 0.0,
        "Pearson": pearson_corr(gt, pred),
        "Spearman": spearman_corr(gt, pred),
        "AUC error": auc_error(t, gt, pred),
    }


def build_time_series_outputs(per_image: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = [row for row in per_image if row.get("time_h") not in ("", None)]
    if not rows:
        return [], []
    ts_rows = [{field: row.get(field, "") for field in TIME_SERIES_FIELDS} for row in rows]
    summary_rows = []
    for method in sorted({row["Method"] for row in rows}):
        method_rows = [row for row in rows if row["Method"] == method]
        for source in ["All", "Human", "Mouse"]:
            source_rows = method_rows if source == "All" else [row for row in method_rows if row["Source"] == source]
            if not source_rows:
                continue
            by_time: dict[int, list[dict]] = defaultdict(list)
            for row in source_rows:
                by_time[int(row["time_h"])].append(row)
            times = sorted(by_time)
            gt_count = [sum(float(r["GT Count"]) for r in by_time[t]) for t in times]
            pred_count = [sum(float(r["Pred Count"]) for r in by_time[t]) for t in times]
            gt_total = [sum(float(r["GT Total Area"]) for r in by_time[t]) for t in times]
            pred_total = [sum(float(r["Pred Total Area"]) for r in by_time[t]) for t in times]
            gt_mean = [float(np.mean([float(r["GT Mean Instance Area"]) for r in by_time[t]])) for t in times]
            pred_mean = [float(np.mean([float(r["Pred Mean Instance Area"]) for r in by_time[t]])) for t in times]
            count_m = curve_metrics(times, gt_count, pred_count)
            total_m = curve_metrics(times, gt_total, pred_total)
            mean_m = curve_metrics(times, gt_mean, pred_mean)
            summary_rows.append(
                {
                    "Method": method,
                    "Source": source,
                    "n_timepoints": len(times),
                    "Count curve MAE": round(count_m["MAE"], 6),
                    "Count curve RMSE": round(count_m["RMSE"], 6),
                    "Count curve MAPE": round(count_m["MAPE"], 6),
                    "Count Pearson correlation": round(count_m["Pearson"], 6),
                    "Count Spearman correlation": round(count_m["Spearman"], 6),
                    "Count AUC error": round(count_m["AUC error"], 6),
                    "Total area curve MAE": round(total_m["MAE"], 6),
                    "Total area curve RMSE": round(total_m["RMSE"], 6),
                    "Total area curve MAPE": round(total_m["MAPE"], 6),
                    "Total area Pearson correlation": round(total_m["Pearson"], 6),
                    "Total area Spearman correlation": round(total_m["Spearman"], 6),
                    "Total area AUC error": round(total_m["AUC error"], 6),
                    "Mean area curve MAE": round(mean_m["MAE"], 6),
                    "Mean area curve RMSE": round(mean_m["RMSE"], 6),
                    "Mean area curve MAPE": round(mean_m["MAPE"], 6),
                    "Mean area Pearson correlation": round(mean_m["Pearson"], 6),
                    "Mean area Spearman correlation": round(mean_m["Spearman"], 6),
                    "Mean area AUC error": round(mean_m["AUC error"], 6),
                }
            )
    return ts_rows, summary_rows


def parse_methods(args: argparse.Namespace) -> list[tuple[str, Path]]:
    methods = []
    for item in args.method:
        if "=" not in item:
            raise ValueError(f"Method must be Name=pred_instance_dir, got: {item}")
        name, path = item.split("=", 1)
        methods.append((name, Path(path)))
    if args.method_name and args.pred_instance_dir:
        methods.append((args.method_name, Path(args.pred_instance_dir)))
    if not methods:
        raise ValueError("Provide --method Name=pred_instance_dir or --method-name plus --pred-instance-dir")
    return methods


def parse_runtime_csvs(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    runtime_maps: dict[str, dict[str, float]] = {}
    for item in getattr(args, "runtime_csv", []) or []:
        if "=" not in item:
            raise ValueError(f"Runtime CSV must be Method=csv_path, got: {item}")
        method, path_text = item.split("=", 1)
        path = Path(path_text)
        mapping: dict[str, float] = {}
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                image_value = row.get("image_id") or row.get("image_name") or row.get("image")
                runtime_value = row.get("runtime_seconds") or row.get("Runtime seconds") or row.get("runtime")
                if not image_value or runtime_value in (None, ""):
                    continue
                mapping[normalize_stem(image_value)] = float(runtime_value)
        runtime_maps[method] = mapping
    return runtime_maps


def main(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_dir = Path(args.gt_mask_dir)
    size_groups = read_size_groups(args.gt_boxes_csv)
    all_summary: list[dict] = []
    by_source: list[dict] = []
    per_image: list[dict] = []
    matched: list[dict] = []
    size_rows: list[dict] = []
    runtime_maps = parse_runtime_csvs(args)
    for method, pred_dir in parse_methods(args):
        if not pred_dir.exists():
            print(f"[WARNING] Missing prediction directory for {method}: {pred_dir}")
            continue
        summary_i, by_source_i, per_image_i, matched_i, size_i = evaluate_method(
            method,
            pred_dir,
            gt_dir,
            size_groups,
            args.iou_thr,
            runtime_by_id=runtime_maps.get(method),
        )
        all_summary.extend(summary_i)
        by_source.extend(by_source_i)
        per_image.extend(per_image_i)
        matched.extend(matched_i)
        size_rows.extend(size_i)

    write_csv(output_dir / "summary_metrics.csv", all_summary, METRIC_FIELDS)
    write_csv(output_dir / "summary_metrics_by_source.csv", by_source, METRIC_FIELDS)
    write_csv(output_dir / "per_image_metrics.csv", per_image, PER_IMAGE_FIELDS)
    write_csv(output_dir / "matched_instances.csv", matched, ["Method", "Source", "image_id", "gt_instance_id", "pred_instance_id", "IoU", "Dice", "size_group"])
    write_csv(output_dir / "size_group_metrics.csv", size_rows, ["Method", "Source", "size_group", "GT Count", "TP", "FN", "Recall"])
    ts_rows, ts_summary_rows = build_time_series_outputs(per_image)
    write_csv(output_dir / "quantification_time_series.csv", ts_rows, TIME_SERIES_FIELDS)
    write_csv(output_dir / "time_series_metrics_by_source.csv", ts_summary_rows, TIME_SERIES_SUMMARY_FIELDS)

    if args.combined_prefix:
        write_csv(output_dir / f"{args.combined_prefix}_summary.csv", all_summary, METRIC_FIELDS)
        write_csv(output_dir / f"{args.combined_prefix}_summary_by_source.csv", by_source, METRIC_FIELDS)
        write_csv(output_dir / f"{args.combined_prefix}_per_image_metrics.csv", per_image, PER_IMAGE_FIELDS)
        write_csv(output_dir / f"{args.combined_prefix}_quantification_time_series.csv", ts_rows, TIME_SERIES_FIELDS)
        write_csv(output_dir / f"{args.combined_prefix}_time_series_metrics_by_source.csv", ts_summary_rows, TIME_SERIES_SUMMARY_FIELDS)
    print(f"[DONE] evaluated methods: {len(set(row['Method'] for row in by_source))}")
    print(f"[DONE] output_dir={output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified instance segmentation evaluator for OrganoID methods.")
    parser.add_argument("--gt-mask-dir", required=True)
    parser.add_argument("--pred-instance-dir", default="")
    parser.add_argument("--method-name", default="")
    parser.add_argument("--method", action="append", default=[], help="Name=pred_instance_dir; repeat for all methods.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gt-boxes-csv", default="detector_dataset_organoid_external_rgb/bbox_instances_test.csv")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--combined-prefix", default="")
    parser.add_argument(
        "--runtime-csv",
        action="append",
        default=[],
        help="Optional Method=runtime_csv mapping. CSV needs image_id/image_name and runtime_seconds.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
