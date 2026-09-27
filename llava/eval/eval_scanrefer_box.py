"""Aggregate ScanRefer mask and raw-coordinate AABB metrics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
    return rows


def load_questions(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def key(row: dict) -> tuple[str, int, int]:
    return (str(row["scene_id"]), int(row["ann_id"]), int(row["object_id"]))


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--question-file", required=True, type=Path)
    parser.add_argument("--scan-folder", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.result_file)
    questions = load_questions(args.question_file)
    if len(rows) != len(questions):
        raise ValueError(f"Expected {len(questions)} rows, found {len(rows)}")

    expected = [(str(q["scene_id"]), int(q["ann_id"]), int(q["object_id"])) for q in questions]
    actual = [key(row) for row in rows]
    if len(set(actual)) != len(actual):
        raise ValueError("Duplicate ScanRefer scene_id/ann_id/object_id keys")
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))[:5]
        extra = sorted(set(actual) - set(expected))[:5]
        raise ValueError(f"ScanRefer identity mismatch; missing={missing}, extra={extra}")

    required = {"iou", "tp25", "tp50", "box_iou", "box_tp25", "box_tp50", "pred_box", "gt_box"}
    for index, row in enumerate(rows):
        missing = required.difference(row)
        if missing:
            raise KeyError(f"Row {index} missing fields: {sorted(missing)}")
        values = [row[field] for field in ("iou", "box_iou")]
        if not all(finite(value) and 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError(f"Invalid IoU values in row {index}: {values}")
        scan_path = args.scan_folder / f"{row['scene_id']}.pth"
        if not scan_path.is_file():
            raise FileNotFoundError(scan_path)
        raw_data = torch.load(scan_path, map_location="cpu")
        coordinates = np.asarray(raw_data["coord"], dtype=np.float32)
        instance = np.asarray(raw_data["instance_gt"])
        gt_points = coordinates[instance == int(row["object_id"])]
        if len(gt_points) == 0:
            raise ValueError(f"Missing GT instance points for row {index}")
        gt_box = np.stack((gt_points.min(axis=0), gt_points.max(axis=0))).tolist()
        if not np.allclose(np.asarray(row["gt_box"], dtype=np.float32), gt_box):
            raise ValueError(f"GT box mismatch for row {index}")
        if int(row["box_tp50"]) > int(row["box_tp25"]):
            raise ValueError(f"box threshold inconsistency in row {index}")

    def mean(field: str) -> float:
        return sum(float(row[field]) for row in rows) / len(rows)

    metrics = {
        "evaluator": "local_raw_coordinate_axis_aligned",
        "official_evaluator_available": False,
        "num_queries": len(rows),
        "mask_mIoU": mean("iou"),
        "mask_Acc@0.25": mean("tp25"),
        "mask_Acc@0.5": mean("tp50"),
        "box_Acc@0.25": mean("box_tp25"),
        "box_Acc@0.5": mean("box_tp50"),
        "box_mIoU": mean("box_iou"),
        "missing_queries": 0,
        "duplicate_queries": 0,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"{name}: {value:.6f}")
        else:
            print(f"{name}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
