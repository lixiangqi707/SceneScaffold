"""Evaluate Multi3DRefer box-set F1 with the official matching definition."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

try:
    from llava.eval.grounding_box_utils import corners_to_aabb, query_f1, hungarian_true_positive
except ModuleNotFoundError:
    from grounding_box_utils import corners_to_aabb, query_f1, hungarian_true_positive


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-file", required=True, type=Path)
    parser.add_argument("--question-file", required=True, type=Path)
    parser.add_argument("--scan-folder", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    rows = load_jsonl(args.result_file)
    questions = json.loads(args.question_file.read_text(encoding="utf-8"))
    by_key = {(str(row["scene_id"]), int(row["ann_id"])): row for row in rows}
    expected_keys = [(str(q["scene_id"]), int(q["ann_id"])) for q in questions]
    if len(rows) != len(questions) or len(by_key) != len(rows) or set(by_key) != set(expected_keys):
        raise ValueError("Multi3DRefer result identities are incomplete or duplicated")

    per_query = []
    by_type = defaultdict(list)
    for source in questions:
        key = (str(source["scene_id"]), int(source["ann_id"]))
        row = by_key[key]
        serialized_boxes = row.get("pred_aabb_corners", [])
        if not isinstance(serialized_boxes, list):
            raise ValueError(f"pred_aabb_corners must be a list for {key}")
        pred_boxes = []
        for box_index, corners in enumerate(serialized_boxes):
            array = np.asarray(corners, dtype=np.float32)
            if array.shape != (8, 3) or not np.isfinite(array).all():
                raise ValueError(f"Invalid predicted box at {key}[{box_index}]: {array.shape}")
            pred_boxes.append(corners_to_aabb(array))
        scan_path = args.scan_folder / f"{key[0]}.pth"
        if not scan_path.is_file():
            raise FileNotFoundError(scan_path)
        raw_data = torch.load(scan_path, map_location="cpu")
        coordinates = np.asarray(raw_data["coord"], dtype=np.float32)
        instance = np.asarray(raw_data["instance_gt"])
        gt_boxes = []
        for object_id in source["object_ids"]:
            points = coordinates[instance == int(object_id)]
            if len(points):
                gt_boxes.append(np.stack((points.min(axis=0), points.max(axis=0))).astype(np.float32))
        result = {"scene_id": key[0], "ann_id": key[1], "eval_type": source.get("eval_type"), "num_pred": len(pred_boxes), "num_gt": len(gt_boxes)}
        for threshold, label in ((0.25, "0.25"), (0.5, "0.5")):
            result[f"tp@{label}"] = hungarian_true_positive(pred_boxes, gt_boxes, threshold)
            result[f"f1@{label}"] = query_f1(pred_boxes, gt_boxes, threshold)
        per_query.append(result)
        by_type[str(source.get("eval_type", "unknown"))].append(result)

    metrics = {
        "evaluator": "local_official_compatible_hungarian_aabb",
        "union_mask": {
            "mIoU": float(np.mean([float(row["iou"]) for row in rows])),
            "Acc@0.25": float(np.mean([float(row["tp25"]) for row in rows])),
            "Acc@0.5": float(np.mean([float(row["tp50"]) for row in rows])),
        },
        "official_evaluator_available": False,
        "num_queries": len(per_query),
        "zero_target_queries": sum(result["num_gt"] == 0 for result in per_query),
        "missing_queries": 0,
        "duplicate_queries": 0,
        "overall": {
            "F1@0.25": float(np.mean([result["f1@0.25"] for result in per_query])),
            "F1@0.5": float(np.mean([result["f1@0.5"] for result in per_query])),
        },
        "by_eval_type": {
            eval_type: {
                "num_queries": len(values),
                "F1@0.25": float(np.mean([result["f1@0.25"] for result in values])),
                "F1@0.5": float(np.mean([result["f1@0.5"] for result in values])),
            }
            for eval_type, values in sorted(by_type.items())
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
