"""Raw-coordinate box utilities for referential grounding evaluation."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


def _as_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape [N, 3], got {points.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates contain non-finite values")
    return points


def validate_mask_length(points: np.ndarray, mask: np.ndarray) -> np.ndarray:
    points = _as_points(points)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if len(mask) != len(points):
        raise ValueError(
            f"Mask/coordinate length mismatch: mask={len(mask)}, points={len(points)}"
        )
    return mask


def points_to_aabb(points: np.ndarray) -> Optional[np.ndarray]:
    points = _as_points(points)
    if len(points) == 0:
        return None
    return np.stack((points.min(axis=0), points.max(axis=0))).astype(np.float32)


def mask_to_aabb(coordinates: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    coordinates = _as_points(coordinates)
    mask = validate_mask_length(coordinates, mask)
    return points_to_aabb(coordinates[mask])


def aabb_iou(box_a: Optional[np.ndarray], box_b: Optional[np.ndarray]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)
    if box_a.shape != (2, 3) or box_b.shape != (2, 3):
        raise ValueError(f"Expected [2, 3] AABBs, got {box_a.shape} and {box_b.shape}")
    inter_extent = np.maximum(np.minimum(box_a[1], box_b[1]) - np.maximum(box_a[0], box_b[0]), 0.0)
    inter_volume = float(np.prod(inter_extent))
    volume_a = float(np.prod(np.maximum(box_a[1] - box_a[0], 0.0)))
    volume_b = float(np.prod(np.maximum(box_b[1] - box_b[0], 0.0)))
    union = volume_a + volume_b - inter_volume
    return 0.0 if union <= 0.0 else inter_volume / union


def aabb_minmax_to_corners(box: np.ndarray) -> np.ndarray:
    box = np.asarray(box, dtype=np.float32)
    if box.shape != (2, 3):
        raise ValueError(f"Expected [2, 3] AABB, got {box.shape}")
    mn, mx = box
    return np.asarray(
        [
            [mn[0], mn[1], mn[2]], [mn[0], mn[1], mx[2]],
            [mn[0], mx[1], mn[2]], [mn[0], mx[1], mx[2]],
            [mx[0], mn[1], mn[2]], [mx[0], mn[1], mx[2]],
            [mx[0], mx[1], mn[2]], [mx[0], mx[1], mx[2]],
        ],
        dtype=np.float32,
    )


def corners_to_aabb(corners: np.ndarray) -> np.ndarray:
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (8, 3):
        raise ValueError(f"Expected [8, 3] corners, got {corners.shape}")
    return points_to_aabb(corners)


def normalize_prediction_masks(prediction: np.ndarray, num_points: int) -> List[np.ndarray]:
    """Normalize model.generate output; 1-D output is the no-[SEG] sentinel."""
    prediction = np.asarray(prediction)
    if prediction.ndim == 1:
        if prediction.shape[0] != num_points:
            raise ValueError(f"1-D prediction has {prediction.shape[0]} points, expected {num_points}")
        return []
    if prediction.ndim != 2 or prediction.shape[1] != num_points:
        raise ValueError(f"Expected [num_masks, {num_points}], got {prediction.shape}")
    return [row.astype(bool, copy=False) for row in prediction]


def _largest_cluster(points: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    if len(points) == 0:
        return points
    labels = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1).fit_predict(points)
    valid = [label for label in np.unique(labels) if label >= 0]
    if not valid:
        return points
    return max((points[labels == label] for label in valid), key=len)


def scanrefer_mask_to_box(coordinates: np.ndarray, pred_mask: np.ndarray, eps: float, min_samples: int) -> Optional[np.ndarray]:
    coordinates = _as_points(coordinates)
    pred_mask = validate_mask_length(coordinates, pred_mask)
    foreground = coordinates[pred_mask]
    if len(foreground) == 0:
        return None
    return points_to_aabb(_largest_cluster(foreground, eps, min_samples))


def multi3drefer_mask_to_boxes(coordinates: np.ndarray, pred_mask: np.ndarray, eps: float) -> List[np.ndarray]:
    coordinates = _as_points(coordinates)
    pred_mask = validate_mask_length(coordinates, pred_mask)
    foreground = coordinates[pred_mask]
    if len(foreground) == 0:
        return []
    labels = DBSCAN(eps=eps, min_samples=1, n_jobs=-1).fit_predict(foreground)
    return [points_to_aabb(foreground[labels == label]) for label in sorted(np.unique(labels))]


def hungarian_true_positive(pred_boxes: Iterable[np.ndarray], gt_boxes: Iterable[np.ndarray], threshold: float) -> int:
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    if not pred_boxes or not gt_boxes:
        return 0
    ious = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            ious[i, j] = aabb_iou(pred, gt)
    rows, cols = linear_sum_assignment(-ious)
    return int(sum(ious[row, col] >= threshold for row, col in zip(rows, cols)))


def query_f1(pred_boxes: Iterable[np.ndarray], gt_boxes: Iterable[np.ndarray], threshold: float) -> float:
    pred_boxes = list(pred_boxes)
    gt_boxes = list(gt_boxes)
    if not gt_boxes:
        return 1.0 if not pred_boxes else 0.0
    if not pred_boxes:
        return 0.0
    tp = hungarian_true_positive(pred_boxes, gt_boxes, threshold)
    return 2.0 * tp / float(len(pred_boxes) + len(gt_boxes))


def json_box(box: Optional[np.ndarray]):
    return None if box is None else np.asarray(box, dtype=np.float32).tolist()
