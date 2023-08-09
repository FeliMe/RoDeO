from typing import Optional, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_matching(
    preds: np.ndarray,
    targets: np.ndarray,
    shape_weight: Optional[float] = 1.0,
    class_weight: Optional[float] = 1.0,
    is_3d: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Performs hungarian matching on a set of predicted and target boxes
    and returns the matched pairs, as well as the unmatched predicted and
    target boxes.

    Args:
        preds: Array of predicted boxes (M_p x 5/7), each (x,y,w,h,cls) or (x,y,z,w,h,d,cls)
        targets: Array of target boxes (M_t x 5/7), each (x,y,w,h,cls) or (x,y,z,w,h,d,cls)
        shape_weight: How to weight shape similarity in the cost matrix
        class_weight: How to weight class similarity in the cost matrix
        is_3d: Whether the boxes are 3D or not
    """
    assert preds.shape[1] == (7 if is_3d else 5) and targets.shape[1] == (7 if is_3d else 5)
    # Perform matching
    if is_3d:
        i_cls = 6
        shape_cost = -generalized_box_iou_3d(preds[:, :i_cls], targets[:, :i_cls])
    else:
        i_cls = 4
        shape_cost = -generalized_box_iou(preds[:, :i_cls], targets[:, :i_cls])
    class_cost = -(preds[:, i_cls][:, None] == targets[:, i_cls][None, :]).astype(np.int32)
    cost_matrix = shape_cost * shape_weight + class_cost * class_weight
    pred_inds, target_inds = linear_sum_assignment(cost_matrix)

    # Find matched predictions and targets
    matched_preds = preds[pred_inds]
    matched_targets = targets[target_inds]

    # Find unmatched predictions and targets
    unmatched_preds_mask = np.ones(preds.shape[0], dtype=bool)
    unmatched_preds_mask[pred_inds] = False
    unmatched_preds = preds[unmatched_preds_mask]
    unmatched_targets_mask = np.ones(targets.shape[0], dtype=bool)
    unmatched_targets_mask[target_inds] = False
    unmatched_targets = targets[unmatched_targets_mask]

    return (matched_preds,
            matched_targets,
            unmatched_preds,
            unmatched_targets)


def centered_iou(preds: np.ndarray, targets: np.ndarray, is_3d: Optional[bool] = False) -> np.ndarray:
    r"""Computes the IoU of pairs of centered boxes (aligned at upper left).

    Args:
        preds: Tensor of predicted boxes (M_p x 4+), each (x,y,w,h,...) or (x,y,z,w,h,d,...)
        targets: Tensor of target boxes (M_t x 4+), each (x,y,w,h,...) or (x,y,z,w,h,d,...)
        is_3d: Whether the boxes are 3D or not
    """
    preds = preds.copy()
    if is_3d:
        # Align at upper left
        preds[:, :3] = targets[:, :3]
        ious = np.diag(box_iou_3d(preds[:, :6], targets[:, :6]))
    else:
        # Align at upper left
        preds[:, :2] = targets[:, :2]
        ious = np.diag(box_iou(preds[:, :4], targets[:, :4]))
    return ious


def get_center(boxes: np.ndarray) -> np.ndarray:
    r"""Return the center of each box

    Args:
        boxes: Tensor of boxes in format (x,y,w,h) for 2D, (x,y,z,h,w,d) for 3D
    Returns:
        centers: Tensor of centers in format (x,y) for 2D, (x,y,z) for 3D
    """
    if boxes.shape[1] == 4:  # 2D
        return boxes[:, :2] + (boxes[:, 2:4] / 2)
    elif boxes.shape[1] == 6:  # 3D
        return boxes[:, :3] + (boxes[:, 3:6] / 2)
    else:
        raise ValueError("Boxes must be 2D or 3D")


def harmonic_mean(x: np.ndarray, eps: Optional[float] = 1e-6) -> np.ndarray:
    return len(x) / (1 / (x + eps)).sum()


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    r"""Compute the intersection over union between all samples in two sets
    of bounding boxes in the (x,y,w,h) format."""
    # Compute the coordinates of the bounding boxes' corners
    x1_min, y1_min = boxes1[:, 0], boxes1[:, 1]
    x1_max, y1_max = x1_min + boxes1[:, 2], y1_min + boxes1[:, 3]
    x2_min, y2_min = boxes2[:, 0], boxes2[:, 1]
    x2_max, y2_max = x2_min + boxes2[:, 2], y2_min + boxes2[:, 3]

    # Compute the areas of the bounding boxes
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # Compute the intersection coordinates and areas
    xA = np.maximum(x1_min[:, np.newaxis], x2_min)
    yA = np.maximum(y1_min[:, np.newaxis], y2_min)
    xB = np.minimum(x1_max[:, np.newaxis], x2_max)
    yB = np.minimum(y1_max[:, np.newaxis], y2_max)
    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Compute the union areas
    union_area = area1[:, np.newaxis] + area2 - inter_area

    # Compute the intersection over union
    return inter_area / union_area


def box_iou_3d(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    r"""Compute the intersection over union between all samples in two sets
    of bounding boxes in the (x,y,z,w,h,d) format."""
    # Compute the coordinates of the bounding boxes' corners
    x1_min, y1_min, z1_min = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2]
    x1_max, y1_max, z1_max = x1_min + boxes1[:, 3], y1_min + boxes1[:, 4], z1_min + boxes1[:, 5]
    x2_min, y2_min, z2_min = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2]
    x2_max, y2_max, z2_max = x2_min + boxes2[:, 3], y2_min + boxes2[:, 4], z2_min + boxes2[:, 5]

    # Compute the areas of the bounding boxes
    area1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]
    area2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]

    # Compute the intersection coordinates and areas
    xA = np.maximum(x1_min[:, np.newaxis], x2_min)
    yA = np.maximum(y1_min[:, np.newaxis], y2_min)
    zA = np.maximum(z1_min[:, np.newaxis], z2_min)
    xB = np.minimum(x1_max[:, np.newaxis], x2_max)
    yB = np.minimum(y1_max[:, np.newaxis], y2_max)
    zB = np.minimum(z1_max[:, np.newaxis], z2_max)
    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA) * np.maximum(0, zB - zA)

    # Compute the union areas
    union_area = area1[:, np.newaxis] + area2 - inter_area

    # Compute the intersection over union
    return inter_area / union_area


def generalized_box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    r"""Compute the generalized intersection over union between all samples
    in two sets of bounding boxes in the (x,y,w,h) format."""
    # Compute the coordinates of the bounding boxes' corners
    x1_min, y1_min = boxes1[:, 0], boxes1[:, 1]
    x1_max, y1_max = x1_min + boxes1[:, 2], y1_min + boxes1[:, 3]
    x2_min, y2_min = boxes2[:, 0], boxes2[:, 1]
    x2_max, y2_max = x2_min + boxes2[:, 2], y2_min + boxes2[:, 3]

    # Compute the areas of the bounding boxes
    area1 = boxes1[:, 2] * boxes1[:, 3]
    area2 = boxes2[:, 2] * boxes2[:, 3]

    # Compute the intersection coordinates and areas
    xA = np.maximum(x1_min[:, np.newaxis], x2_min)
    yA = np.maximum(y1_min[:, np.newaxis], y2_min)
    xB = np.minimum(x1_max[:, np.newaxis], x2_max)
    yB = np.minimum(y1_max[:, np.newaxis], y2_max)
    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Compute the union areas
    union_area = area1[:, np.newaxis] + area2 - inter_area

    # Compute the enclosing box
    xC = np.minimum(x1_min[:, np.newaxis], x2_min)
    yC = np.minimum(y1_min[:, np.newaxis], y2_min)
    xD = np.maximum(x1_max[:, np.newaxis], x2_max)
    yD = np.maximum(y1_max[:, np.newaxis], y2_max)
    enc_area = np.maximum(0, xD - xC) * np.maximum(0, yD - yC)

    # Compute the generalized intersection over union
    return inter_area / union_area - (enc_area - union_area) / enc_area


def generalized_box_iou_3d(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    r"""Compute the generalized intersection over union between all samples
    in two sets of bounding boxes in the (x,y,z,w,h,d) format."""
    # Compute the coordinates of the bounding boxes' corners
    x1_min, y1_min, z1_min = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2]
    x1_max, y1_max, z1_max = x1_min + boxes1[:, 3], y1_min + boxes1[:, 4], z1_min + boxes1[:, 5]
    x2_min, y2_min, z2_min = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2]
    x2_max, y2_max, z2_max = x2_min + boxes2[:, 3], y2_min + boxes2[:, 4], z2_min + boxes2[:, 5]

    # Compute the areas of the bounding boxes
    area1 = boxes1[:, 3] * boxes1[:, 4] * boxes1[:, 5]
    area2 = boxes2[:, 3] * boxes2[:, 4] * boxes2[:, 5]

    # Compute the intersection coordinates and areas
    xA = np.maximum(x1_min[:, np.newaxis], x2_min)
    yA = np.maximum(y1_min[:, np.newaxis], y2_min)
    zA = np.maximum(z1_min[:, np.newaxis], z2_min)
    xB = np.minimum(x1_max[:, np.newaxis], x2_max)
    yB = np.minimum(y1_max[:, np.newaxis], y2_max)
    zB = np.minimum(z1_max[:, np.newaxis], z2_max)
    inter_area = np.maximum(0, xB - xA) * np.maximum(0, yB - yA) * np.maximum(0, zB - zA)

    # Compute the union areas
    union_area = area1[:, np.newaxis] + area2 - inter_area

    # Compute the enclosing box
    xC = np.minimum(x1_min[:, np.newaxis], x2_min)
    yC = np.minimum(y1_min[:, np.newaxis], y2_min)
    zC = np.minimum(z1_min[:, np.newaxis], z2_min)
    xD = np.maximum(x1_max[:, np.newaxis], x2_max)
    yD = np.maximum(y1_max[:, np.newaxis], y2_max)
    zD = np.maximum(z1_max[:, np.newaxis], z2_max)
    enc_area = np.maximum(0, xD - xC) * np.maximum(0, yD - yC) * np.maximum(0, zD - zC)

    # Compute the generalized intersection over union
    return inter_area / union_area - (enc_area - union_area) / enc_area
