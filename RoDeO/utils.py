from typing import Optional, Tuple

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torchvision.ops import box_area, box_iou, generalized_box_iou


def hungarian_matching(
    preds: Tensor,
    targets: Tensor,
    shape_weight: Optional[float] = 1.0,
    class_weight: Optional[float] = 1.0
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Performs hungarian matching on a set of predicted and target boxes
    and returns the matched pairs, as well as the unmatched predicted and
    target boxes.

    Args:
        preds: Tensor of predicted boxes (M_p x >=4)
        targets: Tensor of target boxes (M_t x >=4)
        shape_weight: How to weight shape similarity in the cost matrix
        class_weight: How to weight class similarity in the cost matrix
    """
    # Perform matching
    shape_cost = -generalized_box_iou_xywh(preds[:, :4], targets[:, :4])
    class_cost = -(preds[:, 4][:, None] == targets[:, 4][None, :]).int()
    cost_matrix = shape_cost * shape_weight + class_cost * class_weight
    pred_inds, target_inds = linear_sum_assignment(cost_matrix.numpy())

    # Find matched predictions and targets
    matched_preds = preds[pred_inds]
    matched_targets = targets[target_inds]

    # Find unmatched predictions and targets
    unmatched_preds_mask = torch.ones(preds.shape[0]).scatter_(
        0, torch.from_numpy(pred_inds), 0).bool()
    unmatched_preds = preds[unmatched_preds_mask]
    unmatched_targets_mask = torch.ones(targets.shape[0]).scatter_(
        0, torch.from_numpy(target_inds), 0).bool()
    unmatched_targets = targets[unmatched_targets_mask]

    return (matched_preds,
            matched_targets,
            unmatched_preds,
            unmatched_targets)


def centered_iou(preds: Tensor, targets: Tensor) -> Tensor:
    r"""Computes the IoU of pairs of centered boxes (aligned at upper left).

    Args:
        preds: Tensor of predicted boxes (M x 4)
        targets: Tensor of target boxes (M x 4)
    """
    preds = preds.clone()
    # Align at upper left
    preds[:, :2] = targets[:, :2]
    # Compute IoUs
    ious = torch.diag(box_iou_xywh(preds[:, :4], targets[:, :4]))
    return ious


def to_x1y1x2y2(boxes: Tensor) -> Tensor:
    r"""Convert Tensor of boxes (M x >=4) from (x,y,w,h) to (x1,y1,x2,y2)"""
    boxes = boxes.clone()
    boxes[:, 2:4] += boxes[:, :2]
    return boxes


def box_area_xywh(boxes: Tensor) -> Tensor:
    r"""Performs box_area from torchvision with boxes in (x,y,w,h)"""
    return box_area(to_x1y1x2y2(boxes))


def box_iou_xywh(preds: Tensor, targets: Tensor) -> Tensor:
    r"""Performs box_iou from torchvision with boxes in (x,y,w,h)"""
    return box_iou(to_x1y1x2y2(preds), to_x1y1x2y2(targets))


def generalized_box_iou_xywh(preds: Tensor, targets: Tensor) -> Tensor:
    r"""Performs generalized_box_iou from torchvision with boxes in (x,y,w,h)"""
    return generalized_box_iou(to_x1y1x2y2(preds), to_x1y1x2y2(targets))


def get_center(boxes: Tensor) -> Tensor:
    r"""Boxes in (x,y,w,h) format. Returns the center of each box."""
    return boxes[:, :2] + (boxes[:, 2:4] / 2)


def harmonic_mean(x: Tensor, eps: Optional[float] = 1e-6) -> Tensor:
    return len(x) / (1 / (x + eps)).sum()
