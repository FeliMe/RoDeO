from math import log
from typing import Dict, List, Optional
from warnings import warn

import torch
from torch import Tensor
from torchmetrics.functional.classification import binary_matthews_corrcoef
from utils import centered_iou, get_center, harmonic_mean, hungarian_matching


class RoDeO:
    r"""Robust Detection Outcome (RoDeO).

    Computes five detection properties after a hungarian matching:
    1. Classification (Matthews Correlation Coefficient)
    2. Localization (Distance between centers of matched boxes)
    3. Shape Matching (Centered Box IoU)

    The summary metric is the harmonic mean of the above.
    """
    def __init__(
        self,
        class_names: List[str],
        w_matched: Optional[float] = 1.0,
        w_overpred: Optional[float] = 1.0,
        w_missed: Optional[float] = 1.0,
        class_weight_matching: Optional[float] = None,
        return_per_class: Optional[bool] = False,
    ) -> None:
        r"""Metric class for Robust Detection Outcome (RoDeO).

        Args:
            class_names: List of possible class names.
            w_matched: Weight for matched boxes when weighting the scores.
                Eq. (5) in the paper (default: 1.0).
            w_overpred: Weight for overpredicted boxes when weighting the scores.
                Eq. (5) in the paper (default: 1.0).
            w_missed: Weight for missed boxes when weighting the scores.
                Eq. (5) in the paper (default: 1.0).
            class_weight_matching: Weight for class matching when computing
                the hungarian matching. Will be computed from the
                classification performance if not specified (default).
            return_per_class: Whether to return the scores for each class
                (default: False).
        """
        assert len(class_names) > 1
        self.class_names: List[str] = class_names
        self.num_classes: int = len(self.class_names)
        self.w_matched: float = w_matched
        self.w_overpred: float = w_overpred
        self.w_missed: float = w_missed
        self.class_weight_matching: Optional[float] = class_weight_matching
        self.return_per_class: bool = return_per_class

        self.pred_boxes: List[Tensor]
        self.target_boxes: List[Tensor]
        self.reset()

    def reset(self) -> None:
        r"""Reset the metric. Clear all stored predictions and targets."""
        self.pred_boxes = []
        self.target_boxes = []

    def add(
        self,
        preds: List[Tensor],
        targets: List[Tensor],
    ) -> None:
        r"""Add predictions and targets to the metric.

        Args:
            preds: List of predicted boxes. Each box is a (M_p, 5) tensor
                with (x, y, w, h, cls_id) for each box.
            targets: List of target boxes. Each box is a (M_t, 5) tensor
                with (x, y, w, h, cls_id) for each box.
        """
        assert len(preds) == len(targets)
        self.pred_boxes.extend(preds)
        self.target_boxes.extend(targets)

    def compute(self) -> Dict[str, float]:
        r"""Compute the metric."""
        if self.class_weight_matching is None:
            class_weight = self._get_class_cost_for_matching()
        else:
            class_weight = self.class_weight_matching

        # Get matchings
        matched_preds = []
        matched_targets = []
        unmatched_preds = []
        unmatched_targets = []
        for preds, targets in zip(self.pred_boxes,
                                  self.target_boxes):
            # Compute hungarian matching
            (matched_preds_,
             matched_targets_,
             unmatched_preds_,
             unmatched_targets_) = hungarian_matching(preds, targets, class_weight=class_weight)

            matched_preds.append(matched_preds_)
            matched_targets.append(matched_targets_)
            unmatched_preds.append(unmatched_preds_)
            unmatched_targets.append(unmatched_targets_)

        # Each box now is (x, y, w, h, cls_id)
        matched_preds = torch.cat(matched_preds)  # (n_matched, 5)
        matched_targets = torch.cat(matched_targets)  # (n_matched, 5)
        unmatched_preds = torch.cat(unmatched_preds)  # (n_overpred, 5)
        unmatched_targets = torch.cat(unmatched_targets)  # (n_underpred, 5)

        if len(matched_preds) == 0:
            warn("Unable to calculate RoDeO without predictions or targets. Returning worst possible value.")
            keys = ['total', 'localization', 'shape_matching', 'classification', 'overprediction', 'underprediction']
            classes = ['/' + c for c in self.class_names] + [''] if self.return_per_class else ['']
            return {f'RoDeO{cls}/{k}': 0.0 for cls in classes for k in keys}

        # Compute scores
        res = {}
        res['RoDeO/localization'] = self._localization_score(
            matched_preds,
            matched_targets,
            unmatched_preds,
            unmatched_targets,
        )
        res['RoDeO/shape_matching'] = self._shape_matching_score(
            matched_preds,
            matched_targets,
            unmatched_preds,
            unmatched_targets
        )
        res['RoDeO/classification'] = self._classification_score(
            matched_preds,
            matched_targets,
            unmatched_preds,
            unmatched_targets
        )

        # Combine with (harmonic) mean (Eq. 6 in the paper)
        res['RoDeO/total'] = harmonic_mean(torch.tensor([
            res['RoDeO/localization'],
            res['RoDeO/shape_matching'],
            res['RoDeO/classification']
        ])).item()

        # For each class
        if self.return_per_class:
            for i, cls_name in enumerate(self.class_names):
                key = f'RoDeO/{cls_name}'
                # Filter class
                matched_inds_c = matched_targets[:, 4] == i
                matched_preds_c = matched_preds[matched_inds_c]
                matched_targets_c = matched_targets[matched_inds_c]
                unmatched_preds_c = unmatched_preds[unmatched_preds[:, 4] == i]
                unmatched_targets_c = unmatched_targets[unmatched_targets[:, 4] == i]
                # Compute scores
                res[f'{key}/localization'] = self._localization_score(
                    matched_preds_c,
                    matched_targets_c,
                    unmatched_preds_c,
                    unmatched_targets_c,
                )
                res[f'{key}/shape_matching'] = self._shape_matching_score(
                    matched_preds_c,
                    matched_targets_c,
                    unmatched_preds_c,
                    unmatched_targets_c
                )
                res[f'{key}/classification'] = self._classification_score(
                    matched_preds_c,
                    matched_targets_c,
                    unmatched_preds_c,
                    unmatched_targets_c
                )
                # Combine with (harmonic) mean (Eq. 6 in the paper)
                res[f'{key}/total'] = harmonic_mean(torch.tensor([
                    res[f'{key}/localization'],
                    res[f'{key}/shape_matching'],
                    res[f'{key}/classification']
                ])).item()

        # Return results
        return res

    def _localization_score(
        self,
        matched_preds: Tensor,
        matched_targets: Tensor,
        unmatched_preds: Tensor,
        unmatched_targets: Tensor
    ) -> float:
        r"""Compute the localization score (Eq. 2 in the paper)."""
        # Normalize predictions and targets by size of target boxes
        target_sizes = torch.cat([matched_targets[:, 2:4], matched_targets[:, 2:4]], 1)  # (n_matched, 2)
        pred_center = get_center(matched_preds[:, :4] / target_sizes)  # (n_matched, 2)
        target_center = get_center(matched_targets[:, :4] / target_sizes)  # (n_matched, 2)
        # Get the euclidean distance between the centers
        matched_dists = (pred_center - target_center).pow(2).sum(1)  # (n_matched)
        # Compute the score
        matched_score = (-matched_dists * log(2)).exp().mean()
        unmatched_score = torch.tensor(0.0)

        loc_score = self._weight_scores(
            matched_score,
            unmatched_score,
            matched_preds,
            unmatched_preds,
            unmatched_targets,
        )
        return loc_score.item()

    def _shape_matching_score(
        self,
        matched_preds: Tensor,
        matched_targets: Tensor,
        unmatched_preds: Tensor,
        unmatched_targets: Tensor
    ) -> float:
        r"""Centered IoUs between boxes. Unmatched boxes give IoU=0
        (Eq. 3 in the paper)."""
        matched_score = centered_iou(matched_preds, matched_targets).mean()
        unmatched_score = torch.tensor(0.0)

        shape_score = self._weight_scores(
            matched_score,
            unmatched_score,
            matched_preds,
            unmatched_preds,
            unmatched_targets,
        )
        return shape_score.item()

    def _classification_score(
        self,
        matched_preds: Tensor,
        matched_targets: Tensor,
        unmatched_preds: Tensor,
        unmatched_targets: Tensor
    ) -> float:
        r"""Clamped matthews correlation coefficient
        (Eq. 4 in the paper)."""
        pred_classes = matched_preds[:, 4]
        target_classes = matched_targets[:, 4]
        pred_multi_hot = torch.zeros(
            len(pred_classes), self.num_classes
        ).scatter_(1, pred_classes[:, None].long(), 1)
        target_multi_hot = torch.zeros(
            len(target_classes), self.num_classes
        ).scatter_(1, target_classes[:, None].long(), 1)
        matched_score = binary_matthews_corrcoef(
            pred_multi_hot, target_multi_hot).clamp_min(0.0)
        unmatched_score = torch.tensor(0.0)

        cls_score = self._weight_scores(
            matched_score,
            unmatched_score,
            matched_preds,
            unmatched_preds,
            unmatched_targets,
        )
        return cls_score.item()

    def _weight_scores(
        self,
        matched_score: float,
        unmatched_score: float,
        matched_preds: Tensor,
        unmatched_preds: Tensor,
        unmatched_targets: Tensor,
    ) -> Tensor:
        r"""Weight the scores according to the number of matched, overpredicted
        and missed boxes (Eq. 5 in the paper)."""
        matched = len(matched_preds) * self.w_matched
        overpred = len(unmatched_preds) * self.w_overpred
        missed = len(unmatched_targets) * self.w_missed
        total = matched + overpred + missed
        total_score = (matched_score * matched + (overpred + missed) * unmatched_score) / total
        return total_score

    def _get_class_cost_for_matching(self):
        r"""Get the sample-level classification performance of the data"""
        if len(self.pred_boxes) == 0:
            return torch.tensor(0.0)
        pred_multi_hot = torch.stack([
            torch.zeros(self.num_classes).scatter_(0, pred[:, 4].long(), 1) for pred in self.pred_boxes])
        target_multi_hot = torch.stack([
            torch.zeros(self.num_classes).scatter_(0, target[:, 4].long(), 1) for target in self.target_boxes])
        mcc = binary_matthews_corrcoef(pred_multi_hot, target_multi_hot)
        class_cost = mcc.clamp_min(0.)
        return class_cost


if __name__ == '__main__':
    # Init RoDeO with two classes
    rodeo = RoDeO(class_names=['a', 'b'])
    # Add some predictions and targets
    pred = [torch.tensor([[0.1, 0.1, 0.2, 0.1, 0.0],
                          [0.0, 0.3, 0.1, 0.1, 1.0],
                          [0.2, 0.2, 0.1, 0.1, 0.0]])]
    target = [torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.0],
                            [0.0, 0.2, 0.1, 0.1, 1.0]])]
    rodeo.add(pred, target)
    # Compute the score
    score = rodeo.compute()
    for key, val in score.items():
        print(f'{key}: {val}')
