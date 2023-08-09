from collections import defaultdict
from math import log
from typing import Dict, List, Optional
from warnings import warn

import numpy as np
from sklearn.metrics import matthews_corrcoef

from .utils import centered_iou, get_center, harmonic_mean, hungarian_matching


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
        is_3d: Optional[bool] = False,
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
        self.class_map: Dict[str, int] = {c: i for i, c in enumerate(class_names)}
        self.num_classes: int = len(self.class_names)
        self.w_matched: float = w_matched
        self.w_overpred: float = w_overpred
        self.w_missed: float = w_missed
        self.class_weight_matching: Optional[float] = class_weight_matching
        self.return_per_class: bool = return_per_class
        self.is_3d: bool = is_3d
        self.i_pos = 3 if is_3d else 2
        self.i_size = 6 if is_3d else 4
        self.i_cls = 6 if is_3d else 4
        self.i_conf = 7 if is_3d else 5

        self.pred_boxes: List[np.ndarray]
        self.target_boxes: List[np.ndarray]
        self.reset()

    def reset(self) -> None:
        r"""Reset the metric. Clear all stored predictions and targets."""
        self.pred_boxes = []
        self.target_boxes = []

    def add(
        self,
        preds: List[np.ndarray],
        targets: List[np.ndarray],
    ) -> None:
        r"""Add predictions and targets to the metric.

        Values in || are optional.

        Args:
            preds: List of predicted boxes. Each box is a (M_p, 5) array
                with (x, y, w, h, cls_id, |confidence|) in 2D and (M_p, 7) array
                with (x, y, z, w, h, d, cls_id, |confidence|) in 3D for each box.
            targets: List of target boxes. Each box is a (M_t, 5) array
                with (x, y, w, h, cls_id) in 2D and (M_p, 7) array
                with (x, y, z, w, h, d, cls_id) in 3D for each box.
        """
        assert len(preds) == len(targets)
        if self.is_3d:
            assert preds[0].shape[1] in [7, 8]
            assert targets[0].shape[1] == 7
        else:
            assert preds[0].shape[1] in [5, 6]
            assert targets[0].shape[1] == 5
        self.pred_boxes.extend(preds)
        self.target_boxes.extend(targets)

    def compute(self, cls_thresholds: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        pred_boxes = self.pred_boxes
        target_boxes = self.target_boxes
        if cls_thresholds is not None:
            assert len(cls_thresholds) == self.num_classes
            assert pred_boxes[0].shape[1] == (8 if self.is_3d else 6)
            # Filter predictions
            filtered_preds = []
            for pred in pred_boxes:
                filtered_pred = []
                for cls_name, threshold in cls_thresholds.items():
                    cls_id = self.class_map[cls_name]
                    filtered_pred.append(pred[(pred[:, self.i_cls] == cls_id) & (pred[:, self.i_conf] >= threshold)])
                filtered_preds.append(np.concatenate(filtered_pred))
            pred_boxes = filtered_preds
        return self.compute_(pred_boxes, target_boxes, return_per_class=self.return_per_class)

    def compute_(self,
                 pred_boxes: np.ndarray,
                 target_boxes: np.ndarray,
                 return_per_class: Optional[bool] = False) -> Dict[str, float]:
        r"""Compute the metric."""
        if self.class_weight_matching is None:
            class_weight = self._get_class_cost_for_matching(pred_boxes, target_boxes)
        else:
            class_weight = self.class_weight_matching

        # Get matchings
        matched_preds = []
        matched_targets = []
        unmatched_preds = []
        unmatched_targets = []
        for preds, targets in zip(pred_boxes,
                                  target_boxes):
            # Compute hungarian matching
            (matched_preds_,
             matched_targets_,
             unmatched_preds_,
             unmatched_targets_) = hungarian_matching(
                preds[:, :self.i_cls + 1],
                targets,
                class_weight=class_weight,
                is_3d=self.is_3d
            )

            matched_preds.append(matched_preds_)
            matched_targets.append(matched_targets_)
            unmatched_preds.append(unmatched_preds_)
            unmatched_targets.append(unmatched_targets_)

        # Each pred box now is (x,y,w,h,cls_id,|confidence|) / (x,y,z,w,h,d,cls_id,|confidence|)
        # Each target box now is (x,y,w,h,cls_id) / (x,y,z,w,h,d,cls_id)
        matched_preds = np.concatenate(matched_preds)  # (n_matched, 5/6/7/8)
        matched_targets = np.concatenate(matched_targets)  # (n_matched, 5/7)
        unmatched_preds = np.concatenate(unmatched_preds)  # (n_overpred, 5/6/7/8)
        unmatched_targets = np.concatenate(unmatched_targets)  # (n_underpred, 5/7)

        if len(matched_preds) == 0:
            warn("Unable to calculate RoDeO without predictions or targets. "
                 "Returning worst possible value.")
            keys = ['total', 'localization', 'shape_matching', 'classification',
                    'overprediction', 'underprediction']
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
        res['RoDeO/total'] = harmonic_mean(np.array([
            res['RoDeO/localization'],
            res['RoDeO/shape_matching'],
            res['RoDeO/classification']
        ])).item()

        # For each class
        if return_per_class:
            for i, cls_name in enumerate(self.class_names):
                key = f'RoDeO/{cls_name}'
                # Filter class
                matched_inds_c = matched_targets[:, self.i_cls] == i
                matched_preds_c = matched_preds[matched_inds_c]
                matched_targets_c = matched_targets[matched_inds_c]
                unmatched_preds_c = unmatched_preds[unmatched_preds[:, self.i_cls] == i]
                unmatched_targets_c = unmatched_targets[unmatched_targets[:, self.i_cls] == i]
                if len(matched_preds_c) == 0:
                    res[f'{key}/localization'] = 0.0
                    res[f'{key}/shape_matching'] = 0.0
                    res[f'{key}/classification'] = 0.0
                else:
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
                res[f'{key}/total'] = harmonic_mean(np.array([
                    res[f'{key}/localization'],
                    res[f'{key}/shape_matching'],
                    res[f'{key}/classification']
                ])).item()

        # Return results
        return res

    def confidence_sweep(self, n_thresholds: Optional[int] = None):
        r"""
        Perform a sweep over all possible confidence thresholds to find the
        best one for each class.

        Args:
            n_thresholds: Number of thresholds to test. If None, use the number
                          of unique confidence scores (default: None).
        """
        # Assert that confidence scores are available
        assert self.pred_boxes[0].shape[1] == (8 if self.is_3d else 6)

        pred_boxes = self.pred_boxes
        target_boxes = self.target_boxes

        # Get all confidence scores to check
        confidences = np.concatenate([pred_boxes[:, self.i_conf] for pred_boxes in pred_boxes])
        if n_thresholds is None:
            thresholds = np.unique(confidences)
        else:
            min_conf = confidences.min()
            max_conf = confidences.max()
            thresholds = np.linspace(min_conf, max_conf, n_thresholds)

        # Compute results for each threshold
        results = defaultdict(list)
        for threshold in thresholds:
            # Filter predictions
            filtered_preds = [pred[pred[:, self.i_conf] >= threshold] for pred in pred_boxes]
            # Compute results
            for key, value in self.compute_(filtered_preds, target_boxes, return_per_class=True).items():
                results[key].append(value)

        # Find best threshold for each class
        best_thresholds = {}
        best_scores = {}
        for cls_name in self.class_names:
            key = cls_name
            best_thresholds[key] = thresholds[np.argmax(results[f'RoDeO/{key}/total'])]
            best_scores[key] = results[f'RoDeO/{key}/total'][np.argmax(results[f'RoDeO/{key}/total'])]

        return best_thresholds, best_scores

    def _localization_score(
        self,
        matched_preds: np.ndarray,
        matched_targets: np.ndarray,
        unmatched_preds: np.ndarray,
        unmatched_targets: np.ndarray
    ) -> float:
        r"""Compute the localization score (Eq. 2 in the paper)."""
        # Normalize predictions and targets by size of target boxes
        target_sizes = matched_targets[:, self.i_pos:self.i_size].repeat(2, axis=1)  # (n_matched, 2/3)
        pred_center = get_center(matched_preds[:, :self.i_size] / target_sizes)  # (n_matched, 2/3)
        target_center = get_center(matched_targets[:, :self.i_size] / target_sizes)  # (n_matched, 2/3)
        # Get the euclidean distance between the centers
        matched_dists = np.power(pred_center - target_center, 2).sum(1)  # (n_matched)
        # Compute the score
        matched_score = np.exp(-matched_dists * log(2)).mean()
        unmatched_score = np.array(0.0)

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
        matched_preds: np.ndarray,
        matched_targets: np.ndarray,
        unmatched_preds: np.ndarray,
        unmatched_targets: np.ndarray
    ) -> float:
        r"""Centered IoUs between boxes. Unmatched boxes give IoU=0
        (Eq. 3 in the paper)."""
        matched_score = centered_iou(matched_preds, matched_targets, is_3d=self.is_3d).mean()
        unmatched_score = np.array(0.0)

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
        matched_preds: np.ndarray,
        matched_targets: np.ndarray,
        unmatched_preds: np.ndarray,
        unmatched_targets: np.ndarray
    ) -> float:
        r"""Clamped matthews correlation coefficient (Eq. 4 in the paper)."""
        pred_classes = matched_preds[:, self.i_cls]
        pred_multi_hot = np.zeros((len(pred_classes), self.num_classes))
        np.put_along_axis(pred_multi_hot, pred_classes[:, None].astype(np.int32), 1, 1)

        target_classes = matched_targets[:, self.i_cls]
        target_multi_hot = np.zeros((len(target_classes), self.num_classes))
        np.put_along_axis(target_multi_hot, target_classes[:, None].astype(np.int32), 1, 1)

        matched_score = np.array(matthews_corrcoef(
            pred_multi_hot.reshape(-1),
            target_multi_hot.reshape(-1)
        )).clip(min=0)
        unmatched_score = np.array(0.0)

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
        matched_preds: np.ndarray,
        unmatched_preds: np.ndarray,
        unmatched_targets: np.ndarray,
    ) -> np.ndarray:
        r"""Weight the scores according to the number of matched, overpredicted
        and missed boxes (Eq. 5 in the paper)."""
        matched = len(matched_preds) * self.w_matched
        overpred = len(unmatched_preds) * self.w_overpred
        missed = len(unmatched_targets) * self.w_missed
        total = matched + overpred + missed

        total_score = (matched_score * matched + (overpred + missed) * unmatched_score) / total

        return total_score

    def _get_class_cost_for_matching(
            self,
            pred_boxes: List[np.ndarray],
            target_boxes: List[np.ndarray]) -> np.ndarray:
        r"""Get the sample-level classification performance of the data"""
        if len(pred_boxes) == 0:
            return np.array(0.0)

        pred_multi_hot = np.zeros((len(pred_boxes), self.num_classes))
        for i, pred in enumerate(pred_boxes):
            np.put_along_axis(pred_multi_hot[i], pred[:, self.i_cls].astype(np.int32), 1, 0)

        target_multi_hot = np.zeros((len(pred_boxes), self.num_classes))
        for i, target in enumerate(target_boxes):
            np.put_along_axis(target_multi_hot[i], target[:, self.i_cls].astype(np.int32), 1, 0)

        mcc = matthews_corrcoef(pred_multi_hot.reshape(-1),
                                target_multi_hot.reshape(-1))
        class_cost = max(0, mcc)

        return class_cost
