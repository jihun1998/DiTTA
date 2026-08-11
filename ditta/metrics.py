"""Evaluation metrics for VSPW: mIoU, weighted IoU and video consistency.

Semantics follow the VSPW benchmark and the original DiTTA code:

* ``label = raw_gt - 1`` so that VSPW's 0 ("unlabelled") becomes the ignore
  index and classes run 0..123;
* mIoU averages IoU over classes that actually appear in the ground truth seen
  so far, not over all 124;
* mVC_n is measured per video, over the frames the model is evaluated on: the
  fraction of pixels that keep a correct, constant prediction across every
  window of ``n`` consecutive frames whose ground truth is itself constant.
"""

import numpy as np
import torch

IGNORE_INDEX = 255


def prepare_gt(raw_gt):
    """VSPW mask (0 = unlabelled, 1..124 = classes) -> label with 255 = ignore."""
    label = torch.as_tensor(np.asarray(raw_gt)).to(torch.int64) - 1
    label[(label < 0) | (label == 254)] = IGNORE_INDEX
    return label


class SegmentationMetric:
    """Running confusion matrix over the whole evaluation set."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.confusion = np.zeros((self.num_classes,) * 2, dtype=np.float64)

    def add(self, pred, label):
        """``pred`` and ``label`` are HxW arrays; 255 in ``label`` is ignored."""
        pred = np.asarray(pred)
        label = np.asarray(label)
        assert pred.shape == label.shape
        valid = (label >= 0) & (label < self.num_classes)
        index = self.num_classes * label[valid].astype(np.int64) + pred[valid]
        self.confusion += np.bincount(
            index, minlength=self.num_classes ** 2).reshape(self.confusion.shape)

    @property
    def _iou(self):
        inter = np.diag(self.confusion)
        union = (self.confusion.sum(axis=1) + self.confusion.sum(axis=0) - inter)
        with np.errstate(divide='ignore', invalid='ignore'):
            return inter / union

    def miou(self):
        """Mean IoU over the classes present in the ground truth."""
        present = self.confusion.sum(axis=1) > 0
        return float(np.nansum(self._iou * present) / max(present.sum(), 1))

    def wiou(self):
        """Frequency-weighted IoU."""
        freq = self.confusion.sum(axis=1) / max(self.confusion.sum(), 1)
        iou = self._iou
        return float((freq[freq > 0] * iou[freq > 0]).sum())

    def iou_per_class(self):
        return self._iou


class VideoConsistency:
    """mVC_n accumulated over videos."""

    def __init__(self, windows=(8, 16)):
        self.windows = tuple(windows)
        self.scores = {n: [] for n in self.windows}

    def add_video(self, preds, labels):
        """``preds`` / ``labels``: lists of HxW tensors for one video, in order."""
        for n in self.windows:
            for i in range(len(labels) - n):
                gt_const = torch.ones_like(labels[i], dtype=torch.bool)
                pred_const = torch.ones_like(labels[i], dtype=torch.bool)
                for j in range(1, n):
                    gt_const &= labels[i] == labels[i + j]
                    pred_const &= preds[i] == preds[i + j]
                denom = gt_const.sum()
                self.scores[n].append(
                    float((pred_const & gt_const).sum() / denom) if denom > 0
                    else float('nan'))

    def value(self, n):
        if not self.scores[n]:
            return float('nan')
        return float(np.nanmean(np.array(self.scores[n])))

    def summary(self):
        return {f'mVC{n}': self.value(n) for n in self.windows}
