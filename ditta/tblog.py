"""TensorBoard logging, with optional comparison against a reference run.

Tag names and step semantics match the original DiTTA code base, so a run of
this repository and a run of the original can be opened in the same TensorBoard
and read off the same axes:

    0_mIoU, 1_fwIoU           running metrics, one point per evaluated frame
    0_mIoU_instant, ...       the same metrics for that frame alone
    VC8, VC16                 running video consistency, one point per video
    loss/<name>               one point per TTA step
    IoU_classwise/<class>     per-class IoU (optional: it dominates file size)

Passing ``--reference`` additionally plots the reference curve as ``paper/<tag>``
and the signed difference as ``delta/<tag>`` on the live run's own axes, so the
gap is visible while the run is still going.
"""

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:                                       # tensorboard is optional
    SummaryWriter = None


class ReferenceCurves:
    """Scalar curves exported by tools/export_reference_curves.py."""

    def __init__(self, path):
        data = np.load(path)
        self.curves = {}
        for key in data.files:
            if key.endswith('/step'):
                tag = key[:-len('/step')]
                self.curves[tag] = (data[key], data[f'{tag}/value'])

    def at(self, tag, step):
        """Reference value at ``step``, or None if the curve does not reach it."""
        if tag not in self.curves:
            return None
        steps, values = self.curves[tag]
        i = int(np.searchsorted(steps, step))
        if i >= len(steps):
            return None
        return float(values[i])

    def final(self, tag):
        if tag not in self.curves or not len(self.curves[tag][1]):
            return None
        return float(self.curves[tag][1][-1])


class Logger:
    """Thin SummaryWriter wrapper; a no-op when TensorBoard is unavailable."""

    def __init__(self, log_dir, reference=None, classwise=False, classes=()):
        self.writer = SummaryWriter(log_dir) if SummaryWriter is not None else None
        if SummaryWriter is None:
            print('[tb] tensorboard is not installed, logging disabled')
        self.reference = ReferenceCurves(reference) if reference else None
        self.classwise = classwise
        self.classes = classes
        self.tta_step = 0        # counts optimisation steps across all videos

    def scalar(self, tag, value, step):
        """Log a scalar, plus the reference and the difference when available."""
        if self.writer is None:
            return
        self.writer.add_scalar(tag, value, global_step=step)
        if self.reference is not None:
            ref = self.reference.at(tag, step)
            if ref is not None:
                self.writer.add_scalar(f'paper/{tag}', ref, global_step=step)
                self.writer.add_scalar(f'delta/{tag}', value - ref, global_step=step)

    def losses(self, values):
        for name, value in values.items():
            self.scalar(f'loss/{name}', value, self.tta_step)
        self.tta_step += 1

    def frame_metrics(self, metric, instant, step):
        self.scalar('0_mIoU', metric.miou(), step)
        self.scalar('1_fwIoU', metric.wiou(), step)
        self.scalar('0_mIoU_instant', instant.miou(), step)
        self.scalar('1_fwIoU_instant', instant.wiou(), step)
        if self.classwise and self.writer is not None:
            for i, iou in enumerate(metric.iou_per_class()):
                if not np.isnan(iou):
                    self.writer.add_scalar(f'IoU_classwise/{self.classes[i]}',
                                           iou, global_step=step)

    def video_metrics(self, consistency, step):
        for tag, value in consistency.summary().items():
            self.scalar(tag.replace('mVC', 'VC'), value, step)

    def summary_table(self, result):
        """Final numbers, next to the reference ones, as a TensorBoard text panel."""
        if self.writer is None:
            return
        rows = ['| metric | this run | paper | delta |', '|---|---|---|---|']
        for key, tag in (('mIoU', '0_mIoU'), ('wIoU', '1_fwIoU'),
                         ('mVC8', 'VC8'), ('mVC16', 'VC16')):
            value = result.get(key)
            ref = self.reference.final(tag) if self.reference else None
            rows.append(f'| {key} | {100 * value:.2f} | '
                        + (f'{100 * ref:.2f} | {100 * (value - ref):+.2f} |'
                           if ref is not None else '- | - |'))
        self.writer.add_text('summary', '\n'.join(rows))

    def close(self):
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
