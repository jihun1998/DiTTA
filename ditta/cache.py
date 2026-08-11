"""Optional on-disk cache for the SAM2 distillation targets.

DiTTA needs no cache: a full run recomputes everything per video.  But when you
sweep TTA hyper-parameters, stage B is the expensive part and does not depend on
them, so ``--cache_dir`` lets you pay for it once.  One compressed .npz per
video, a few MB each.
"""

import os

import numpy as np
import torch


class TargetCache:
    def __init__(self, root, device='cuda'):
        self.root = root
        self.device = device
        os.makedirs(root, exist_ok=True)

    def _path(self, name):
        return os.path.join(self.root, f'{name}.npz')

    def load_targets(self, name, warmup_frames):
        """Return cached targets, or None on a miss."""
        path = self._path(name)
        if not os.path.isfile(path):
            return None
        with np.load(path) as data:
            frames = sorted(int(k.split('_')[0]) for k in data.files if k.endswith('_id'))
            if frames != list(warmup_frames):
                return None            # different warm-up ratio: recompute
            return {i: {'obj_id': torch.from_numpy(data[f'{i}_id']).to(self.device),
                        'score': torch.from_numpy(data[f'{i}_score']).to(self.device)}
                    for i in frames}

    def save_targets(self, name, targets):
        arrays = {}
        for i, target in targets.items():
            arrays[f'{i}_id'] = target['obj_id'].cpu().numpy().astype(np.int64)
            arrays[f'{i}_score'] = target['score'].cpu().numpy().astype(np.float32)
        np.savez_compressed(self._path(name), **arrays)
