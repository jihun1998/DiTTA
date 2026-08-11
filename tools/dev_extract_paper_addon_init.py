"""Dump the add-on initialisation used for the paper's experiments.

Development utility: it needs the original CFFM code base (the repo DiTTA was
developed in) on the import path, because it reproduces the exact random
initialisation that `tools/run_tta_final.py` produced -- seed 5123, then
`build_segmentor`.  Those weights are baked into the released checkpoint so the
reported numbers can be reproduced without depending on RNG order.

    python tools/dev_extract_paper_addon_init.py \
        --legacy_root /path/to/VSS-CFFM \
        --config local_configs/cffm/B5/cffm_tta_contra_attn_final.py \
        --out checkpoints/ditta_addon_paper_init.pth
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

ADDON_PREFIXES = ('decode_head.query_layer.', 'decode_head.key_layer.',
                  'decode_head.proj_feat.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--legacy_root', required=True, help='path to the CFFM repo')
    p.add_argument('--config', default='local_configs/cffm/B5/cffm_tta_contra_attn_final.py')
    p.add_argument('--out', required=True)
    p.add_argument('--seed', type=int, default=5123, help='seed used by run_tta_final.py')
    args = p.parse_args()

    sys.path.insert(0, args.legacy_root)
    os.chdir(args.legacy_root)
    import mmcv
    from mmseg.models import build_segmentor

    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    sd = {k: v.clone() for k, v in model.state_dict().items()
          if k.startswith(ADDON_PREFIXES)}
    out = os.path.abspath(os.path.join(args.legacy_root, args.out)) \
        if not os.path.isabs(args.out) else args.out
    torch.save({'state_dict': sd,
                'meta': {'seed': args.seed, 'config': args.config}}, out)
    print(f'wrote {out} with {len(sd)} tensors:')
    for k in sorted(sd):
        print(f'  {k} {tuple(sd[k].shape)}')


if __name__ == '__main__':
    main()
