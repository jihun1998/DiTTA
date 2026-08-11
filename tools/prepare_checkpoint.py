"""Convert a CFFM/SegFormer-style ISS checkpoint into a DiTTA checkpoint.

DiTTA's decode head adds two modules that no ISS checkpoint contains: the
temporal add-on (``query_layer`` / ``key_layer``) and the contrastive projection
(``proj_feat``).  In the original code base these were left at their random
initialisation, so the TTA result depended on the RNG state at model-build time.
This script freezes that choice into the checkpoint instead, which makes runs
reproducible across torch versions and machines.

The released checkpoint was produced with ``--addon`` pointing at the exact
initialisation used for the paper's experiments (see
``tools/dev_extract_paper_addon_init.py``).  When converting your own ISS
checkpoint, use the default ``seed:5123`` instead.

    python tools/prepare_checkpoint.py \
        --src  /path/to/segformer_b5_vspw2_baseline.pth \
        --out  checkpoints/ditta_segformer_b5_vspw.pth
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ditta.segformer import DiTTASegmentor  # noqa: E402

ADDON_PREFIXES = ('decode_head.query_layer.', 'decode_head.key_layer.',
                  'decode_head.proj_feat.')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--src', required=True,
                   help='ISS checkpoint holding backbone.* and decode_head.linear_*')
    p.add_argument('--out', required=True, help='output DiTTA checkpoint')
    p.add_argument('--addon', default='seed:5123',
                   help='"seed:<N>" to initialise the add-on randomly with that '
                        'seed, or a path to a .pth holding add-on weights')
    p.add_argument('--num_classes', type=int, default=124)
    return p.parse_args()


def main():
    args = parse_args()

    src = torch.load(args.src, map_location='cpu')
    meta = src.get('meta', {})
    src_sd = src.get('state_dict', src)

    if args.addon.startswith('seed:'):
        torch.manual_seed(int(args.addon.split(':', 1)[1]))
        addon_sd = {k: v for k, v in DiTTASegmentor(num_classes=args.num_classes)
                    .state_dict().items() if k.startswith(ADDON_PREFIXES)}
    else:
        addon = torch.load(args.addon, map_location='cpu')
        addon_sd = addon.get('state_dict', addon)
        addon_sd = {k: v for k, v in addon_sd.items() if k.startswith(ADDON_PREFIXES)}
    if not addon_sd:
        raise SystemExit(f'no add-on weights found in {args.addon}')

    model = DiTTASegmentor(num_classes=args.num_classes)
    wanted = set(model.state_dict())

    out_sd, taken_from_src = {}, 0
    for key in wanted:
        if key in addon_sd:
            out_sd[key] = addon_sd[key]
        elif key in src_sd:
            out_sd[key] = src_sd[key]
            taken_from_src += 1
    missing = sorted(wanted - set(out_sd))
    if missing:
        raise SystemExit(f'cannot fill {len(missing)} parameters, e.g. {missing[:5]}')

    model.load_state_dict(out_sd, strict=True)  # fails loudly on any mismatch
    dropped = sorted(set(src_sd) - wanted)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    torch.save({'state_dict': out_sd,
                'meta': {'CLASSES': meta.get('CLASSES'),
                         'PALETTE': meta.get('PALETTE'),
                         'ditta_addon_init': args.addon,
                         'source_checkpoint': os.path.basename(args.src)}},
               args.out)

    size = os.path.getsize(args.out) / 1e6
    print(f'wrote {args.out} ({size:.0f} MB)\n'
          f'  {taken_from_src} tensors from {args.src}\n'
          f'  {len(addon_sd)} add-on tensors from {args.addon}\n'
          f'  {len(dropped)} unused tensors dropped'
          + (f', e.g. {dropped[:3]}' if dropped else ''))


if __name__ == '__main__':
    main()
