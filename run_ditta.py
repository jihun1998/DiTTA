"""DiTTA on VSPW: one command runs the whole method.

For every video in the split, this
  1. runs the frozen ISS model on the warm-up frames,
  2. builds SAM2 distillation targets from those predictions,
  3. adapts a copy of the ISS model on the warm-up frames,
  4. evaluates the adapted model on the remaining frames,
and reports mIoU / wIoU / mVC over the whole split.

    python run_ditta.py \
        --checkpoint checkpoints/ditta_segformer_b5_vspw.pth \
        --sam2_checkpoint checkpoints/sam2_hiera_large.pt
"""

import argparse
import json
import os
import random
import time

# The temporal add-on materialises a (H*W) x (H*W) attention matrix -- 2.5 GiB at
# VSPW's 480p -- and over a few hundred videos the caching allocator fragments
# enough that no contiguous block that large is left: a full sweep on a 24 GB card
# used to die of OOM around video 230 of 343, with memory to spare. Expandable
# segments fix that. This has to be set before torch is imported.
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import numpy as np  # noqa: E402
import torch  # noqa: E402
import tqdm  # noqa: E402

from ditta.cache import TargetCache
from ditta.config import format_config, load_config
from ditta.data import VSPW_CLASSES, list_videos
from ditta.metrics import SegmentationMetric, VideoConsistency
from ditta.pipeline import run_video
from ditta.sam2_target import DistillTargetBuilder
from ditta.segformer import build_model
from ditta.tblog import Logger


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/vspw_b5_q10.py')
    p.add_argument('--checkpoint', default='checkpoints/ditta_segformer_b5_vspw.pth',
                   help='ISS checkpoint prepared by tools/prepare_checkpoint.py')
    p.add_argument('--sam2_checkpoint', default='checkpoints/sam2_hiera_large.pt')
    p.add_argument('--out', default=None,
                   help='output directory (default: exps/<timestamp>_q<ratio>)')
    p.add_argument('--cache_dir', default=None,
                   help='reuse SAM2 distillation targets across runs')
    p.add_argument('--videos', type=int, default=None,
                   help='only run the first N videos (smoke test)')
    p.add_argument('--video', action='append', default=None,
                   help='run a specific video; repeatable')
    p.add_argument('--seed', type=int, default=5123)
    p.add_argument('--baseline', action='store_true',
                   help='skip adaptation and evaluate the frozen ISS model, '
                        'reproducing the ISS baseline row')
    p.add_argument('--device', default='cuda')
    # logging
    p.add_argument('--no_tensorboard', action='store_true')
    p.add_argument('--reference', default=None,
                   help='.npz of reference curves (tools/export_reference_curves.py); '
                        'plotted as paper/<tag> with the difference as delta/<tag>')
    p.add_argument('--tb_classwise', action='store_true',
                   help='also log per-class IoU (large event files)')
    # config overrides
    p.add_argument('--data_root', default=None)
    p.add_argument('--warmup_ratio', type=int, default=None)
    p.add_argument('--tta_lr', type=float, default=None)
    p.add_argument('--tta_iters', type=int, default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    cfg = load_config(args.config, {
        'data_root': args.data_root,
        'warmup_ratio': args.warmup_ratio,
        'tta.lr': args.tta_lr,
        'tta.iters': args.tta_iters,
    })
    set_seed(args.seed)

    out_dir = args.out or os.path.join(
        'exps', time.strftime('%Y%m%d_%H%M%S') + f'_q{cfg["warmup_ratio"]}')
    os.makedirs(out_dir, exist_ok=True)

    videos = args.video or list_videos(cfg['data_root'], cfg['split'])
    if args.videos:
        videos = videos[:args.videos]

    header = (f'{"ISS baseline (no adaptation)" if args.baseline else "DiTTA"} | '
              f'{len(videos)} videos | warm-up {cfg["warmup_ratio"]}%\n'
              f'output: {out_dir}\n' + format_config(cfg))
    print(header)
    with open(os.path.join(out_dir, 'log.txt'), 'w') as f:
        f.write(header + '\n\n')

    model, _ = build_model(cfg['model'], args.checkpoint, device=args.device)
    target_builder = None if args.baseline else DistillTargetBuilder(
        checkpoint=args.sam2_checkpoint, num_classes=cfg['model']['num_classes'],
        device=args.device, **cfg['sam2'])
    cache = TargetCache(args.cache_dir, args.device) if args.cache_dir else None

    logger = None if args.no_tensorboard else Logger(
        out_dir, reference=args.reference, classwise=args.tb_classwise,
        classes=VSPW_CLASSES)

    metric = SegmentationMetric(cfg['model']['num_classes'])
    consistency = VideoConsistency(cfg['evaluation']['vc_windows'])
    summaries, step = [], 0

    bar = tqdm.tqdm(videos, desc='videos')
    for name in bar:
        summary = run_video(name, cfg, model, target_builder, metric, consistency,
                            cache=cache, adapt=not args.baseline,
                            logger=logger, step=step)
        step += summary['evaluated']
        summary.update(consistency.summary())
        summaries.append(summary)
        bar.set_postfix({'mIoU': f'{summary["mIoU"]:.4f}',
                         'wIoU': f'{summary["wIoU"]:.4f}'})
        with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
            f.write(json.dumps(summary) + '\n')

    result = {'videos': len(summaries), 'frames_evaluated': step,
              'mIoU': metric.miou(), 'wIoU': metric.wiou(),
              **consistency.summary()}
    print('\n' + '=' * 46)
    for key in ('videos', 'frames_evaluated'):
        print(f'{key:>16}: {result[key]}')
    for key in ('mIoU', 'wIoU', *consistency.summary()):
        print(f'{key:>16}: {100 * result[key]:.2f}')
    print('=' * 46)

    if logger is not None:
        logger.summary_table(result)
        logger.close()
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
        json.dump({'result': result, 'per_video': summaries}, f, indent=2)
    with open(os.path.join(out_dir, 'log.txt'), 'a') as f:
        f.write('\n' + json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
