"""Check this implementation against the original DiTTA code base.

Development utility used to validate the refactor.  Each check is independent;
run the ones you have artefacts for.

  1. ``pipeline``  our clip loader vs. the mmseg test pipeline
                   (needs the original CFFM repo)               -- expects exact
  2. ``model``     our ISS forward vs. the original segmentor, same process
                   (needs the original CFFM repo)               -- expects exact
  3. ``iss``       our ISS prediction vs. the cached ``res_b5`` / ``res_b5_conf``
                                                                -- expects close
  4. ``target``    our SAM2 distillation targets vs. a cached ``<exp>/result``
                                                                -- expects close

Checks 3 and 4 are tolerance checks by necessity: the cached artefacts were
produced on different hardware, and SAM2 runs in bfloat16, so its features are
reproducible only to ~0.5% relative error.  Feeding the *same* features to both
implementations yields an identical prompt sequence, which is what check 4's
tolerance is calibrated against.

    python tools/verify_against_legacy.py --videos 3 \
        --data_root data/vspw/VSPW_480p \
        --checkpoint checkpoints/ditta_segformer_b5_vspw.pth \
        --sam2_checkpoint checkpoints/sam2_hiera_large.pt \
        --legacy_root /path/to/VSS-CFFM \
        --legacy_iss_dir /path/to/vspw/res_b5 \
        --legacy_iss_conf_dir /path/to/vspw/res_b5_conf \
        --legacy_target_dir /path/to/abl_Q10_f080_b03_c08_clsall/result
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ditta.config import load_config          # noqa: E402
from ditta.data import VSPWVideo, list_videos, warmup_length  # noqa: E402

# tolerances for the two checks that cannot be exact (see module docstring)
ISS_PRED_TOL = 1e-4          # fraction of pixels whose class may differ
ISS_CONF_TOL = 5e-3          # max absolute difference of the reliability map
TARGET_LABEL_TOL = 0.05      # fraction of pixels whose distillation label may differ


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_root', required=True)
    p.add_argument('--split', default='val')
    p.add_argument('--videos', type=int, default=3, help='how many videos to check')
    p.add_argument('--config', default='configs/vspw_b5_q10.py')
    p.add_argument('--checkpoint', default='checkpoints/ditta_segformer_b5_vspw.pth')
    p.add_argument('--sam2_checkpoint', default='checkpoints/sam2_hiera_large.pt')
    p.add_argument('--seed', type=int, default=5123)
    p.add_argument('--legacy_root', default=None, help='CFFM repo, for checks 1 and 2')
    p.add_argument('--legacy_config',
                   default='local_configs/cffm/B5/cffm_tta_contra_attn_final.py')
    p.add_argument('--legacy_checkpoint',
                   default='checkpoints/segformer_b5_vspw2_baseline.pth',
                   help='relative to --legacy_root')
    p.add_argument('--legacy_iss_dir', default=None)
    p.add_argument('--legacy_iss_conf_dir', default=None)
    p.add_argument('--legacy_target_dir', default=None)
    p.add_argument('--target_from_legacy_iss', action='store_true',
                   help='feed the cached ISS predictions to the target builder, '
                        'isolating stage B from ISS differences')
    return p.parse_args()


def report(name, ok, detail=''):
    print(f'  [{"PASS" if ok else "FAIL"}] {name}{"  " + detail if detail else ""}')
    return ok


def in_legacy_repo(root, fn):
    """Run ``fn`` with the CFFM repo as cwd and on the import path."""
    cwd = os.getcwd()
    if root not in sys.path:
        sys.path.insert(0, root)
    os.chdir(root)
    try:
        return fn()
    finally:
        os.chdir(cwd)


def build_legacy_model(args):
    """The original segmentor, built and loaded exactly as run_tta_final.py did."""
    def _build():
        import mmcv
        from mmcv.runner import load_checkpoint
        from mmseg.models import build_segmentor
        cfg = mmcv.Config.fromfile(args.legacy_config)
        cfg.model.pretrained = None
        cfg.model.train_cfg = None
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
        load_checkpoint(model, args.legacy_checkpoint, map_location='cpu')
        return model.cuda().eval()
    return in_legacy_repo(args.legacy_root, _build)


# --------------------------------------------------------------------------- #


def check_pipeline(args, videos, cfg):
    """Our clip tensors vs. the mmseg test pipeline, frame by frame."""
    print('\n== 1. data pipeline (expect exact) ==')

    def _build():
        import mmcv
        from mmseg.datasets import build_dataset
        legacy_cfg = mmcv.Config.fromfile(args.legacy_config)
        legacy_cfg.data.test.test_mode = True
        return build_dataset(legacy_cfg.data.test)
    legacy = in_legacy_repo(args.legacy_root, _build)

    # the legacy dataset is a flat list of frames, in split order
    offset, pos = {}, 0
    for name in list_videos(args.data_root, args.split):
        offset[name] = pos
        pos += len(os.listdir(os.path.join(args.data_root, 'data', name, 'origin')))

    ok = True
    for name in videos:
        video = VSPWVideo(args.data_root, name, dilation=cfg['dilation'], device='cpu')
        n_warm = warmup_length(len(video), cfg['warmup_ratio'])
        probe = sorted({0, 1, n_warm - 1, len(video) - 1})
        mismatches = []
        for i in probe:
            data = in_legacy_repo(args.legacy_root, lambda: legacy[offset[name] + i])
            ref = torch.stack(data['img'][0])
            meta = data['img_metas'][0]
            meta = meta.data if hasattr(meta, 'data') else meta
            assert os.path.basename(meta['filename']) == video.frames[i]
            got = video.clip(i)
            if got.shape != ref.shape or not torch.equal(got, ref):
                mismatches.append(i)
        ok = report(f'{name}: {len(probe)} clips bit-identical',
                    not mismatches, f'differing frames: {mismatches}'
                    if mismatches else '') and ok
    return ok


def check_model(args, videos, model, cfg):
    """Our ISS forward vs. the original segmentor, on identical inputs."""
    print('\n== 2. ISS model vs. original segmentor (expect exact) ==')
    legacy = build_legacy_model(args)
    ok = True
    for name in videos:
        video = VSPWVideo(args.data_root, name, dilation=cfg['dilation'])
        n_warm = warmup_length(len(video), cfg['warmup_ratio'])
        worst = 0.0
        for i in sorted({0, 1, n_warm - 1}):
            clip = video.clip(i)
            meta = [dict(ori_shape=video.ori_shape + (3,),
                         img_shape=tuple(clip.shape[2:]) + (3,),
                         pad_shape=tuple(clip.shape[2:]) + (3,),
                         flip=False, filename=video.frame_path(i))]
            with torch.no_grad():
                _, ref, _ = legacy(return_loss=False, return_logit=True,
                                   return_feat=False,
                                   img=[[clip[k].unsqueeze(0) for k in range(len(clip))]],
                                   img_metas=[meta])
                _, got, _ = model(clip, video.ori_shape)
            worst = max(worst, float((ref - got).abs().max()))
        ok = report(f'{name}: max |prob - prob_legacy| = {worst:.3e}', worst == 0.0) and ok
    del legacy
    torch.cuda.empty_cache()
    return ok


def check_iss(args, videos, model, cfg):
    """Our ISS prediction/reliability vs. the cached res_b5 files."""
    print('\n== 3. ISS prediction vs. cached res_b5 (expect close) ==')
    from ditta.pipeline import iss_predict
    ok = True
    for name in videos:
        video = VSPWVideo(args.data_root, name, dilation=cfg['dilation'])
        n_warm = warmup_length(len(video), cfg['warmup_ratio'])
        bad, worst_conf, total = 0, 0.0, 0
        for i in range(n_warm):
            pred, conf = iss_predict(model, video, i)
            ref_p, ref_c = _load(args.legacy_iss_dir, name, video.frame_name(i)), \
                _load(args.legacy_iss_conf_dir, name, video.frame_name(i))
            bad += int((pred.cpu().numpy() != ref_p).sum())
            worst_conf = max(worst_conf, float(np.abs(conf.cpu().numpy() - ref_c).max()))
            total += ref_p.size
        ok = report(f'{name}: class differs on {100 * bad / total:.4f}% of pixels, '
                    f'max |reliability diff| = {worst_conf:.2e}',
                    bad / total < ISS_PRED_TOL and worst_conf < ISS_CONF_TOL) and ok
    return ok


def check_target(args, videos, model, cfg):
    """Our SAM2 distillation targets vs. a cached result directory."""
    print('\n== 4. SAM2 distillation targets vs. cache (expect close) ==')
    from ditta.pipeline import iss_predict
    from ditta.sam2_target import DistillTargetBuilder, target_to_label
    builder = DistillTargetBuilder(checkpoint=args.sam2_checkpoint,
                                   num_classes=cfg['model']['num_classes'],
                                   **cfg['sam2'])
    ok = True
    for name in videos:
        video = VSPWVideo(args.data_root, name, dilation=cfg['dilation'])
        n_warm = warmup_length(len(video), cfg['warmup_ratio'])
        if args.target_from_legacy_iss:
            preds = {i: torch.from_numpy(
                _load(args.legacy_iss_dir, name, video.frame_name(i))).cuda()
                for i in range(n_warm)}
            confs = {i: torch.from_numpy(
                _load(args.legacy_iss_conf_dir, name, video.frame_name(i))).cuda()
                for i in range(n_warm)}
        else:
            out = {i: iss_predict(model, video, i) for i in range(n_warm)}
            preds = {i: v[0] for i, v in out.items()}
            confs = {i: v[1] for i, v in out.items()}

        targets = builder.build(video, n_warm, preds, confs)
        bad_id = bad_label = total = 0
        for i in range(n_warm):
            ref_id = torch.from_numpy(
                _load(os.path.join(args.legacy_target_dir, 'id'), name,
                      video.frame_name(i))).cuda()
            ref_score = torch.from_numpy(
                _load(os.path.join(args.legacy_target_dir, 'score'), name,
                      video.frame_name(i))).cuda()
            got_id, got_score = targets[i]['obj_id'], targets[i]['score']
            bad_id += int((got_id != ref_id).sum())
            bad_label += int((target_to_label(ref_score, ref_id)
                              != target_to_label(got_score, got_id)).sum())
            total += ref_id.numel()
        ok = report(f'{name}: object id differs on {100 * bad_id / total:.4f}% of '
                    f'pixels, distillation label on {100 * bad_label / total:.4f}%',
                    bad_label / total < TARGET_LABEL_TOL) and ok
    return ok


def _load(directory, name, frame):
    return np.load(os.path.join(directory, f'{name}_{frame}.npy'))


def main():
    args = parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    cfg = load_config(args.config, {'data_root': args.data_root})
    videos = list_videos(args.data_root, args.split)[:args.videos]
    print(f'checking {len(videos)} video(s): {", ".join(videos)}')

    from ditta.segformer import build_model
    model, _ = build_model(cfg['model'], args.checkpoint)

    results = {}
    if args.legacy_root:
        results['pipeline'] = check_pipeline(args, videos, cfg)
        results['model'] = check_model(args, videos, model, cfg)
    if args.legacy_iss_dir and args.legacy_iss_conf_dir:
        results['iss'] = check_iss(args, videos, model, cfg)
    if args.legacy_target_dir:
        results['target'] = check_target(args, videos, model, cfg)

    print('\n== summary ==')
    for key, value in results.items():
        print(f'  {key}: {"PASS" if value else "FAIL"}')
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
