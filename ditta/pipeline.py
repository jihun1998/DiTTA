"""The DiTTA pipeline for one video, from ISS inference to evaluation.

Four stages run back to back per video, entirely in memory:

  A. ISS inference on the warm-up frames -> label + reliability map
  B. SAM2 distillation targets from those predictions   (ditta/sam2_target.py)
  C. test-time adaptation on the warm-up frames         (ditta/tta.py)
  D. inference + evaluation on the remaining frames, model frozen

In the original code base these were four separate programs communicating
through ~250 GB of cached .npy/.pth files.  Nothing here touches the disk unless
a cache directory is requested.
"""

import torch
import tqdm

from .data import VSPWVideo, warmup_length
from .metrics import SegmentationMetric, prepare_gt


@torch.no_grad()
def iss_predict(model, video, i):
    """Frozen ISS prediction for frame ``i``.

    Returns ``(pred, reliability)`` at the original image resolution: the
    per-pixel class, and the reliability map ``R = 1 - E / max(E)`` of Eq. (2)
    where ``E`` is the pixel-wise entropy of the ISS class distribution.
    """
    model.eval()
    _, prob, _ = model(video.clip(i), video.ori_shape)
    prob = prob[0]
    entropy = -(prob * prob.log()).sum(0)
    return prob.argmax(0), 1 - entropy / entropy.max()


@torch.no_grad()
def fuse_and_predict(model, video, i, tau):
    """Adapted-model inference for one evaluation frame.

    Where the plain branch is unreliable (``R < tau``) and the add-on branch is
    more reliable, the two are blended by their reliabilities (Eq. 3).
    """
    model.eval()
    _, prob, _ = model(video.clip(i), video.ori_shape)
    if prob.shape[0] == 2:
        entropy = -(prob * prob.log()).sum(1)
        rel = 1 - entropy / entropy.amax(dim=(1, 2), keepdim=True)
        weak = rel[0] < tau
        weight = rel[0][weak] / (rel[0][weak] + rel[1][weak])
        prob[0][:, weak] = (weight.unsqueeze(0) * prob[0][:, weak]
                            + (1 - weight).unsqueeze(0) * prob[1][:, weak])
    return prob[0].argmax(0)


def run_video(name, cfg, model, target_builder, metric, consistency, cache=None,
              adapt=True, logger=None, step=0, verbose=True):
    """Run all four stages on one video and fold the result into the metrics.

    With ``adapt=False`` stages B and C are skipped and the frozen ISS model is
    evaluated as is, which reproduces the ISS baseline row of the paper.
    ``step`` is the number of frames evaluated so far and serves as the
    TensorBoard x-axis.

    Returns a per-video summary dict.
    """
    video = VSPWVideo(cfg['data_root'], name, dilation=cfg['dilation'])
    num_warmup = warmup_length(len(video), cfg['warmup_ratio'])
    warmup_frames = list(range(num_warmup))
    eval_frames = list(range(num_warmup, len(video)))
    steps = num_warmup * (1 + cfg['tta']['iters']) if adapt else 0
    bar = tqdm.tqdm(total=steps + len(eval_frames), desc=name, leave=False,
                    disable=not verbose)

    adapted, last_losses = model, {}
    if adapt:
        # --- A. frozen ISS predictions on the warm-up frames ----------------
        preds, reliabilities = {}, {}
        for i in warmup_frames:
            preds[i], reliabilities[i] = iss_predict(model, video, i)
            bar.update(1)

        # --- B. SAM2 distillation targets -----------------------------------
        bar.set_description(f'{name} [sam2]')
        targets = cache.load_targets(name, warmup_frames) if cache else None
        if targets is None:
            targets = target_builder.build(video, num_warmup, preds, reliabilities)
            if cache:
                cache.save_targets(name, targets)

        # --- C. test-time adaptation -----------------------------------------
        from .tta import adapt_video   # local import keeps `ditta.pipeline` light
        bar.set_description(f'{name} [tta]')
        adapted, last_losses = adapt_video(model, video, warmup_frames, preds,
                                           targets, cfg['tta'], progress=bar,
                                           logger=logger)
        del preds, reliabilities, targets
        torch.cuda.empty_cache()

    # --- D. frozen inference on the remaining frames ------------------------
    bar.set_description(f'{name} [eval]')
    # without adaptation the add-on is untrained, so its branch is not fused in
    tau = cfg['evaluation']['tau'] if adapt else 0.0
    instant = SegmentationMetric(cfg['model']['num_classes'])
    video_preds, video_labels = [], []
    for i in eval_frames:
        pred = fuse_and_predict(adapted, video, i, tau)
        label = prepare_gt(video.gt(i)).to(pred.device)
        pred_np, label_np = pred.cpu().numpy(), label.cpu().numpy()
        metric.add(pred_np, label_np)
        if logger is not None:
            instant.reset()
            instant.add(pred_np, label_np)
            logger.frame_metrics(metric, instant, step)
        step += 1
        video_preds.append(pred)
        video_labels.append(label)
        bar.update(1)
    consistency.add_video(video_preds, video_labels)
    if logger is not None:
        logger.video_metrics(consistency, step)
    bar.close()

    if adapted is not model:
        del adapted
    del video_preds, video_labels
    video.drop_cache()
    torch.cuda.empty_cache()

    return {'video': name, 'frames': len(video), 'warmup': num_warmup,
            'evaluated': len(eval_frames), 'loss': last_losses,
            'mIoU': metric.miou(), 'wIoU': metric.wiou()}
