"""Test-time adaptation of the ISS model on a video's warm-up frames.

Stage C.  For every warm-up frame the loss is

    L = L_distill + L_contrastive                                   (Eq. 8)

    L_distill      cross entropy of both output branches (plain and add-on)
                   against the distillation label, on the pixels SAM2 covers,
                   with the ISS prediction filling in everywhere else  (Eq. 5)
    L_contrastive  per-object prototypes are read off a momentum copy of the
                   model and the live features are pulled towards the prototype
                   of the object they belong to                     (Eq. 6, 7)

Only the decoder is adapted; the backbone stays frozen.  A fresh copy of the
pre-trained model is adapted for each video, so videos never influence one
another.
"""

import copy

import torch
import torch.nn.functional as F

from .sam2_target import IGNORE_INDEX, target_to_label

PARAM_GROUPS = {
    'decoder': lambda head: head.parameters(),
    'cls_head': lambda head: head.linear_pred.parameters(),
    'addon': lambda head: list(head.query_layer.parameters())
                          + list(head.key_layer.parameters()),
}


def softmax_entropy(scores):
    return -(scores.softmax(1) * scores.log_softmax(1)).sum(1)


def cross_entropy(scores, target):
    """Cross entropy averaged over *all* pixels, ignored ones counting as zero.

    This is mmseg's ``CrossEntropyLoss`` reduction, which the original DiTTA
    implementation used.  It differs from ``F.cross_entropy(..., reduction=
    'mean')``, which would divide by the number of non-ignored pixels only --
    a large difference for the contrastive term, where most pixels are ignored.
    """
    return F.cross_entropy(scores, target, reduction='none',
                           ignore_index=IGNORE_INDEX).mean()


class VideoTTA:
    """Holds the adapted model, its momentum copy and the optimiser for a video."""

    def __init__(self, base_model, lr=1e-3, opt_params='decoder', momentum=0.99,
                 temperature=1.0, soft_contra=True):
        self.model = copy.deepcopy(base_model)
        self.momentum_model = copy.deepcopy(base_model)
        self.momentum = momentum
        self.temperature = temperature
        self.soft_contra = soft_contra
        self.prototypes = {}

        for model in (self.model, self.momentum_model):
            for name, param in model.named_parameters():
                param.requires_grad = 'decode_head' in name

        if opt_params not in PARAM_GROUPS:
            raise ValueError(f'unknown opt_params={opt_params!r}, '
                             f'expected one of {sorted(PARAM_GROUPS)}')
        params = list(PARAM_GROUPS[opt_params](self.model.decode_head))
        self.optimizer = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999))

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _update_momentum(self):
        for ema, live in zip(self.momentum_model.parameters(), self.model.parameters()):
            ema.data.mul_(self.momentum).add_(live.data, alpha=1 - self.momentum)

    def step(self, clip, ori_shape, iss_pred, target):
        """One optimisation step on a single warm-up frame.

        Args:
            clip: model input for this frame, ``[num_clips, 3, H, W]``.
            ori_shape: original image size.
            iss_pred: frozen ISS prediction, ``[H, W]``; it fills the pixels no
                SAM2 mask covers.
            target: ``{'obj_id', 'score'}`` for this frame, from stage B.

        Returns a dict of scalar losses.
        """
        obj_id, score = target['obj_id'], target['score']
        label = target_to_label(score, obj_id)
        distill_label = iss_pred.clone()
        keep = label != IGNORE_INDEX
        distill_label[keep] = label[keep]
        distill_label = distill_label.unsqueeze(0)

        self.model.train()
        _, prob, feat = self.model(clip, ori_shape, return_feat=True)
        with torch.no_grad():
            self.momentum_model.train()
            _, prob_mo, feat_mo = self.momentum_model(clip, ori_shape, return_feat=True)

        # Note: the model outputs class *probabilities*, and -- as in the
        # original implementation -- they are fed to cross entropy directly, so
        # a log_softmax is applied on top of the softmax.  Kept as is; changing
        # it changes the reported numbers.
        losses = {}
        losses['distill'] = cross_entropy(prob[:1], distill_label)
        if prob.shape[0] == 2:                       # add-on branch
            losses['distill_addon'] = cross_entropy(prob[1:], distill_label)
        contra = self._contrastive(obj_id, feat, feat_mo, prob_mo)
        if contra is not None:
            losses['contra'] = contra

        total = sum(losses.values())
        self.optimizer.zero_grad()
        total.backward()
        self.optimizer.step()
        self._update_momentum()

        losses['total'] = total
        return {k: float(v.detach()) for k, v in losses.items()}

    # ------------------------------------------------------------------ #

    def _contrastive(self, obj_id, feat, feat_mo, prob_mo):
        """Mask-based contrastive alignment against momentum prototypes."""
        objects = [o for o in obj_id.unique().tolist() if o != 0]
        if not objects:
            return None
        h, w = obj_id.shape

        with torch.no_grad():
            feat_mo = F.normalize(feat_mo, dim=1)
            feat_mo = F.interpolate(feat_mo, (h, w), mode='bilinear',
                                    align_corners=False)[-1]
            feat_mo = F.normalize(feat_mo, dim=0)

            if self.soft_contra:
                # as in the original: entropy of the already-softmaxed output
                reliability = softmax_entropy(prob_mo[:1])
                reliability = 1 - (reliability - reliability.min()) / \
                    (reliability.max() - reliability.min())
                reliability = F.interpolate(reliability.unsqueeze(0), (h, w),
                                            mode='bilinear', align_corners=False)[0]

            prototypes, assignment = [], obj_id.new_full((1, h, w), IGNORE_INDEX)
            for slot, oid in enumerate(objects):
                region = obj_id == oid
                inside = feat_mo[:, region].detach()
                if self.soft_contra:
                    proto = (inside * reliability[:, region]).mean(dim=1)
                else:
                    proto = inside.mean(dim=1)
                proto = F.normalize(proto, dim=0)
                if oid in self.prototypes:            # running average over the video
                    proto = F.normalize((proto + self.prototypes[oid]) / 2, dim=0)
                self.prototypes[oid] = proto
                prototypes.append(proto)
                assignment[0, region] = slot
            prototypes = torch.stack(prototypes).permute(1, 0)   # [D, O]

        dim = feat.shape[1]
        feat = F.normalize(feat, dim=1)
        feat = F.interpolate(feat, (h, w), mode='bilinear', align_corners=False)[-1]
        feat = F.normalize(feat, dim=0)

        similarity = torch.mm(feat.view(dim, h * w).permute(1, 0), prototypes)
        similarity = similarity.permute(1, 0).view(-1, h, w).unsqueeze(0)
        return cross_entropy(similarity / self.temperature, assignment)


def adapt_video(base_model, video, warmup_frames, iss_preds, targets, cfg,
                progress=None, logger=None):
    """Adapt a copy of ``base_model`` on the warm-up frames and return it."""
    tta = VideoTTA(base_model, lr=cfg['lr'], opt_params=cfg['opt_params'],
                   momentum=cfg['momentum'], temperature=cfg['temperature'],
                   soft_contra=cfg['soft_contra'])
    last = {}
    for _ in range(cfg['iters']):
        for i in warmup_frames:
            if i not in targets:
                continue
            last = tta.step(video.clip(i), video.ori_shape, iss_preds[i], targets[i])
            if logger is not None:
                logger.losses(last)
            if progress is not None:
                progress.update(1)
                progress.set_postfix({k: f'{v:.3f}' for k, v in last.items()})
    return tta.model, last
