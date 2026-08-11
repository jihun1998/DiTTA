"""Distillation targets for DiTTA: spatio-temporal masks from SAM2 + ISS labels.

Stage B of the pipeline.  Given the frozen ISS predictions and reliability maps
of the warm-up frames, this module

  1. picks reliable prompt points (the highest-reliability pixel of each
     predicted class, skipping classes whose peak already falls inside a tracked
     object) and hands them to SAM2 frame by frame,
  2. propagates every prompt both forward and backward over the warm-up clip and
     merges the two passes,
  3. scores each (object, class) pair with the soft vote of Eq. (4) and lets each
     object take its winning class.

Per frame the result is an object-id map plus a per-object class-score table;
stage C turns those into the per-pixel distillation label (Eq. 5) and the object
masks used by the contrastive loss (Eq. 7).

The heavy lifting is SAM2's.  The only thing this file changes about SAM2 is
caching image features for the whole clip (``DiTTAVideoPredictor``), because the
prompt search re-runs propagation over the same frames many times.
"""

from collections import OrderedDict

import torch
from scipy.ndimage import binary_dilation

from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.utils.misc import _load_img_as_tensor

IGNORE_INDEX = 255


class DiTTAVideoPredictor(SAM2VideoPredictor):
    """SAM2 video predictor that keeps image features for every frame of a clip.

    Upstream caches only the most recently used frame, which is right for
    interactive use but makes DiTTA's prompt search re-run the image encoder tens
    of times per video.  The original DiTTA code avoided that by pre-extracting
    features for the whole dataset to disk (~240 GB); holding the clip's features
    in memory is equivalent and needs no disk at all.
    """

    @torch.inference_mode()
    def init_state_from_paths(self, frame_paths, offload_video_to_cpu=False,
                              offload_state_to_cpu=False):
        """Like ``init_state``, but for an explicit ordered list of frame files."""
        device = self.device
        images = torch.zeros(len(frame_paths), 3, self.image_size, self.image_size,
                             dtype=torch.float32)
        for n, path in enumerate(frame_paths):
            images[n], video_height, video_width = _load_img_as_tensor(
                path, self.image_size)
        img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32)[:, None, None]
        img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32)[:, None, None]
        if not offload_video_to_cpu:
            images = images.to(device)
            img_mean, img_std = img_mean.to(device), img_std.to(device)
        images -= img_mean
        images /= img_std

        return {
            'images': images,
            'num_frames': len(images),
            'offload_video_to_cpu': offload_video_to_cpu,
            'offload_state_to_cpu': offload_state_to_cpu,
            'video_height': video_height,
            'video_width': video_width,
            'device': device,
            'storage_device': torch.device('cpu') if offload_state_to_cpu else device,
            'point_inputs_per_obj': {},
            'mask_inputs_per_obj': {},
            'cached_features': {},
            'constants': {},
            'obj_id_to_idx': OrderedDict(),
            'obj_idx_to_id': OrderedDict(),
            'obj_ids': [],
            'output_dict': {'cond_frame_outputs': {}, 'non_cond_frame_outputs': {}},
            'output_dict_per_obj': {},
            'temp_output_dict_per_obj': {},
            'consolidated_frame_inds': {'cond_frame_outputs': set(),
                                        'non_cond_frame_outputs': set()},
            'tracking_has_started': False,
            'frames_already_tracked': {},
        }

    def _get_image_feature(self, inference_state, frame_idx, batch_size):
        """Identical to upstream, except that the feature cache is not evicted."""
        image, backbone_out = inference_state['cached_features'].get(
            frame_idx, (None, None))
        if backbone_out is None:
            device = inference_state['device']
            image = inference_state['images'][frame_idx].to(device).float().unsqueeze(0)
            backbone_out = self.forward_image(image)
            inference_state['cached_features'][frame_idx] = (image, backbone_out)

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            'backbone_fpn': backbone_out['backbone_fpn'].copy(),
            'vision_pos_enc': backbone_out['vision_pos_enc'].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out['backbone_fpn']):
            expanded_backbone_out['backbone_fpn'][i] = feat.expand(
                batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out['vision_pos_enc']):
            expanded_backbone_out['vision_pos_enc'][i] = pos.expand(
                batch_size, -1, -1, -1)

        features = self._prepare_backbone_features(expanded_backbone_out)
        return (expanded_image,) + features


def build_predictor(model_cfg, checkpoint, device='cuda'):
    """Build a SAM2 video predictor of the DiTTA subclass."""
    return build_sam2_video_predictor(
        model_cfg, ckpt_path=checkpoint, device=device,
        hydra_overrides_extra=['++model._target_=ditta.sam2_target.DiTTAVideoPredictor'])


class DistillTargetBuilder:
    """Builds DiTTA's distillation targets, one video at a time."""

    def __init__(self, checkpoint, model_cfg='sam2_hiera_l.yaml', max_track=100,
                 frame0_conf_thre=0.8, frameN_conf_thre=0.8, b=0.3, c=0.8,
                 num_classes=124, device='cuda'):
        self.predictor = build_predictor(model_cfg, checkpoint, device)
        self.max_track = max_track
        self.frame0_conf_thre = frame0_conf_thre
        self.frameN_conf_thre = frameN_conf_thre
        self.b = b
        self.c = c
        self.num_classes = num_classes
        self.device = device

    def build(self, video, num_warmup, preds, reliabilities):
        """Return ``{frame: {'obj_id': [H,W] int64, 'score': [O+1,C] float32}}``.

        ``preds`` / ``reliabilities`` map a warm-up frame index to the tensors
        produced by :func:`ditta.pipeline.iss_predict`.
        """
        predictor = self.predictor
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            state = predictor.init_state_from_paths(
                [video.frame_path(i) for i in range(num_warmup)])
            try:
                selected, forward_masks, pixels_per_class = self._forward_pass(
                    predictor, state, num_warmup, preds, reliabilities)
                if not selected:
                    return {}
                backward_masks = self._backward_pass(predictor, state, selected)
                merged = self._merge(forward_masks, backward_masks, selected)
                scores = self._vote(merged, preds, reliabilities, pixels_per_class)
                targets = self._render(merged, scores, preds)
            finally:
                predictor.reset_state(state)
                state['cached_features'].clear()
                del state
                torch.cuda.empty_cache()
        # clone out of inference mode so the targets can be used by autograd
        return {i: {k: v.clone() for k, v in t.items()} for i, t in targets.items()}

    # ------------------------------------------------------------------ #

    def _forward_pass(self, predictor, state, num_warmup, preds, reliabilities):
        """Search for prompts frame by frame, propagating one frame at a time."""
        global_obj = 1
        tracked_per_class = [0] * self.num_classes
        pixels_per_class = [0] * self.num_classes
        selected, forward_masks = {}, {}
        obj_ids, masks = [], None

        for frame_idx in range(num_warmup):
            pred, conf = preds[frame_idx], reliabilities[frame_idx]

            # Region already explained by a tracked object (dilated by 5 px), so a
            # prompt is only spent where the current tracks say nothing.
            occupied = None
            if frame_idx != 0:
                occupied = torch.zeros_like(pred, dtype=torch.bool)
                for i in range(len(obj_ids)):
                    grown = binary_dilation((masks[i, 0] > 0.0).cpu().numpy(),
                                            iterations=5)
                    occupied |= torch.from_numpy(grown).to(occupied.device)

            conf_thre = self.frame0_conf_thre if frame_idx == 0 else self.frameN_conf_thre
            classes = sorted(torch.unique(pred).tolist(),
                             key=lambda c: torch.max(conf[pred == c]), reverse=True)
            reset_done = False
            prev_obj_ids, prev_masks = [], None

            for cat in classes:
                pixels_per_class[cat] += int(torch.sum(pred == cat))
                if sum(tracked_per_class) >= self.max_track:
                    continue
                peak = torch.max(conf[pred == cat])
                if peak < conf_thre:
                    continue
                y, x = ((pred == cat) & (conf == peak)).nonzero()[0].tolist()
                if occupied is not None and occupied[y, x]:
                    continue

                # SAM2 wants every prompt of a frame registered before the frame
                # is tracked, so the first new prompt on a frame resets the state
                # and re-adds the objects tracked so far as mask inputs.
                if frame_idx != 0 and not reset_done:
                    reset_done = True
                    prev_obj_ids = list(obj_ids)
                    prev_masks = masks.clone().detach()
                    predictor.reset_state(state)

                point = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
                labels = torch.tensor([1], dtype=torch.int32, device=self.device)
                _, obj_ids, masks = predictor.add_new_points_or_box(
                    state, frame_idx, global_obj, points=point, labels=labels)

                tracked_per_class[cat] += 1
                selected.setdefault(frame_idx, []).append(
                    {'obj_id': global_obj, 'point': point})
                global_obj += 1

            if reset_done:
                for i, obj_id in enumerate(prev_obj_ids):
                    _, obj_ids, masks = predictor.add_new_mask(
                        state, frame_idx, obj_id, prev_masks[i, 0] > 0.0)

            forward_masks[frame_idx] = {'obj_ids': list(obj_ids),
                                        'masks': (masks > 0).clone()}

            if frame_idx < num_warmup - 1:
                gen = predictor.propagate_in_video(
                    state, start_frame_idx=frame_idx, max_frame_num_to_track=1)
                next(gen)                       # this frame
                _, obj_ids, masks = next(gen)   # the next one
                gen.close()

        return selected, forward_masks, pixels_per_class

    def _backward_pass(self, predictor, state, selected):
        """Replay exactly the same prompts, from the last prompted frame back."""
        predictor.reset_state(state)
        backward_masks = {}
        obj_ids, masks = [], None
        last_ann = max(selected)

        for frame_idx in range(last_ann, -1, -1):
            reset_done = False
            prev_obj_ids, prev_masks = [], None
            if frame_idx in selected:
                if frame_idx != last_ann:
                    reset_done = True
                    prev_obj_ids = list(obj_ids)
                    prev_masks = masks.clone().detach()
                    predictor.reset_state(state)
                labels = torch.tensor([1], dtype=torch.int32, device=self.device)
                for ann in selected[frame_idx]:
                    _, obj_ids, masks = predictor.add_new_points_or_box(
                        state, frame_idx, ann['obj_id'], points=ann['point'],
                        labels=labels)
            if reset_done:
                for i, obj_id in enumerate(prev_obj_ids):
                    _, obj_ids, masks = predictor.add_new_mask(
                        state, frame_idx, obj_id, prev_masks[i, 0] > 0.0)

            backward_masks[frame_idx] = {'obj_ids': list(obj_ids),
                                         'masks': (masks > 0).clone()}

            if frame_idx > 0:
                gen = predictor.propagate_in_video(
                    state, start_frame_idx=frame_idx, max_frame_num_to_track=1,
                    reverse=True)
                next(gen)
                _, obj_ids, masks = next(gen)
                gen.close()

        return backward_masks

    @staticmethod
    def _merge(forward_masks, backward_masks, selected):
        """Union of the forward and backward tracks, per frame.

        A frame that carries prompts holds those objects in both passes, so the
        backward copies are dropped before concatenating.
        """
        merged = {}
        last_ann = max(selected)
        for frame_idx in sorted(forward_masks):
            fwd = forward_masks[frame_idx]
            if frame_idx > last_ann:
                merged[frame_idx] = {'obj_ids': list(fwd['obj_ids']),
                                     'masks': fwd['masks']}
                continue
            bwd = backward_masks[frame_idx]
            prompted = {ann['obj_id'] for ann in selected.get(frame_idx, [])}
            keep = [i for i, oid in enumerate(bwd['obj_ids']) if oid not in prompted]
            merged[frame_idx] = {
                'obj_ids': list(fwd['obj_ids']) + [bwd['obj_ids'][i] for i in keep],
                'masks': torch.cat((fwd['masks'], bwd['masks'][keep]), dim=0),
            }
        return merged

    def _vote(self, merged, preds, reliabilities, pixels_per_class):
        """Class score per object: mean reliability x area weight x frequency weight."""
        pixels, confs = {}, {}   # obj -> class -> pixel count / reliability values
        for frame_idx, entry in merged.items():
            pred, conf = preds[frame_idx], reliabilities[frame_idx]
            for obj_idx, obj_id in enumerate(entry['obj_ids']):
                mask = entry['masks'][obj_idx, 0]
                pixels.setdefault(obj_id, {})
                confs.setdefault(obj_id, {})
                for cat in torch.unique(pred[mask]).tolist():
                    inside = mask & (pred == cat)
                    pixels[obj_id][cat] = pixels[obj_id].get(cat, 0) + int(inside.sum())
                    values = conf[inside]
                    confs[obj_id][cat] = values if cat not in confs[obj_id] else \
                        torch.cat((confs[obj_id][cat], values), dim=0)

        video_pixels = sum(pixels_per_class)
        scores = {}
        for obj_id, per_class in confs.items():
            obj_pixels = sum(pixels[obj_id].values())
            scores[obj_id] = {}
            for cat, values in per_class.items():
                mean_conf = torch.sum(values) / pixels[obj_id][cat]
                area_w = (pixels[obj_id][cat] / obj_pixels) ** self.b
                freq_w = (1 - pixels_per_class[cat] / video_pixels) ** self.c
                scores[obj_id][cat] = mean_conf * area_w * freq_w
        return scores

    def _render(self, merged, scores, preds):
        """Paint object ids per frame, lowest-scoring object first so it loses."""
        targets = {}
        for frame_idx, entry in merged.items():
            obj_ids, masks = entry['obj_ids'], entry['masks']
            ref = preds[frame_idx]
            obj_id_map = torch.zeros_like(ref, dtype=torch.int64)
            score_table = torch.zeros((max(obj_ids) + 1, self.num_classes),
                                      dtype=torch.float32, device=ref.device)
            order = sorted(obj_ids, key=lambda i: max(scores[i].values(), default=0.0))
            for obj_id in order:
                obj_id_map[masks[obj_ids.index(obj_id), 0]] = int(obj_id)
                for cat, value in scores[obj_id].items():
                    score_table[obj_id, cat] = value
            targets[frame_idx] = {'obj_id': obj_id_map, 'score': score_table}
        return targets


def target_to_label(score, obj_id):
    """Per-pixel distillation label implied by a target.

    Each object takes its highest-scoring class.  Object 0 (no mask) and objects
    whose score row is all zero are ignored.
    """
    cls_of_obj = score.argmax(1)
    cls_of_obj[0] = IGNORE_INDEX
    cls_of_obj[score.max(1).values == 0] = IGNORE_INDEX
    return cls_of_obj[obj_id]
