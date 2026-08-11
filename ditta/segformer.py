"""ISS model used by DiTTA: SegFormer (MiT-B5) + the DiTTA temporal add-on head.

This is a self-contained extraction of the two classes DiTTA actually uses from
the CFFM code base (`EncoderDecoder_clips_final` and
`CFFMHead_clips_resize1_8_final`).  The forward math is unchanged; only the
mmcv/mmseg plumbing is replaced by the small local equivalents below, so the
model needs nothing beyond torch + timm.

Differences from the original CFFM classes, all of them numerically inert:

* `conv_seg` (created by the mmseg decode-head base class but never called by
  this head) and `value_layer` (created but never used in the forward) are not
  instantiated.
* the local-window attention mask is cached per (h, w) instead of being rebuilt
  on every forward, and is applied with `masked_fill` instead of an additive
  -inf tensor.  Same result, ~4x less memory, no per-frame allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mit_backbone import mit_b5

# --------------------------------------------------------------------------- #
# local replacements for mmseg.ops.resize / mmcv.cnn.ConvModule
# --------------------------------------------------------------------------- #


def resize(input, size=None, scale_factor=None, mode='bilinear', align_corners=None):
    """mmseg.ops.resize without the shape-warning bookkeeping."""
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class ConvModule(nn.Module):
    """conv + norm + activation, with mmcv's ``conv``/``bn`` submodule names.

    Only the configuration CFFM uses is supported (1x1 conv, BN, ReLU).  The
    submodule names are kept so checkpoint keys line up with mmcv's ConvModule.
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, with_norm=True,
                 with_activation=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=not with_norm)
        self.bn = nn.BatchNorm2d(out_channels) if with_norm else None
        self.activate = nn.ReLU(inplace=True) if with_activation else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activate is not None:
            x = self.activate(x)
        return x


class MLP(nn.Module):
    """Linear embedding of a feature map, as in SegFormer's decoder."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def local_window_mask(h, w, window_size, device):
    """Boolean [h*w, h*w] mask, True where attention is allowed.

    Equivalent to the original `create_local_mask`, which returned the same
    neighbourhood as an additive 0 / -inf float tensor.
    """
    total = h * w
    mask = torch.zeros((total, total), dtype=torch.bool, device=device)
    y = torch.arange(h, device=device).repeat_interleave(w)
    x = torch.arange(w, device=device).repeat(h)
    center = y * w + x
    for dy in range(-window_size, window_size + 1):
        for dx in range(-window_size, window_size + 1):
            ny, nx = y + dy, x + dx
            valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
            mask[center[valid], (ny * w + nx)[valid]] = True
    return mask


# --------------------------------------------------------------------------- #
# decode head
# --------------------------------------------------------------------------- #


class DiTTAHead(nn.Module):
    """SegFormer MLP decoder + DiTTA's temporal attention add-on.

    Returns ``(feat, logits)``:

    * ``logits`` is ``[1, C, h, w]`` for a single-frame clip, and
      ``[2, C, h, w]`` for a multi-frame clip, where row 0 is the plain
      per-frame prediction for the current frame and row 1 is the add-on
      (temporally fused) prediction.
    * ``feat`` is the projected feature map used by the contrastive loss, or
      ``None`` when ``return_feat`` is False.
    """

    def __init__(self,
                 in_channels=(64, 128, 320, 512),
                 embed_dim=256,
                 num_classes=124,
                 dropout_ratio=0.1,
                 window=5,
                 align_corners=False):
        super().__init__()
        c1, c2, c3, c4 = in_channels
        self.num_classes = num_classes
        self.embedding_dim = embed_dim
        self.align_corners = align_corners
        self.window = window
        self._mask_cache = {}

        self.linear_c4 = MLP(input_dim=c4, embed_dim=embed_dim)
        self.linear_c3 = MLP(input_dim=c3, embed_dim=embed_dim)
        self.linear_c2 = MLP(input_dim=c2, embed_dim=embed_dim)
        self.linear_c1 = MLP(input_dim=c1, embed_dim=embed_dim)

        self.linear_fuse = ConvModule(embed_dim * 4, embed_dim, kernel_size=1)
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None

        # --- DiTTA add-on: temporal fusion by local cross-frame attention ---
        self.query_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        self.key_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim))
        # --- projection head for the mask-based contrastive loss ---
        self.proj_feat = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1), nn.ReLU(),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1))

    def _mask(self, h, w, device):
        key = (h, w, self.window, str(device))
        if key not in self._mask_cache:
            self._mask_cache[key] = local_window_mask(h, w, self.window, device)
        return self._mask_cache[key]

    def forward(self, inputs, num_clips, return_feat=False):
        c1, c2, c3, c4 = inputs
        n = c4.shape[0]

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        _, _, h, w = _c.shape

        x = self.dropout(_c) if self.dropout is not None else _c
        x = self.linear_pred(x)
        x = x.reshape(1, num_clips, -1, h, w)

        feat = self.proj_feat(_c) if return_feat else None

        if num_clips == 1:
            return feat, x[:, -1]

        # Add-on: query from the current frame, key/value from the first frame
        # of the clip, restricted to a local spatial window.
        query = self.query_layer(_c[-1].view(self.embedding_dim, -1).transpose(1, 0))
        key = self.key_layer(_c[0].view(self.embedding_dim, -1).transpose(1, 0))
        value = x[0, 0].view(self.num_classes, -1).transpose(1, 0)

        attn = torch.matmul(query, key.transpose(1, 0))
        attn = attn.masked_fill(~self._mask(h, w, attn.device), float('-inf'))
        attention_feature = torch.matmul(F.softmax(attn, dim=-1), value)
        attention_feature = attention_feature.transpose(1, 0).view(
            self.num_classes, h, w).unsqueeze(0)

        return feat, torch.cat((x[0, -1].unsqueeze(0), attention_feature), dim=0)


# --------------------------------------------------------------------------- #
# segmentor
# --------------------------------------------------------------------------- #


class DiTTASegmentor(nn.Module):
    """SegFormer-B5 ISS model with the DiTTA add-on head."""

    def __init__(self, num_classes=124, embed_dim=256, dropout_ratio=0.1,
                 window=5, align_corners=False):
        super().__init__()
        self.backbone = mit_b5()
        self.decode_head = DiTTAHead(num_classes=num_classes, embed_dim=embed_dim,
                                     dropout_ratio=dropout_ratio, window=window,
                                     align_corners=align_corners)
        self.align_corners = align_corners
        self.num_classes = num_classes

    def forward(self, clip, ori_shape, return_feat=False):
        """Run the model on one clip.

        Args:
            clip: ``[num_clips, 3, H, W]`` normalised input; the last frame is
                the current frame.
            ori_shape: ``(H_ori, W_ori)`` of the source image.
            return_feat: also return the projected feature map.

        Returns:
            ``(pred, prob, feat)`` where ``prob`` is softmax over classes at the
            original resolution and ``pred = prob.argmax(1)``.  Both have a
            leading dimension of 1 (single frame) or 2 (plain, add-on).
        """
        num_clips = clip.shape[0]
        x = self.backbone(clip)
        feat, out = self.decode_head(x, num_clips=num_clips, return_feat=return_feat)
        # two-step resize, matching CFFM's encode_decode + whole_inference
        out = resize(out, size=clip.shape[2:], mode='bilinear',
                     align_corners=self.align_corners)
        out = resize(out, size=tuple(ori_shape), mode='bilinear',
                     align_corners=self.align_corners)
        prob = F.softmax(out, dim=1)
        return prob.argmax(dim=1), prob, feat


def build_model(cfg, checkpoint, device='cuda'):
    """Build the ISS model and load a DiTTA checkpoint."""
    model = DiTTASegmentor(num_classes=cfg['num_classes'],
                           embed_dim=cfg['embed_dim'],
                           dropout_ratio=cfg['dropout_ratio'],
                           window=cfg['window'])
    state = torch.load(checkpoint, map_location='cpu')
    meta = state.get('meta', {})
    state = state.get('state_dict', state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f'checkpoint {checkpoint} is missing {len(missing)} parameters, '
            f'e.g. {missing[:4]}.\nRun tools/prepare_checkpoint.py to convert a '
            'CFFM/SegFormer checkpoint into a DiTTA checkpoint.')
    if unexpected:
        print(f'[model] ignoring {len(unexpected)} unexpected keys, '
              f'e.g. {unexpected[:4]}')
    model.to(device).eval()
    return model, meta
