# Acknowledgements and third-party licences

DiTTA builds on the following work. Please respect the licences below when using
this repository.

## SegFormer — vendored

[`ditta/mit_backbone.py`](ditta/mit_backbone.py) is the MiT-B5 backbone from
[SegFormer](https://github.com/NVlabs/SegFormer) (NVlabs), copied from
`mmseg/models/backbones/mix_transformer.py`. Only the mmcv/mmseg glue was
removed (registry decorators, the mmcv checkpoint loader, and the unused
`mit_b0`..`mit_b4` variants); the module definitions are untouched.

> Copyright (c) 2021, NVIDIA Corporation. Licensed under the
> [NVIDIA Source Code License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE)
> — **non-commercial use only**. This licence governs that file and any
> derivative of it, including the released checkpoints.

## CFFM / mmsegmentation — derived

[`ditta/segformer.py`](ditta/segformer.py) (decode head and segmentor) and
[`ditta/data.py`](ditta/data.py) (the VSPW clip pipeline) are re-implementations
of the corresponding classes in
[VSS-CFFM](https://github.com/GuoleiSun/VSS-CFFM), which is itself built on
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation) (Apache-2.0).
The forward computation is preserved exactly; the mmcv dependency is not.

## SAM2 — used as a dependency

[SAM2](https://github.com/facebookresearch/segment-anything-2) (Meta,
Apache-2.0) provides the temporal segmentation knowledge DiTTA distils. No SAM2
source file is modified: [`ditta/sam2_target.py`](ditta/sam2_target.py)
subclasses `SAM2VideoPredictor` to initialise state from an explicit list of
frame paths and to keep image features cached for a whole clip.

## VSPW

Experiments use the [VSPW](https://www.vspwdataset.com/) benchmark; see the
dataset's own terms for its licence.

## This repository

Original DiTTA code (everything not listed above) is released for research use.
Because it links against the SegFormer backbone above, the repository as a whole
is subject to the NVIDIA Source Code License's non-commercial restriction.
