# DiTTA — Bootstrapping a Video Semantic Segmentation Model via Distillation-assisted Test-Time Adaptation (CVPR 2026)

[**Jihun Kim**](https://jihun1998.github.io/)<sup>1\*</sup>,
[**Hoyong Kwon**](https://kwonhoyong3.github.io/)<sup>1\*</sup>,
[**Hyeokjun Kweon**](https://fovlab.cau.ac.kr/)<sup>2\*</sup>,
[**Kuk-Jin Yoon**](https://vi.kaist.ac.kr/)<sup>1</sup>

<sup>1</sup>KAIST &nbsp;&nbsp; <sup>2</sup>Chung-Ang University &nbsp;&nbsp; (\*: equal contribution)

<p>
  <a href="https://openaccess.thecvf.com/content/CVPR2026/papers/Kim_Bootstrapping_Video_Semantic_Segmentation_Model_via_Distillation-assisted_Test-Time_Adaptation_CVPR_2026_paper.pdf"><img alt="Paper" src="https://img.shields.io/badge/Paper-CVPR%202026-4c6ef5?style=for-the-badge"></a>
  <a href="https://openaccess.thecvf.com/content/CVPR2026/supplemental/Kim_Bootstrapping_Video_Semantic_CVPR_2026_supplemental.pdf"><img alt="Supplementary" src="https://img.shields.io/badge/Supp-PDF-2f9e44?style=for-the-badge"></a>
  <a href="https://arxiv.org/abs/2604.10950"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2604.10950-b31b1b?style=for-the-badge"></a>
</p>

Official implementation of *"Bootstrapping Video Semantic Segmentation Model via
Distillation-assisted Test-Time Adaptation"* (CVPR 2026).

DiTTA turns a frame-wise **Image** Semantic Segmentation (ISS) model into a
temporally-aware **Video** Semantic Segmentation (VSS) model at test time, using
no video annotations. For each test video it adapts on a short warm-up prefix
(10% of the frames by default) by distilling SAM2's temporal segmentation
behaviour into the ISS model, then freezes the model and runs on the rest of the
video — SAM2 is never invoked at inference time.

One command runs the whole method:

```bash
python run_ditta.py \
    --checkpoint checkpoints/ditta_segformer_b5_vspw.pth \
    --sam2_checkpoint checkpoints/sam2_hiera_large.pt
```

For every video this runs all four stages back to back, in memory:

| stage | what happens | code |
|---|---|---|
| A | frozen ISS inference on the warm-up frames → labels + reliability maps | [`ditta/pipeline.py`](ditta/pipeline.py) |
| B | SAM2 prompting, bidirectional propagation and class voting → distillation targets | [`ditta/sam2_target.py`](ditta/sam2_target.py) |
| C | test-time adaptation of the decoder (distillation + contrastive losses) | [`ditta/tta.py`](ditta/tta.py) |
| D | frozen inference and evaluation on the remaining 90% of frames | [`ditta/pipeline.py`](ditta/pipeline.py) |

---

## 1. Environment

Python 3.10 and CUDA 12.1, set up with conda. A single GPU is enough; adaptation
peaks at about 20 GB of memory.

```bash
conda create -n ditta python=3.10.14 -y
conda activate ditta

conda install pytorch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
              pytorch-cuda=12.1 "mkl=2023.1.0" -c pytorch -c nvidia -y

git clone https://github.com/jihun1998/DiTTA.git
cd DiTTA
pip install -r requirements.txt
```

> The `mkl` pin matters: with mkl 2024 or newer, this build of PyTorch fails to
> import with `undefined symbol: iJIT_NotifyEvent`.

### SAM2

DiTTA calls SAM2 through its public API and subclasses its video predictor; no
SAM2 source file is modified. Install it from the official repository, pinned to
the commit this code was developed against (later revisions renamed the package
layout and the config files):

```bash
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
git checkout 7e1596c0b6462eb1d1ba7e1492430fed95023598
pip install -e .
cd checkpoints && ./download_ckpts.sh && cd ../..
```

Then point DiTTA at the SAM2 weights, e.g.

```bash
ln -s $(pwd)/segment-anything-2/checkpoints/sam2_hiera_large.pt \
      checkpoints/sam2_hiera_large.pt
```

> The original CFFM / mmsegmentation stack is **not** required. The ISS model
> lives in [`ditta/segformer.py`](ditta/segformer.py) and
> [`ditta/mit_backbone.py`](ditta/mit_backbone.py) and depends only on torch and
> timm — no mmcv, no mmsegmentation, no CUDA extensions to compile. You only
> need the [CFFM repo](https://github.com/GuoleiSun/VSS-CFFM) if you want to run
> the optional cross-checks in [`tools/`](tools/).

## 2. Data

Download [VSPW 480p](https://github.com/sssdddwww2/vspw_dataset_download) and
symlink it in:

```bash
mkdir -p data/vspw
ln -s /path/to/VSPW_480p data/vspw/VSPW_480p
```

The layout DiTTA expects is the one the dataset ships with:

```
data/vspw/VSPW_480p/
├── val.txt                       # one video name per line (343 for val)
└── data/<video>/
    ├── origin/*.jpg              # frames
    └── mask/*.png                # ground truth, used for evaluation only
```

## 3. Checkpoints

| file | what it is | where to get it |
|---|---|---|
| `checkpoints/ditta_segformer_b5_vspw.pth` | SegFormer MiT-B5 trained on the VSPW train split, frame-wise (the ISS model DiTTA adapts) | [Google Drive](https://drive.google.com/file/d/1Afjx_3C1FlxMnQ4C5H8-5MpP_q2ek2nE) (330 MB) |
| `checkpoints/sam2_hiera_large.pt` | stock SAM2 weights | the SAM2 repo (see above) |

Download the ISS checkpoint and put it in `checkpoints/`.

To convert your own ISS checkpoint instead, use
[`tools/prepare_checkpoint.py`](tools/prepare_checkpoint.py):

```bash
python tools/prepare_checkpoint.py \
    --src /path/to/your_segformer_b5_vspw.pth \
    --out checkpoints/ditta_segformer_b5_vspw.pth
```

DiTTA's decode head adds a temporal add-on (`query_layer`, `key_layer`) and a
projection head (`proj_feat`) that no ISS checkpoint contains. The script writes
their initialisation into the checkpoint so a run is reproducible without
depending on RNG order; the released checkpoint carries the exact initialisation
used for the paper's experiments.

## 4. Running

```bash
# full VSPW val split, 10% warm-up (the paper's main setting)
python run_ditta.py

# a quick smoke test on the first three videos
python run_ditta.py --videos 3

# one specific video
python run_ditta.py --video 127_-hIVCYO4C90
```

Useful flags: `--out <dir>` (default `exps/<timestamp>_q10`), `--tta_lr`,
`--tta_iters`, `--warmup_ratio`, `--seed`, and `--cache_dir <dir>` to store the
stage-B distillation targets so a hyper-parameter sweep over stage C does not
recompute them. Everything else lives in
[`configs/vspw_b5_q10.py`](configs/vspw_b5_q10.py).

Each run writes `result.json` (overall metrics plus a per-video breakdown),
`log.txt`, and TensorBoard scalars to the output directory.

### Watching a run, and comparing runs

Metrics are logged per evaluated frame under the same tag names the original
code base used (`0_mIoU`, `1_fwIoU`, `0_mIoU_instant`, `VC8`, `VC16`,
`loss/*`), so runs of this repository and of the original can be read off the
same axes. Point TensorBoard at a directory holding several runs:

```bash
tensorboard --logdir exps/compare
```

`--tb_classwise` adds the 124 per-class IoU curves (this is what makes the
original event files gigabytes large); `--no_tensorboard` turns logging off.

To overlay a reference run inside the live run's own charts, export its curves
once and pass them in — they show up as `paper/<tag>` with the signed difference
as `delta/<tag>`:

```bash
python tools/export_reference_curves.py \
    --run /path/to/reference_tb_run --out reference/paper_q10.npz \
    --tb_out exps/compare/paper_q10
python run_ditta.py --reference reference/paper_q10.npz --out exps/compare/mine
```

### Expected results

VSPW val, SegFormer-B5 ISS model, 10% warm-up (Table 1 of the paper):

| method | mIoU | wIoU | mVC₈ | mVC₁₆ |
|---|---|---|---|---|
| SegFormer (ISS baseline) | 49.0 | 66.3 | 88.3 | 84.3 |
| **DiTTA (ours)** | **51.1** | **66.5** | **94.1** | **92.2** |

A full run takes roughly 5 hours on a single GPU (~55 s per video), and needs no
disk space beyond the outputs.

## 5. Repository layout

```
run_ditta.py               the single entry point
configs/vspw_b5_q10.py     all hyper-parameters
ditta/
├── pipeline.py            the four stages, per video
├── segformer.py           DiTTA head + segmentor (mmcv/mmseg-free)
├── mit_backbone.py        SegFormer MiT-B5 backbone (vendored)
├── data.py                VSPW class metadata and the clip loader
├── sam2_target.py         SAM2 subclass + distillation-target construction
├── tta.py                 test-time adaptation losses and loop
├── metrics.py             mIoU / wIoU / mVC
├── tblog.py               TensorBoard logging and reference comparison
├── cache.py               optional target cache
└── config.py              config loading
tools/
├── prepare_checkpoint.py  ISS checkpoint -> DiTTA checkpoint
├── export_reference_curves.py     TensorBoard run -> compact reference curves
├── verify_against_legacy.py       cross-checks against the original code base
└── dev_extract_paper_addon_init.py  how the released add-on init was produced
```

## Citation

```bibtex
@inproceedings{kim2026ditta,
  title     = {Bootstrapping Video Semantic Segmentation Model via
               Distillation-assisted Test-Time Adaptation},
  author    = {Kim, Jihun and Kwon, Hoyong and Kweon, Hyeokjun and Yoon, Kuk-Jin},
  booktitle = {CVPR},
  year      = {2026}
}
```

## Acknowledgements

See [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md) for the third-party code this
repository builds on and the licences that apply to it.
